"""A multi-agent limit-order-book market over the native M3 matching engine.

:class:`LOBMarketEnv` is a PettingZoo ``ParallelEnv`` where ``n_agents`` market makers
post bid/ask quotes into one shared, deterministic integer-tick order book (the native
``PyOrderBook``: price-time priority, real fills). A seeded noise trader sends market
orders each step so quotes actually fill. Distinct from the bar-level position env and the
M2 endogenous (batch-clearing) market: here orders match against a real resting book.

**Leak-free.** An agent's observation is the post-step public depth ladder plus its own
inventory/cash; it never sees other agents' pending same-step orders (all quotes are
collected, then the book clears, then the next observation is produced).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - exercised only when pettingzoo is installed
    from pettingzoo import ParallelEnv

    _HAS_PZ = True
except Exception:  # noqa: BLE001
    ParallelEnv = object  # type: ignore[assignment,misc]
    _HAS_PZ = False

from .openoutcry_py import PyOrderBook

_MID_TICK = 1000  # the reference mid starts here (in ticks)


def _splitmix(state: int) -> tuple[int, float]:
    """One SplitMix64 draw -> (new_state, unit in [0, 1)); deterministic, no numpy RNG."""
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return state, (z >> 11) / float(1 << 53)


class LOBMarketEnv(ParallelEnv):  # type: ignore[misc]
    """N market makers quoting into one shared limit-order book.

    Each agent's action is a 2-vector ``[bid_offset, ask_offset]`` of ticks from the
    reference mid (clamped to ``[1, max_offset]``); it posts a buy at ``mid - bid_offset``
    and a sell at ``mid + ask_offset``, each of size ``quote_qty``. A seeded noise trader
    then sends a market order, the book clears, and reward is the change in mark-to-mid
    equity minus a squared-inventory penalty.
    """

    metadata = {"render_modes": [], "name": "openoutcry_lob_v0"}

    def __init__(
        self,
        n_agents: int = 2,
        *,
        n_steps: int = 120,
        seed: int = 0,
        tick_size: float = 0.01,
        levels: int = 5,
        quote_qty: int = 10,
        max_offset: int = 20,
        inventory_penalty: float = 0.001,
        noise_intensity: float = 2.0,
    ) -> None:
        if not _HAS_PZ:
            raise RuntimeError(
                "pettingzoo is not installed. Install the 'pettingzoo' extra to use "
                "LOBMarketEnv; the rest of the openoutcry package works without it."
            )
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        self._n_agents = int(n_agents)
        self._n_steps = int(n_steps)
        self._seed = int(seed)
        self._tick_size = float(tick_size)
        self._levels = int(levels)
        self._quote_qty = int(quote_qty)
        self._max_offset = int(max_offset)
        self._inv_pen = float(inventory_penalty)
        self._noise = float(noise_intensity)
        self.possible_agents = [f"agent_{i}" for i in range(self._n_agents)]

        from gymnasium import spaces

        self._obs_dim = 4 * self._levels + 5  # ladder (bids+asks) + mid/micro/imb + inv/cash
        self._obs_space = spaces.Box(-np.inf, np.inf, shape=(self._obs_dim,), dtype=np.float64)
        self._act_space = spaces.Box(
            low=1.0, high=float(self._max_offset), shape=(2,), dtype=np.float32
        )

    # -- PettingZoo API ----------------------------------------------------

    def observation_space(self, agent):  # noqa: D401
        return self._obs_space

    def action_space(self, agent):  # noqa: D401
        return self._act_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._seed = int(seed)
        self.agents = list(self.possible_agents)
        self._book = PyOrderBook(tick_size=self._tick_size, levels=self._levels)
        self._book.reset_book()
        self._mid = _MID_TICK
        self._rng = self._seed ^ 0x1234_5678_9ABC_DEF0
        self._step = 0
        self._inventory = {a: 0 for a in self.agents}
        self._cash = {a: 0.0 for a in self.agents}
        ladder = json.loads(self._book.ladder())
        obs = {a: self._obs(a, ladder) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict):
        # 1. every live agent posts a two-sided quote (collected before any clear).
        orders: list[dict] = []
        for i, a in enumerate(self.possible_agents):
            if a not in actions:
                continue
            bid_off, ask_off = (int(round(float(x))) for x in np.asarray(actions[a]).reshape(-1)[:2])
            bid_off = max(1, min(self._max_offset, bid_off))
            ask_off = max(1, min(self._max_offset, ask_off))
            orders.append({"agent": i, "kind": "limit", "side": "buy",
                           "price_tick": self._mid - bid_off, "qty": self._quote_qty})
            orders.append({"agent": i, "kind": "limit", "side": "sell",
                           "price_tick": self._mid + ask_off, "qty": self._quote_qty})

        # 2. a seeded noise trader sends one market order (agent id n_agents = exogenous).
        self._rng, u = _splitmix(self._rng)
        if u < 0.5 + 0.1 * self._noise:
            self._rng, u2 = _splitmix(self._rng)
            side = "buy" if u2 < 0.5 else "sell"
            self._rng, u3 = _splitmix(self._rng)
            qty = 1 + int(u3 * self._noise * self._quote_qty)
            orders.append({"agent": self._n_agents, "kind": "market", "side": side, "qty": qty})

        out = json.loads(self._book.step_book(json.dumps(orders)))
        ladder = out["ladder"]
        self._apply_fills(out["fills"], ladder)
        self._mid = self._next_mid(ladder)
        self._step += 1

        done = self._step >= self._n_steps
        obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}
        for a in self.agents:
            obs[a] = self._obs(a, ladder)
            rewards[a] = self._reward(a, ladder)
            terms[a] = False
            truncs[a] = done
            infos[a] = {"inventory": self._inventory[a], "cash": self._cash[a]}
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    # -- internals ---------------------------------------------------------

    def _apply_fills(self, fills, ladder) -> None:
        mid = ladder["mid"] or float(self._mid)
        for f in fills:
            price = f["price_tick"]
            qty = f["qty"]
            maker = self.possible_agents[f["maker_agent"]] if f["maker_agent"] < self._n_agents else None
            taker = self.possible_agents[f["taker_agent"]] if f["taker_agent"] < self._n_agents else None
            # maker side is the opposite of the taker side.
            if maker is not None:
                if f["taker_side"] == "buy":  # maker sold
                    self._inventory[maker] -= qty
                    self._cash[maker] += price * qty
                else:
                    self._inventory[maker] += qty
                    self._cash[maker] -= price * qty
            if taker is not None:
                if f["taker_side"] == "buy":
                    self._inventory[taker] += qty
                    self._cash[taker] -= price * qty
                else:
                    self._inventory[taker] -= qty
                    self._cash[taker] += price * qty

    def _equity(self, agent: str, mid: float) -> float:
        return self._cash[agent] + self._inventory[agent] * mid

    def _reward(self, agent: str, ladder) -> float:
        mid = ladder["mid"] or float(self._mid)
        eq = self._equity(agent, mid)
        prev = getattr(self, "_prev_equity", {}).get(agent, 0.0)
        if not hasattr(self, "_prev_equity"):
            self._prev_equity = {}
        self._prev_equity[agent] = eq
        return float(eq - prev - self._inv_pen * self._inventory[agent] ** 2)

    def _next_mid(self, ladder) -> int:
        # the reference mid follows the cleared microprice when available, else a seeded walk.
        if ladder["bids"] and ladder["asks"]:
            return int(round((ladder["bids"][0][0] + ladder["asks"][0][0]) / 2))
        self._rng, u = _splitmix(self._rng)
        return self._mid + (1 if u < 0.5 else -1)

    def _obs(self, agent: str, ladder) -> np.ndarray:
        bids = ladder["bids"][: self._levels]
        asks = ladder["asks"][: self._levels]
        vec = np.zeros(self._obs_dim, dtype=np.float64)
        for j, lvl in enumerate(bids):
            vec[2 * j] = lvl[0]
            vec[2 * j + 1] = lvl[1]
        base = 2 * self._levels
        for j, lvl in enumerate(asks):
            vec[base + 2 * j] = lvl[0]
            vec[base + 2 * j + 1] = lvl[1]
        tail = 4 * self._levels
        vec[tail] = ladder["mid"]
        vec[tail + 1] = ladder["microprice"]
        vec[tail + 2] = ladder["queue_imbalance"]
        vec[tail + 3] = self._inventory[agent]
        vec[tail + 4] = self._cash[agent]
        return vec


def symmetric_quote_policy(observation: Any = None, *, offset: int = 3) -> np.ndarray:
    """A fixed symmetric two-sided quote `offset` ticks from mid (the MM reference)."""
    return np.array([offset, offset], dtype=np.float32)


def noise_trader_policy(observation: Any = None, *, max_offset: int = 20) -> np.ndarray:
    """A wide, passive quote that rarely fills (a near-inactive reference)."""
    return np.array([max_offset, max_offset], dtype=np.float32)


__all__ = ["LOBMarketEnv", "symmetric_quote_policy", "noise_trader_policy"]
