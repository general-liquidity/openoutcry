"""Avellaneda-Stoikov market-making env with a closed-form optimal baseline.

A canonical single-asset market-making environment: the agent quotes a bid and an ask
around a mid that follows a seeded arithmetic random walk, earns the spread on filled
market-order arrivals, and carries inventory risk (a squared running penalty plus a
forced terminal liquidation at an unfavorable price). Shipped alongside is the
Avellaneda-Stoikov *analytically optimal* quoting policy as a committed ground-truth
baseline, so a learner can be scored on **regret versus a provable optimum** rather than
on relative ranking alone.

The model (Avellaneda & Stoikov 2008):

* Mid follows arithmetic Brownian motion ``dS = sigma dW``.
* Per step, market orders arrive ``~Poisson(lambda)`` on each side; an arrival fills the
  maker's quote at depth ``delta`` with probability ``exp(-kappa * delta)``.
* The maker's reservation (indifference) price skews with inventory ``q``:
  ``r = s - q * gamma * sigma**2 * tau`` (``tau`` = remaining time).
* The optimal total spread is ``gamma*sigma**2*tau + (2/gamma)*ln(1 + gamma/kappa)``,
  i.e. an optimal half-spread ``delta* = gamma*sigma**2*tau/2 + (1/gamma)*ln(1+gamma/kappa)``
  quoted symmetrically around ``r`` — so the inventory-skewed quote depths are
  ``bid_depth = delta* + q*gamma*sigma**2*tau`` and ``ask_depth = delta* - q*gamma*sigma**2*tau``.

Pure Python (numpy): every draw comes from a seeded ``np.random.default_rng`` so an episode
is reproducible given its seed. This is a NEW training surface, deterministic-given-seed in
Python — NOT part of the cross-runtime byte-identical scored core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

Action = np.ndarray
Policy = Callable[[dict], Action]


@dataclass(frozen=True)
class MMParams:
    """Frozen Avellaneda-Stoikov market-making parameters.

    ``sigma``/``gamma``/``kappa`` are the A-S volatility, risk-aversion and order-book
    depth-decay constants; ``arrival_rate`` is the per-side Poisson market-order intensity
    per unit time; ``n_steps``/``dt`` set the horizon (total time ``n_steps*dt``). The
    closed-form optimal policy is a pure function of these, so the same params instance
    feeds both the env and :func:`analytically_optimal_policy`.
    """

    sigma: float = 2.0
    gamma: float = 0.1
    kappa: float = 1.5
    arrival_rate: float = 140.0
    n_steps: int = 200
    dt: float = 0.005
    s0: float = 100.0
    inventory_cap: int = 50
    phi: float = 0.0015
    terminal_liq_penalty: float = 0.2
    max_depth: float = 5.0
    tick_size: float = 0.01
    initial_cash: float = 0.0

    @property
    def horizon(self) -> float:
        return self.n_steps * self.dt


def _optimal_depths(q: float, tau: float, p: MMParams) -> tuple[float, float]:
    """Avellaneda-Stoikov optimal ``(bid_depth, ask_depth)`` at inventory ``q``, remaining
    time ``tau``. Half-spread widens with time-to-go and skews by inventory; depths are
    clipped to the env's quotable ``[0, max_depth]`` band."""
    skew = q * p.gamma * p.sigma**2 * tau
    half = 0.5 * p.gamma * p.sigma**2 * tau + (1.0 / p.gamma) * math.log1p(p.gamma / p.kappa)
    bid = min(max(half + skew, 0.0), p.max_depth)
    ask = min(max(half - skew, 0.0), p.max_depth)
    return bid, ask


class MarketMakingEnv(gym.Env):
    """Single-asset Avellaneda-Stoikov market-making env.

    State: inventory ``q``, cash, mid price (a seeded arithmetic random walk) and time
    remaining. Action is ``(bid_depth, ask_depth)`` quote distances from mid (price units,
    a multiple of ``tick_size``). Each step draws Poisson market-order arrivals per side and
    fills the maker's quote with probability ``exp(-kappa*depth)``, updating inventory under
    a hard cap ``+/-inventory_cap``. Reward is the mark-to-mid value change minus a running
    squared-inventory penalty ``phi*q**2``; the terminal step force-liquidates remaining
    inventory at an unfavorable price.
    """

    metadata = {"render_modes": []}

    def __init__(self, params: Optional[MMParams] = None, **overrides) -> None:
        super().__init__()
        self.params = replace(params or MMParams(), **overrides)
        p = self.params
        self.action_space = spaces.Box(
            low=0.0, high=p.max_depth, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "inventory": spaces.Box(
                    low=-p.inventory_cap, high=p.inventory_cap, shape=(1,), dtype=np.float64
                ),
                "mid": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64),
                "time_remaining": spaces.Box(
                    low=0.0, high=p.horizon, shape=(1,), dtype=np.float64
                ),
                "cash": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self._rng = np.random.default_rng(0)
        self._q = 0
        self._cash = p.initial_cash
        self._mid = p.s0
        self._t = 0

    # -- internal helpers --------------------------------------------------

    def _obs(self) -> dict:
        p = self.params
        return {
            "inventory": np.array([float(self._q)], dtype=np.float64),
            "mid": np.array([self._mid], dtype=np.float64),
            "time_remaining": np.array([(p.n_steps - self._t) * p.dt], dtype=np.float64),
            "cash": np.array([self._cash], dtype=np.float64),
        }

    def _value(self) -> float:
        return self._cash + self._q * self._mid

    # -- gymnasium API -----------------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        p = self.params
        self._q = 0
        self._cash = p.initial_cash
        self._mid = p.s0
        self._t = 0
        info = {"value": self._value(), "inventory": self._q}
        return self._obs(), info

    def step(self, action: Action) -> tuple[dict, float, bool, bool, dict]:
        p = self.params
        bid_depth, ask_depth = (
            float(np.clip(action[0], 0.0, p.max_depth)),
            float(np.clip(action[1], 0.0, p.max_depth)),
        )
        value_before = self._value()

        # Market-order arrivals per side, then a fill draw per arrival at prob exp(-k*depth).
        lam = p.arrival_rate * p.dt
        n_buy_orders = int(self._rng.poisson(lam))   # market BUYs lift the maker's ask
        n_sell_orders = int(self._rng.poisson(lam))  # market SELLs hit the maker's bid
        ask_fills = int(self._rng.binomial(n_buy_orders, math.exp(-p.kappa * ask_depth)))
        bid_fills = int(self._rng.binomial(n_sell_orders, math.exp(-p.kappa * bid_depth)))

        # Hard inventory cap: cannot sell below -cap or buy above +cap.
        ask_fills = min(ask_fills, self._q + p.inventory_cap)
        bid_fills = min(bid_fills, p.inventory_cap - self._q)

        ask_price = self._mid + ask_depth
        bid_price = self._mid - bid_depth
        self._cash += ask_fills * ask_price - bid_fills * bid_price
        self._q += bid_fills - ask_fills

        # Mid advances as arithmetic Brownian motion.
        self._mid += p.sigma * math.sqrt(p.dt) * float(self._rng.standard_normal())
        self._t += 1

        value_after = self._value()
        reward = (value_after - value_before) - p.phi * self._q**2

        terminated = self._t >= p.n_steps
        liquidated = 0.0
        if terminated and self._q != 0:
            # Forced liquidation crosses the spread at an unfavorable price.
            sign = 1.0 if self._q > 0 else -1.0
            liq_price = self._mid - sign * p.terminal_liq_penalty
            proceeds = self._q * liq_price
            reward += proceeds - self._q * self._mid
            self._cash += proceeds
            liquidated = float(self._q)
            self._q = 0

        info = {
            "value": self._value(),
            "inventory": self._q,
            "bid_fills": bid_fills,
            "ask_fills": ask_fills,
            "mid": self._mid,
            "liquidated": liquidated,
        }
        return self._obs(), float(reward), bool(terminated), False, info

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        return None


# -- policies ---------------------------------------------------------------


def analytically_optimal_policy(env_params: MMParams) -> Policy:
    """The Avellaneda-Stoikov closed-form optimal quoting policy as a ``(obs)->action``
    callable — the ground-truth optimum this benchmark scores regret against.

    Reservation price ``r = s - q*gamma*sigma**2*tau`` and optimal half-spread
    ``delta* = gamma*sigma**2*tau/2 + (1/gamma)*ln(1+gamma/kappa)`` give inventory-skewed
    depths ``bid = delta*+q*gamma*sigma**2*tau``, ``ask = delta*-q*gamma*sigma**2*tau``.
    """
    p = env_params

    def policy(obs: dict) -> Action:
        q = float(np.asarray(obs["inventory"]).reshape(-1)[0])
        tau = float(np.asarray(obs["time_remaining"]).reshape(-1)[0])
        bid, ask = _optimal_depths(q, tau, p)
        return np.array([bid, ask], dtype=np.float32)

    return policy


def fixed_spread_policy(half_spread: float) -> Policy:
    """A naive symmetric fixed-spread maker: quotes ``half_spread`` on both sides, ignoring
    inventory and time-to-go. The reference suboptimal policy regret is measured against."""

    def policy(obs: dict) -> Action:
        return np.array([half_spread, half_spread], dtype=np.float32)

    return policy


# -- regret metric ----------------------------------------------------------


def _rollout_reward(env: MarketMakingEnv, policy: Policy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        if terminated or truncated:
            return total


def mm_regret(
    policy: Policy,
    *,
    params: Optional[MMParams] = None,
    n_episodes: int = 16,
    seed_base: int = 0,
) -> float:
    """Mean reward gap between :func:`analytically_optimal_policy` and ``policy`` over
    ``n_episodes`` seeded episodes — the regret-vs-optimal metric. Both policies run on the
    *same* seeds, so the optimal scores ~0 regret against itself and a suboptimal policy
    scores a positive gap.
    """
    p = params or MMParams()
    optimal = analytically_optimal_policy(p)
    env = MarketMakingEnv(p)
    gap = 0.0
    for i in range(n_episodes):
        seed = seed_base + i
        opt_r = _rollout_reward(env, optimal, seed)
        pol_r = _rollout_reward(env, policy, seed)
        gap += opt_r - pol_r
    return gap / n_episodes


__all__ = [
    "MMParams",
    "MarketMakingEnv",
    "analytically_optimal_policy",
    "fixed_spread_policy",
    "mm_regret",
]
