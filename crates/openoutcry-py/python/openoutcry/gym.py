"""A gymnasium-compatible wrapper around the native OpenOutcry binding.

:class:`OpenOutcryEnv` adapts the JSON-at-the-boundary native ``TradingEnv`` to the
gymnasium 1.x API: ``reset() -> (obs, info)`` and ``step(action) -> (obs, reward,
terminated, truncated, info)``. The action is a **target-weight vector** over the
environment's symbols (one weight per symbol, in the observation's symbol order),
which the wrapper converts into the wire-format ``Decision`` JSON the binding
expects.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .openoutcry_py import TradingEnv

# Action label carried alongside the (signed) target weight. Sizing lives in the
# weight; the label is descriptive and scored for calibration on the Rust side.
_BUY, _SELL, _HOLD = "buy", "sell", "hold"


def _action_label(weight: float) -> str:
    if weight > 0.0:
        return _BUY
    if weight < 0.0:
        return _SELL
    return _HOLD


class OpenOutcryEnv(gym.Env):
    """Gymnasium env over a leak-free, point-in-time market.

    Parameters mirror the native binding's synthetic constructor; pass
    ``csv_text`` to build over a frozen CSV instead. ``max_weight`` bounds the
    per-symbol target weight the action space allows (set ``allow_short=False`` to
    clip the lower bound to 0).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_symbols: int = 4,
        n_days: int = 120,
        seed: int = 0,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
        csv_text: Optional[str] = None,
        max_weight: float = 1.0,
        allow_short: bool = True,
        env_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._seed = int(seed)
        self._n_symbols = n_symbols
        self._n_days = n_days
        self._window_start = window_start
        self._window_end = window_end
        self._csv_text = csv_text
        self._kwargs: dict[str, Any] = dict(env_kwargs or {})
        self._env = self._build_env(self._seed)

        # Discover the symbol axis from the first observation so the spaces match
        # the dataset exactly (works for both synthetic and CSV datasets).
        first = json.loads(self._env.reset())
        self._symbols = [s["symbol"] for s in first["symbols"]]
        n = len(self._symbols)

        low = -max_weight if allow_short else 0.0
        self.action_space = spaces.Box(
            low=low, high=max_weight, shape=(n,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "closes": spaces.Box(
                    low=0.0, high=np.inf, shape=(n,), dtype=np.float64
                ),
                "positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64
                ),
                "cash": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
                ),
            }
        )

    # -- internal helpers --------------------------------------------------

    def _build_env(self, seed: int) -> TradingEnv:
        """Construct the native env at ``seed``. For a synthetic dataset the seed
        selects the scenario (a different point-in-time price path); for a frozen CSV
        the path is fixed and the seed only varies execution noise."""
        if self._csv_text is not None:
            return TradingEnv.from_csv(
                self._csv_text,
                seed=seed,
                window_start=self._window_start,
                window_end=self._window_end,
                **self._kwargs,
            )
        return TradingEnv(
            n_symbols=self._n_symbols,
            n_days=self._n_days,
            seed=seed,
            window_start=self._window_start,
            window_end=self._window_end,
            **self._kwargs,
        )

    def _decode_obs(self, obs_json: str) -> dict[str, np.ndarray]:
        obs = json.loads(obs_json)
        by_symbol = {s["symbol"]: s for s in obs["symbols"]}
        closes = np.array(
            [by_symbol[sym]["close_history"][-1] for sym in self._symbols],
            dtype=np.float64,
        )
        pos = {p["symbol"]: p["shares"] for p in obs.get("portfolio", [])}
        positions = np.array(
            [pos.get(sym, 0.0) for sym in self._symbols], dtype=np.float64
        )
        return {
            "closes": closes,
            "positions": positions,
            "cash": np.array([obs["cash"]], dtype=np.float64),
        }

    def _action_to_decision_json(self, action: np.ndarray) -> str:
        weights = np.asarray(action, dtype=np.float64).reshape(-1)
        orders = [
            {
                "symbol": sym,
                "action": _action_label(float(w)),
                "target_weight": float(w),
                "confidence": 0.5,
            }
            for sym, w in zip(self._symbols, weights)
        ]
        return json.dumps({"orders": orders, "reasoning": "OpenOutcryEnv.step"})

    # -- gymnasium API -----------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def decision_space(self) -> dict[str, Any]:
        """A descriptive schema of the wire-contract ``Decision`` the binding consumes.

        This is the wire shape (what ``_action_to_decision_json`` emits and the native
        ``step`` parses), distinct from the gymnasium ``action_space`` (the
        target-weight ``Box``). Use it to validate or construct raw decisions.
        """
        return {
            "orders": {
                "symbol": "str (one of self.symbols)",
                "action": "str: buy | sell | hold | close",
                "target_weight": "float in [-1, 1] (signed for shorts)",
                "confidence": "float in [0, 1]",
                "rationale": "str (optional, default '')",
            },
            "reasoning": "str (optional, default '')",
        }

    def observation_space_schema(self) -> dict[str, Any]:
        """A descriptive schema of the wire-contract ``MarketObservation`` JSON the
        binding emits (the point-in-time shape decoded by ``_decode_obs``), distinct
        from the gymnasium ``observation_space`` ``Dict``."""
        return {
            "date": "str (ISO-8601 decision date)",
            "cash": "float",
            "symbols": {
                "symbol": "str",
                "close_history": "list[float] (trailing closes, oldest first)",
                "fundamentals": "dict[str, float] (optional)",
                "news": "list[str] (optional, headlines on/before date)",
            },
            "portfolio": {
                "symbol": "str",
                "shares": "float",
                "avg_price": "float",
            },
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Reset the episode. Passing an int ``seed`` selects the scenario: a synthetic
        env rebuilds on the new seed (a different point-in-time price path), so
        ``reset(seed=k)`` is reproducible and distinct seeds give distinct markets.
        ``seed=None`` keeps the current scenario (the gymnasium "seed once" paradigm)."""
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._env = self._build_env(self._seed)
        obs_json = self._env.reset()
        return self._decode_obs(obs_json), {"scenario_seed": self._seed}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Advance one bar.

        The engine's ``done`` means *ran out of bars* (end of the point-in-time
        window), which is **truncation**, not termination — so we map it to
        ``truncated``. True **termination** is bankruptcy (``nav <= 0``), an absorbing
        state. The distinction is the classic RL bootstrapping rule: a value estimate
        bootstraps past a ``truncated`` step (the episode was merely cut short) but not
        past a ``terminated`` one (there is no future).
        """
        decision_json = self._action_to_decision_json(action)
        obs_json, reward, done, info_json = self._env.step(decision_json)
        obs = self._decode_obs(obs_json)
        info = json.loads(info_json)
        info["scenario_seed"] = self._seed
        truncated = bool(done)
        terminated = float(info.get("nav", 1.0)) <= 0.0
        return obs, float(reward), terminated, truncated, info

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        return None
