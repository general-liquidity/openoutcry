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
        kwargs: dict[str, Any] = dict(env_kwargs or {})
        if csv_text is not None:
            self._env = TradingEnv.from_csv(
                csv_text,
                seed=seed,
                window_start=window_start,
                window_end=window_end,
                **kwargs,
            )
        else:
            self._env = TradingEnv(
                n_symbols=n_symbols,
                n_days=n_days,
                seed=seed,
                window_start=window_start,
                window_end=window_end,
                **kwargs,
            )

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

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        obs_json = self._env.reset()
        return self._decode_obs(obs_json), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        decision_json = self._action_to_decision_json(action)
        obs_json, reward, done, info_json = self._env.step(decision_json)
        obs = self._decode_obs(obs_json)
        info = json.loads(info_json)
        # gymnasium 1.x 5-tuple: (obs, reward, terminated, truncated, info).
        return obs, float(reward), bool(done), False, info

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        return None
