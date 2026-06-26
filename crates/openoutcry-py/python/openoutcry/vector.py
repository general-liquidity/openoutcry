"""A gymnasium-vector wrapper around the native batched OpenOutcry binding.

:class:`OpenOutcryVectorEnv` adapts the JSON-at-the-boundary native ``VecTradingEnv``
(``B`` independent leak-free lanes) to the :class:`gymnasium.vector.VectorEnv` API:
``reset() -> (obs_batch, infos)`` and ``step(actions) -> (obs_batch, rewards,
terminated, truncated, infos)``, with numpy arrays of leading dim ``B``.

**Autoreset mode — gym3 same-step.** When a lane finishes (``terminated`` on a blow-up,
``truncated`` on running out of bars) the native engine resets that lane *in place* on
the same step, so ``obs_batch[i]`` is already the new episode's t0 and ``infos["first"][i]``
is ``True``. This is gymnasium's :class:`~gymnasium.vector.AutoresetMode.SAME_STEP`
convention (equivalently gym3's ``first`` flag), not the 1.x ``NEXT_STEP`` default — the
batch never stalls and there is no "do-nothing" reset step. The terminal observation is
not surfaced (consistent with gym3); ``rewards``/``infos`` describe the step that ended.

The action is a per-lane **target-weight vector** over the environment's symbols (shape
``(B, n_symbols)``), converted into the wire-format ``Decision`` JSON the binding expects.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from .openoutcry_py import VecTradingEnv

_BUY, _SELL, _HOLD = "buy", "sell", "hold"


def _action_label(weight: float) -> str:
    if weight > 0.0:
        return _BUY
    if weight < 0.0:
        return _SELL
    return _HOLD


class OpenOutcryVectorEnv(VectorEnv):
    """Vectorized gymnasium env over ``B`` leak-free, point-in-time market lanes.

    Pass either an explicit list of ``seeds`` (one synthetic scenario per seed) or
    ``num_envs`` (seeds become ``range(num_envs)``). All lanes share the panel shape,
    window and cost overrides. ``max_weight`` bounds the per-symbol target weight the
    action space allows (set ``allow_short=False`` to clip the lower bound to 0).
    """

    metadata = {"render_modes": [], "autoreset_mode": "same_step"}

    def __init__(
        self,
        num_envs: Optional[int] = None,
        *,
        seeds: Optional[Sequence[int]] = None,
        n_symbols: int = 4,
        n_days: int = 120,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
        max_weight: float = 1.0,
        allow_short: bool = True,
        env_kwargs: Optional[dict] = None,
    ) -> None:
        if seeds is None:
            if num_envs is None:
                raise ValueError("pass either num_envs or seeds")
            seeds = list(range(int(num_envs)))
        else:
            seeds = [int(s) for s in seeds]
            if num_envs is not None and num_envs != len(seeds):
                raise ValueError("num_envs must match len(seeds) when both are given")
        if not seeds:
            raise ValueError("seeds must be non-empty")

        self._seeds = list(seeds)
        self.num_envs = len(self._seeds)

        kwargs: dict[str, Any] = dict(env_kwargs or {})
        self._env = VecTradingEnv(
            seeds=self._seeds,
            n_symbols=n_symbols,
            n_days=n_days,
            window_start=window_start,
            window_end=window_end,
            **kwargs,
        )

        # Discover the symbol axis from the first lane's first observation (every lane
        # shares the same panel shape), so the spaces match the dataset exactly.
        first = json.loads(self._env.reset_batch())
        self._symbols = [s["symbol"] for s in first["observations"][0]["symbols"]]
        n = len(self._symbols)

        low = -max_weight if allow_short else 0.0
        self.single_action_space = spaces.Box(
            low=low, high=max_weight, shape=(n,), dtype=np.float32
        )
        self.single_observation_space = spaces.Dict(
            {
                "closes": spaces.Box(low=0.0, high=np.inf, shape=(n,), dtype=np.float64),
                "positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64
                ),
                "cash": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

    # -- internal helpers --------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def scenario_seeds(self) -> list[int]:
        return list(self._seeds)

    def _decode_obs(self, obs: dict) -> dict[str, np.ndarray]:
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

    def _stack_obs(self, observations: list[dict]) -> dict[str, np.ndarray]:
        decoded = [self._decode_obs(o) for o in observations]
        return {
            "closes": np.stack([d["closes"] for d in decoded]),
            "positions": np.stack([d["positions"] for d in decoded]),
            "cash": np.stack([d["cash"] for d in decoded]),
        }

    def _actions_to_decisions_json(self, actions: np.ndarray) -> str:
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, -1)
        decisions = [
            {
                "orders": [
                    {
                        "symbol": sym,
                        "action": _action_label(float(w)),
                        "target_weight": float(w),
                        "confidence": 0.5,
                    }
                    for sym, w in zip(self._symbols, lane)
                ],
                "reasoning": "OpenOutcryVectorEnv.step",
            }
            for lane in actions
        ]
        return json.dumps(decisions)

    # -- gymnasium vector API ----------------------------------------------

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        out = json.loads(self._env.reset_batch())
        obs = self._stack_obs(out["observations"])
        infos = {
            "scenario_seed": np.array(self._seeds, dtype=np.int64),
            "first": np.ones(self.num_envs, dtype=bool),
        }
        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        out = json.loads(self._env.step_batch(self._actions_to_decisions_json(actions)))
        obs = self._stack_obs(out["observations"])
        rewards = np.asarray(out["rewards"], dtype=np.float64)
        terminated = np.asarray(out["terminated"], dtype=bool)
        truncated = np.asarray(out["truncated"], dtype=bool)
        infos = {
            "scenario_seed": np.array(self._seeds, dtype=np.int64),
            "first": np.asarray(out["first"], dtype=bool),
            "nav": np.array([i["nav"] for i in out["infos"]], dtype=np.float64),
        }
        return obs, rewards, terminated, truncated, infos

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self, **kwargs):  # pragma: no cover
        return None
