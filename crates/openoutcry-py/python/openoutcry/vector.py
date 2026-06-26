"""A gymnasium-vector wrapper around the native batched OpenOutcry binding.

:class:`OpenOutcryVectorEnv` adapts the JSON-at-the-boundary native ``VecTradingEnv``
(``B`` independent leak-free lanes) to the :class:`gymnasium.vector.VectorEnv` API:
``reset() -> (obs_batch, infos)`` and ``step(actions) -> (obs_batch, rewards,
terminated, truncated, infos)``, with numpy arrays of leading dim ``B``.

**Autoreset mode — selectable.** ``autoreset_mode`` chooses how a finished lane
(``terminated`` on a blow-up, ``truncated`` on running out of bars) is recycled:

- ``"next_step"`` (default, Gymnasium 1.x): the terminal step is returned verbatim and
  the reset surfaces on the *following* step (reward 0, flags ``False``, ``first`` ``True``).
- ``"same_step"`` (gym3): the lane resets *in place*, so ``obs_batch[i]`` is already the
  new episode's t0 and ``infos["first"][i]`` is ``True``; the terminal obs/info ride in
  ``infos["final_obs"][i]`` / ``infos["final_info"][i]``.
- ``"disabled"``: the lane never auto-resets and stays at its terminal bar.

Either way the batch never stalls; ``rewards``/``infos`` describe the step that executed.

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

try:  # gymnasium >= 1.0 exposes the AutoresetMode enum; fall back to the string label.
    from gymnasium.vector import AutoresetMode

    _AUTORESET_ENUM = {
        "next_step": AutoresetMode.NEXT_STEP,
        "same_step": AutoresetMode.SAME_STEP,
        "disabled": AutoresetMode.DISABLED,
    }
except Exception:  # noqa: BLE001
    _AUTORESET_ENUM = {}

from .openoutcry_py import VecTradingEnv

_BUY, _SELL, _HOLD = "buy", "sell", "hold"

# Eval scenarios live in a disjoint seed band (must match ``dataset.EVAL_SEED_BASE``).
_EVAL_SEED_BASE = 1_000_000


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

    metadata = {"render_modes": [], "autoreset_mode": "next_step"}

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
        distribution_mode: str = "calm",
        autoreset_mode: str = "next_step",
        mode: str = "train",
        max_episode_steps: Optional[int] = None,
        env_kwargs: Optional[dict] = None,
    ) -> None:
        # `max_episode_steps` is accepted for `gymnasium.make_vec` compatibility (it
        # forwards the spec value to the vector entry point); the engine already
        # truncates at the window end, so it is an advisory cap, not enforced here.
        del max_episode_steps
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if seeds is None:
            if num_envs is None:
                raise ValueError("pass either num_envs or seeds")
            base = _EVAL_SEED_BASE if mode == "eval" else 0
            seeds = list(range(base, base + int(num_envs)))
        else:
            seeds = [int(s) for s in seeds]
            if num_envs is not None and num_envs != len(seeds):
                raise ValueError("num_envs must match len(seeds) when both are given")
        if not seeds:
            raise ValueError("seeds must be non-empty")

        self._seeds = list(seeds)
        self.num_envs = len(self._seeds)
        self._autoreset_mode = autoreset_mode
        self.metadata = {
            **self.metadata,
            "autoreset_mode": _AUTORESET_ENUM.get(autoreset_mode, autoreset_mode),
        }

        kwargs: dict[str, Any] = dict(env_kwargs or {})
        self._env = VecTradingEnv(
            seeds=self._seeds,
            n_symbols=n_symbols,
            n_days=n_days,
            window_start=window_start,
            window_end=window_end,
            distribution_mode=distribution_mode,
            autoreset_mode=autoreset_mode,
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
        if self._autoreset_mode == "same_step":
            # SAME_STEP surfaces each finished lane's terminal obs/info alongside the
            # already-reset batch (None for lanes that did not finish this step).
            infos["final_obs"] = np.array(
                [self._decode_obs(o) if o is not None else None for o in out["final_obs"]],
                dtype=object,
            )
            infos["final_info"] = np.array(out["final_info"], dtype=object)
        return obs, rewards, terminated, truncated, infos

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self, **kwargs):  # pragma: no cover
        return None
