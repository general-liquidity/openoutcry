"""Point-in-time-SAFE vector wrappers over :class:`~openoutcry.vector.OpenOutcryVectorEnv`.

These mirror :mod:`gymnasium.wrappers.vector` for the leak-free transforms, but operate
against OpenOutcry's gym3 **same-step** autoreset contract (a finished lane is reset in
place on the same step, surfaced via ``infos["first"][i] == True``) rather than the 1.x
``NEXT_STEP`` default the stock vector wrappers assume. The stock wrappers also assert the
``autoreset_mode`` metadata is an :class:`~gymnasium.vector.AutoresetMode` enum; OpenOutcry
tags it as the string ``"same_step"`` and reports lane ends through ``first`` — so we
subclass :class:`~gymnasium.vector.VectorWrapper` directly and honour that contract.

The causal property is identical to the single-env wrappers: the value emitted for bar
``t`` of a lane is computed from that lane's bars ``0..t-1`` only; bar ``t`` is folded in
*after* emission, so no future bar reaches back. Per-lane statistics are **reset on the
autoreset boundary** (``first[i]``) so one episode's distribution never leaks into the
next on a recycled lane.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorWrapper, VectorEnv
from gymnasium.vector.utils import batch_space

_EPS = 1e-8


class _LaneWelford:
    """Per-lane online mean/variance (Welford) for one batched feature array.

    ``count`` is per-lane (shape ``(B,)``); ``mean``/``m2`` carry the feature shape per
    lane (shape ``(B, *feat)``). Lanes update and reset independently.
    """

    __slots__ = ("num_envs", "_ndim", "count", "mean", "m2")

    def __init__(self, num_envs: int, shape: tuple[int, ...]) -> None:
        self.num_envs = num_envs
        self._ndim = len(shape)
        self.count = np.zeros(num_envs, dtype=np.int64)
        self.mean = np.zeros((num_envs, *shape), dtype=np.float64)
        self.m2 = np.zeros((num_envs, *shape), dtype=np.float64)

    def std(self) -> np.ndarray:
        denom = np.maximum(self.count, 1).reshape((self.num_envs, *([1] * self._ndim)))
        return np.sqrt(self.m2 / denom)

    def reset_lanes(self, mask: np.ndarray) -> None:
        self.count[mask] = 0
        self.mean[mask] = 0.0
        self.m2[mask] = 0.0

    def update(self, x: np.ndarray, mask: np.ndarray) -> None:
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return
        xi = x[idx]
        self.count[idx] += 1
        c = self.count[idx].astype(np.float64).reshape((idx.size, *([1] * self._ndim)))
        delta = xi - self.mean[idx]
        self.mean[idx] += delta / c
        delta2 = xi - self.mean[idx]
        self.m2[idx] += delta * delta2


class VectorCausalNormalizeObservation(VectorWrapper):
    """Per-lane causal z-scoring of the batched Dict observation.

    Each lane keeps its own running mean/std per observation key, updated with **strictly
    past** bars of that lane's current episode. A lane that just autoreset (``first[i]``)
    has its statistics cleared *before* emission, so its new t0 is emitted as zeros and no
    prior-episode statistics leak across the boundary. Lanes with fewer than two bars of
    history emit zeros.
    """

    def __init__(self, env: VectorEnv) -> None:
        super().__init__(env)
        base = env.single_observation_space
        assert isinstance(base, spaces.Dict), (
            "VectorCausalNormalizeObservation expects a Dict single_observation_space"
        )
        self._stats: dict[str, _LaneWelford] = {
            k: _LaneWelford(self.num_envs, tuple(s.shape)) for k, s in base.spaces.items()
        }
        self.single_observation_space = spaces.Dict(
            {
                k: spaces.Box(low=-np.inf, high=np.inf, shape=s.shape, dtype=np.float64)
                for k, s in base.spaces.items()
            }
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def _apply(self, obs: dict, first: np.ndarray) -> dict:
        out: dict[str, np.ndarray] = {}
        for k, v in obs.items():
            x = np.asarray(v, dtype=np.float64)
            st = self._stats[k]
            if first.any():
                st.reset_lanes(first)
            res = np.zeros_like(x)
            valid = st.count >= 2
            if valid.any():
                vi = np.nonzero(valid)[0]
                std = st.std()
                res[vi] = (x[vi] - st.mean[vi]) / (std[vi] + _EPS)
            st.update(x, np.ones(self.num_envs, dtype=bool))
            out[k] = res
        return out

    def reset(self, *, seed: Any = None, options: Any = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        first = np.ones(self.num_envs, dtype=bool)
        return self._apply(obs, first), infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        first = np.asarray(
            infos.get("first", np.zeros(self.num_envs, dtype=bool)), dtype=bool
        )
        return self._apply(obs, first), rewards, terminated, truncated, infos


class VectorRecordEpisodeStatistics(VectorWrapper):
    """Inject per-lane episode statistics into ``infos`` when a lane ends.

    Tracks each lane's cumulative return and length; on the step a lane finishes
    (``terminated[i]`` or ``truncated[i]``) the gymnasium-standard ``episode`` block is
    written with that lane's ``r`` (return), ``l`` (length), ``t`` (wall-clock seconds)
    plus trading-specific ``sharpe`` and ``max_drawdown`` (mirroring the single-env
    :class:`~openoutcry.wrappers.RecordEpisodeStatistics`). ``infos["_episode"]`` is the
    boolean mask of lanes that ended this step; non-ending lanes hold zeros. The lane's
    accumulators reset after recording (its observation is already the new episode's t0
    under same-step autoreset).
    """

    def __init__(self, env: VectorEnv) -> None:
        super().__init__(env)
        self._rewards: list[list[float]] = [[] for _ in range(self.num_envs)]
        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._t0 = np.zeros(self.num_envs, dtype=np.float64)

    @staticmethod
    def _sharpe(rewards: np.ndarray) -> float:
        if rewards.size < 2:
            return 0.0
        sd = float(np.std(rewards))
        return float(np.mean(rewards) / sd) if sd > 0.0 else 0.0

    @staticmethod
    def _max_drawdown(rewards: np.ndarray) -> float:
        if rewards.size == 0:
            return 0.0
        equity = np.cumprod(1.0 + rewards)
        peak = np.maximum.accumulate(equity)
        return float(np.max((peak - equity) / peak))

    def reset(self, *, seed: Any = None, options: Any = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        now = time.perf_counter()
        self._rewards = [[] for _ in range(self.num_envs)]
        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._t0 = np.full(self.num_envs, now, dtype=np.float64)
        return obs, infos

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        for i in range(self.num_envs):
            self._rewards[i].append(float(rewards_arr[i]))
            self._lengths[i] += 1

        if done.any():
            now = time.perf_counter()
            ep_r = np.zeros(self.num_envs, dtype=np.float64)
            ep_l = np.zeros(self.num_envs, dtype=np.int64)
            ep_t = np.zeros(self.num_envs, dtype=np.float64)
            ep_sharpe = np.zeros(self.num_envs, dtype=np.float64)
            ep_mdd = np.zeros(self.num_envs, dtype=np.float64)
            for i in np.nonzero(done)[0]:
                series = np.asarray(self._rewards[i], dtype=np.float64)
                ep_r[i] = float(series.sum())
                ep_l[i] = int(self._lengths[i])
                ep_t[i] = float(now - self._t0[i])
                ep_sharpe[i] = self._sharpe(series)
                ep_mdd[i] = self._max_drawdown(series)
                self._rewards[i] = []
                self._lengths[i] = 0
                self._t0[i] = now
            infos = dict(infos)
            infos["episode"] = {
                "r": ep_r,
                "l": ep_l,
                "t": ep_t,
                "sharpe": ep_sharpe,
                "max_drawdown": ep_mdd,
            }
            infos["_episode"] = done
        return obs, rewards, terminated, truncated, infos


__all__ = [
    "VectorCausalNormalizeObservation",
    "VectorRecordEpisodeStatistics",
]
