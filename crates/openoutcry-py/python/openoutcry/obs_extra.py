"""Extra causal observation-feature wrappers for :class:`OpenOutcryEnv`.

Cheap, leak-free augmentations that OpenOutcry's near-raw obs lacks. Each reads the
per-step ``closes`` the base env already emits and accumulates them into a per-episode
rolling buffer (cleared on ``reset``) — exactly the pattern in
:mod:`openoutcry.indicators` — so the value emitted at bar ``t`` depends only on closes
at bars ``0..t`` and needs no change to ``gym.py``.

- :class:`MultiTimescaleMomentum` — vol-normalized momentum at several horizons.
- :class:`RollingCovarianceObservation` — trailing cross-asset return covariance.
- :class:`TimeToHorizonObservation` — fraction of the episode remaining (pure counter).
- :class:`CounterfactualInfo` — privileged ``info``-only "other action" channel.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _require_closes(env: gym.Env, who: str) -> int:
    assert isinstance(env.observation_space, spaces.Dict), (
        f"{who} expects a Dict observation space"
    )
    assert "closes" in env.observation_space.spaces, (
        f"{who} requires a 'closes' obs key"
    )
    return int(env.observation_space.spaces["closes"].shape[-1])


class MultiTimescaleMomentum(gym.ObservationWrapper):
    """Append vol-normalized momentum at each horizon per symbol.

    For horizon ``h`` the feature is ``(close_t/close_{t-h} - 1) / (vol·√h)`` where
    ``vol`` is the std of all available one-step returns in the causal buffer. Emitted
    as ``obs["momentum"]`` of shape ``(n_symbols, len(horizons))`` (column ``j`` ==
    ``horizons[j]``). Warmup-zeroed until a horizon's window and a 2-sample vol exist.
    """

    def __init__(
        self, env: gym.Env, *, horizons: tuple[int, ...] = (1, 5, 20, 60)
    ) -> None:
        super().__init__(env)
        self._n = _require_closes(env, "MultiTimescaleMomentum")
        horizons = tuple(int(h) for h in horizons)
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError("horizons must be non-empty positive ints")
        self._horizons = horizons
        self._history: list[np.ndarray] = []

        out_spaces = dict(env.observation_space.spaces)
        out_spaces["momentum"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n, len(horizons)), dtype=np.float64
        )
        self.observation_space = spaces.Dict(out_spaces)

    @property
    def horizons(self) -> tuple[int, ...]:
        return self._horizons

    @staticmethod
    def _momentum(series: np.ndarray, h: int) -> float:
        if series.size < h + 1:
            return 0.0
        rets = np.diff(series) / series[:-1]
        if rets.size < 2:
            return 0.0
        vol = float(np.std(rets))
        if vol == 0.0:
            return 0.0
        mom = float(series[-1] / series[-1 - h] - 1.0)
        return mom / (vol * np.sqrt(h))

    def observation(self, obs: dict) -> dict:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        self._history.append(closes)
        hist = np.stack(self._history, axis=0)  # (T, n_symbols), oldest-first
        block = np.zeros((self._n, len(self._horizons)), dtype=np.float64)
        for j, h in enumerate(self._horizons):
            for s in range(self._n):
                block[s, j] = self._momentum(hist[:, s], h)
        out = dict(obs)
        out["momentum"] = block
        return out

    def reset(self, **kwargs: Any):
        self._history = []
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class RollingCovarianceObservation(gym.ObservationWrapper):
    """Append the flattened upper-triangle of the trailing-window return covariance.

    Computes the covariance of one-step returns over the trailing ``window`` from the
    causal buffer and emits its upper triangle (incl. diagonal) as
    ``obs["covariance"]`` of shape ``(n·(n+1)/2,)``. For ``n == 1`` this is the single
    return variance. Warmup-zeroed until at least two returns are available.
    """

    def __init__(self, env: gym.Env, *, window: int = 20) -> None:
        super().__init__(env)
        self._n = _require_closes(env, "RollingCovarianceObservation")
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = int(window)
        self._iu = np.triu_indices(self._n)
        self._k = self._n * (self._n + 1) // 2
        self._history: list[np.ndarray] = []

        out_spaces = dict(env.observation_space.spaces)
        out_spaces["covariance"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._k,), dtype=np.float64
        )
        self.observation_space = spaces.Dict(out_spaces)

    def _cov(self, hist: np.ndarray) -> np.ndarray:
        if hist.shape[0] < 3:  # need >= 2 one-step returns
            return np.zeros(self._k, dtype=np.float64)
        tail = hist[-(self._window + 1):]
        rets = np.diff(tail, axis=0) / tail[:-1]
        cov = np.atleast_2d(np.cov(rets, rowvar=False))
        return cov[self._iu].astype(np.float64)

    def observation(self, obs: dict) -> dict:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        self._history.append(closes)
        hist = np.stack(self._history, axis=0)
        out = dict(obs)
        out["covariance"] = self._cov(hist)
        return out

    def reset(self, **kwargs: Any):
        self._history = []
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


class TimeToHorizonObservation(gym.Wrapper):
    """Append a scalar ``(max_steps - step)/max_steps`` — the fraction of the episode
    remaining. Pure step counter (reset to ``step = 0``), deterministic, clamped to
    ``[0, 1]``. Emitted as ``obs["time_to_horizon"]`` of shape ``(1,)``.
    """

    def __init__(self, env: gym.Env, *, max_steps: int) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "TimeToHorizonObservation expects a Dict observation space"
        )
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._max = int(max_steps)
        self._step = 0

        out_spaces = dict(env.observation_space.spaces)
        out_spaces["time_to_horizon"] = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float64
        )
        self.observation_space = spaces.Dict(out_spaces)

    def _augment(self, obs: dict) -> dict:
        frac = max(0.0, (self._max - self._step) / self._max)
        out = dict(obs)
        out["time_to_horizon"] = np.array([frac], dtype=np.float64)
        return out

    def reset(self, **kwargs: Any):
        self._step = 0
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        return self._augment(obs), reward, terminated, truncated, info


class CounterfactualInfo(gym.Wrapper):
    """Attach an "other-action" reward/NAV channel to ``info["counterfactual"]``.

    Each step, estimates what the untaken extreme actions (all-flat, all-long) would
    have produced over the same bar and writes them to ``info`` — never the obs, since
    this is privileged off-policy information. The estimate is the price-taker bar
    return ``Σ_i w_i·(close_t/close_{t-1} - 1)`` applied to the prior NAV; it is
    **exact only under the price-taker single-agent env** (where one agent's order does
    not move the bar's price), and is otherwise an approximation. Re-stepping the
    underlying env for an exact figure would mutate its state, so this stays an
    estimate (``info["counterfactual"]["estimate"] is True``).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._prev_closes: Optional[np.ndarray] = None
        self._prev_nav = 1.0

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._prev_closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1).copy()
        self._prev_nav = float(info.get("nav", 1.0))
        return obs, info

    def _bar_return(self, closes: np.ndarray) -> np.ndarray:
        prev = self._prev_closes
        if prev is None or prev.shape != closes.shape:
            return np.zeros_like(closes)
        return np.where(prev > 0.0, closes / prev - 1.0, 0.0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        r = self._bar_return(closes)
        high = np.asarray(self.action_space.high, dtype=np.float64).reshape(-1)
        flat_ret = 0.0
        long_ret = float(np.sum(high * r))
        nav = self._prev_nav

        info = dict(info)
        info["counterfactual"] = {
            "estimate": True,
            "note": "price-taker bar-return estimate; exact only for single-agent",
            "all_flat": {"reward": flat_ret, "nav": nav * (1.0 + flat_ret)},
            "all_long": {"reward": long_ret, "nav": nav * (1.0 + long_ret)},
        }
        self._prev_closes = closes.copy()
        self._prev_nav = float(info.get("nav", self._prev_nav))
        return obs, reward, terminated, truncated, info


__all__ = [
    "MultiTimescaleMomentum",
    "RollingCovarianceObservation",
    "TimeToHorizonObservation",
    "CounterfactualInfo",
]
