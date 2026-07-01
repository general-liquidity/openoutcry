"""A trailing-window spread z-score observation for cointegrated-pair trading.

:class:`SpreadObservation` augments a Dict observation with a single
``spread_zscore`` feature for the first symbol pair (``closes[0]`` vs
``closes[1]``). The score is **leak-free by construction**: the value emitted at
bar ``t`` is a function of only the trailing ``window`` closes the wrapper has
accumulated up to *and including* ``t`` — no future bar can ever reach back, the
same property the causal wrappers in :mod:`openoutcry.wrappers` guarantee. The
wrapper keeps its own rolling buffer (it never reads the dataset), so it composes
over any point-in-time env without a ``gym.py`` change.

The score is built the way a pairs trader reads a spread: a rolling OLS regresses
``y = closes[1]`` on ``x = closes[0]`` over the buffer to estimate the hedge ratio
``beta``, forms the residual spread ``y - (alpha + beta*x)``, and z-scores its
latest value against the buffer's own mean/std. A large ``|z|`` is the entry
signal a mean-reverting (``cointegrated_pairs``-generated) spread produces.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_EPS = 1e-8


class SpreadObservation(gym.ObservationWrapper):
    """Add a trailing-window ``spread_zscore`` for the first symbol pair.

    Parameters
    ----------
    env:
        A point-in-time env whose observation space is a ``Dict`` with a 1-D
        ``closes`` key over at least two symbols.
    window:
        Trailing-buffer length (``>= 2``) the rolling OLS / z-score is computed
        over. The buffer resets on every episode boundary.
    key:
        The observation key the z-score is written under (default
        ``"spread_zscore"``).
    """

    def __init__(self, env: gym.Env, window: int = 20, key: str = "spread_zscore") -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "SpreadObservation expects a Dict observation space"
        )
        closes_space = env.observation_space.spaces.get("closes")
        assert (
            closes_space is not None
            and len(closes_space.shape) == 1
            and closes_space.shape[0] >= 2
        ), "SpreadObservation needs a 1-D 'closes' over >= 2 symbols"
        self._window = int(window)
        assert self._window >= 2, "window must be >= 2"
        self._key = key
        self._buf: deque[tuple[float, float]] = deque(maxlen=self._window)
        augmented = dict(env.observation_space.spaces)
        augmented[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Dict(augmented)

    def reset(self, **kwargs: Any):
        # Clear the rolling buffer before the base wrapper folds in the first obs.
        self._buf.clear()
        return super().reset(**kwargs)

    def observation(self, obs: dict) -> dict:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        self._buf.append((float(closes[0]), float(closes[1])))
        out = dict(obs)
        out[self._key] = np.array([self._spread_z()], dtype=np.float64)
        return out

    def _spread_z(self) -> float:
        """Z-score of the latest residual spread over the trailing buffer.

        Returns ``0.0`` for a history-free first bar, a degenerate (flat)
        regressor, or a zero-variance spread. Uses only mul/add/div/sqrt.
        """
        if len(self._buf) < 2:
            return 0.0
        arr = np.asarray(self._buf, dtype=np.float64)
        x, y = arr[:, 0], arr[:, 1]
        mx, my = float(x.mean()), float(y.mean())
        var_x = float(np.sum((x - mx) * (x - mx)))
        if var_x <= _EPS:
            return 0.0
        beta = float(np.sum((x - mx) * (y - my)) / var_x)
        alpha = my - beta * mx
        spread = y - (alpha + beta * x)
        sd = float(spread.std())
        if sd <= _EPS:
            return 0.0
        return float((spread[-1] - float(spread.mean())) / (sd + _EPS))


class KalmanSpreadObservation(gym.ObservationWrapper):
    """Add a recursive Kalman hedge-ratio spread + innovation z for the first pair.

    Where :class:`SpreadObservation` refits a rolling OLS every bar and then
    *approximates* a z-score against the buffer's own residual std, this wrapper
    tracks the hedge ratio with a delta-parameterized 2-state Kalman filter and
    emits the exact quantity a pairs trader wants: the one-step innovation (the
    residual spread against the pre-update hedge) and its **innovation-normalized**
    z-score ``innovation / sqrt(innovation_variance)`` — the standardized residual
    the filter itself produces, not a buffer approximation of it.

    The state is ``[beta, intercept]`` for the model ``closes[1] ≈ beta·closes[0] +
    intercept``. The transition is a random walk (``F = I``) and the process noise is
    Ernie Chan's delta parameterization ``Q = delta/(1-delta)·I`` — a single knob
    that trades adaptation speed (large ``delta``) against smoothness (small
    ``delta``). Because the filter is a pure forward recursion over the closes it is
    handed (it keeps only ``(beta, P)`` state, never a look-back buffer and never the
    dataset), the value at bar ``t`` is a function of bars ``0..t`` only — leak-free
    by construction, the same guarantee :class:`SpreadObservation` gives.

    Parameters
    ----------
    env:
        A point-in-time env whose observation space is a ``Dict`` with a 1-D
        ``closes`` key over at least two symbols.
    delta:
        Process-noise ratio in ``(0, 1)``; ``Q = delta/(1-delta)·I``. Smaller is
        smoother/slower to adapt (default ``1e-4``, Chan's value).
    obs_var:
        Measurement-noise variance ``Ve > 0`` (default ``1e-3``). Sets the floor of
        the innovation variance and hence the z-score scale.
    spread_key / z_key:
        Observation keys the innovation spread and its z-score are written under
        (defaults ``"kalman_spread"`` / ``"kalman_spread_z"``).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        delta: float = 1e-4,
        obs_var: float = 1e-3,
        spread_key: str = "kalman_spread",
        z_key: str = "kalman_spread_z",
    ) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "KalmanSpreadObservation expects a Dict observation space"
        )
        closes_space = env.observation_space.spaces.get("closes")
        assert (
            closes_space is not None
            and len(closes_space.shape) == 1
            and closes_space.shape[0] >= 2
        ), "KalmanSpreadObservation needs a 1-D 'closes' over >= 2 symbols"
        assert 0.0 < float(delta) < 1.0, "delta must be in (0, 1)"
        assert float(obs_var) > 0.0, "obs_var must be > 0"
        self._delta = float(delta)
        self._q = self._delta / (1.0 - self._delta)
        self._obs_var = float(obs_var)
        self._spread_key = spread_key
        self._z_key = z_key
        self._beta = np.zeros(2, dtype=np.float64)
        self._P = np.zeros((2, 2), dtype=np.float64)
        self._count = 0
        augmented = dict(env.observation_space.spaces)
        augmented[spread_key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        augmented[z_key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Dict(augmented)

    @property
    def hedge_ratio(self) -> float:
        """Current filtered hedge ratio ``beta`` (state component 0)."""
        return float(self._beta[0])

    @property
    def intercept(self) -> float:
        """Current filtered intercept (state component 1)."""
        return float(self._beta[1])

    def reset(self, **kwargs: Any):
        self._beta = np.zeros(2, dtype=np.float64)
        self._P = np.zeros((2, 2), dtype=np.float64)
        self._count = 0
        return super().reset(**kwargs)

    def observation(self, obs: dict) -> dict:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        spread, z = self._update(float(closes[0]), float(closes[1]))
        out = dict(obs)
        out[self._spread_key] = np.array([spread], dtype=np.float64)
        out[self._z_key] = np.array([z], dtype=np.float64)
        return out

    def _update(self, x_val: float, y_val: float) -> tuple[float, float]:
        """Advance the filter one bar; return ``(innovation, innovation_z)``.

        The first bar of an episode has no prior spread distribution, so it emits the
        documented history-free sentinel ``0.0`` (while still folding the bar into the
        state, matching the way the OLS wrapper's first bar is history-free zero).
        """
        H = np.array([x_val, 1.0], dtype=np.float64)
        P_pred = self._P + self._q * np.eye(2, dtype=np.float64)  # F = I
        innovation = y_val - float(H @ self._beta)
        innovation_var = float(H @ P_pred @ H) + self._obs_var
        gain = (P_pred @ H) / innovation_var
        self._beta = self._beta + gain * innovation
        self._P = P_pred - np.outer(gain, H @ P_pred)
        self._count += 1
        if self._count < 2:
            return 0.0, 0.0
        return innovation, innovation / np.sqrt(innovation_var)


__all__ = ["SpreadObservation", "KalmanSpreadObservation"]
