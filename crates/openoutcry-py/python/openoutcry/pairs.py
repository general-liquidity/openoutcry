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


__all__ = ["SpreadObservation"]
