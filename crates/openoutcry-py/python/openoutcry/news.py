"""Synthetic, leak-free news/sentiment observation channel for OpenOutcry.

The price-only observation gives an LLM agent nothing qualitative to reason over, so
"react to the news" objectives are ungradeable. :class:`SyntheticNewsObservation` adds an
optional per-symbol sentiment scalar in ``[-1, 1]`` (and optional templated headline
strings) derived **purely from the scenario seed**, never a live feed — so it stays
reproducible.

Leak-free argument. The sentiment shown at bar ``t`` for symbol ``j`` is
``f(scenario_seed, t - lag, j)`` via a SplitMix64 hash. It is computed up front from the
seed alone and never reads any close/return, so it cannot encode bar ``t``'s return. The
``lag`` (default 1) reveals the value that was already fully determined ``lag`` bars in the
past, so the channel is a function of (seed, t) revealed strictly before the move it hints
at — causal, and correlated-with-but-not-leaking the price path (which shares the seed).
``intensity=0`` collapses the channel to all zeros, restoring the price-only baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_HEADLINE_THRESHOLD = 0.5
_POSITIVE_HEADLINE = "raises guidance"
_NEGATIVE_HEADLINE = "guidance cut"


def _splitmix64(x: int) -> int:
    z = (x + _GAMMA) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _unit(seed: int, idx: int, sym: int) -> float:
    """A deterministic scalar in ``[-1, 1]`` from (seed, latent index, symbol index)."""
    h = _splitmix64(int(seed) & _MASK)
    h = _splitmix64(h ^ ((idx + 1) * _GAMMA & _MASK))
    h = _splitmix64(h ^ ((sym + 1) * 0x94D049BB133111EB & _MASK))
    return (h / float(1 << 64)) * 2.0 - 1.0


def _row(seed: int, idx: int, n_symbols: int, intensity: float) -> np.ndarray:
    if idx < 0 or intensity == 0.0:
        return np.zeros((n_symbols,), dtype=np.float32)
    vals = [np.clip(_unit(seed, idx, j) * intensity, -1.0, 1.0) for j in range(n_symbols)]
    return np.asarray(vals, dtype=np.float32)


def news_series(
    seed: int,
    n_steps: int,
    n_symbols: int,
    *,
    intensity: float = 1.0,
    lag: int = 1,
) -> np.ndarray:
    """The revealed sentiment series, shape ``(n_steps, n_symbols)`` in ``[-1, 1]``.

    Pure and env-free. Row ``t`` is the latent sentiment generated for index ``t - lag``
    (zeros where ``t < lag``), so ``news_series(..., lag=1)[1:] == news_series(..., lag=0)[:-1]``.
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    return np.stack(
        [_row(seed, t - lag, n_symbols, intensity) for t in range(n_steps)], axis=0
    )


class SyntheticNewsObservation(gym.Wrapper):
    """Add a leak-free ``obs["sentiment"]`` channel derived from the scenario seed.

    The channel is off the critical path of the price observation: with ``intensity=0`` it
    is all zeros and the wrapped obs matches the price-only baseline (plus the extra key).
    With ``headlines=True``, symbols whose revealed sentiment exceeds ``_HEADLINE_THRESHOLD``
    in magnitude contribute a short templated string to ``info["headlines"]`` — text only,
    never folded into the numeric observation.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        intensity: float = 1.0,
        lag: int = 1,
        headlines: bool = False,
    ) -> None:
        super().__init__(env)
        if lag < 0:
            raise ValueError("lag must be >= 0")
        assert isinstance(env.observation_space, spaces.Dict), (
            "SyntheticNewsObservation expects a Dict observation space"
        )
        self._intensity = float(intensity)
        self._lag = int(lag)
        self._headlines = bool(headlines)
        self._n = int(env.observation_space["closes"].shape[0])
        self._symbols = list(getattr(env.unwrapped, "symbols", range(self._n)))
        self._seed_val = int(getattr(env.unwrapped, "_seed", 0))
        self._t = 0
        self.observation_space = spaces.Dict(
            {
                **env.observation_space.spaces,
                "sentiment": spaces.Box(
                    low=-1.0, high=1.0, shape=(self._n,), dtype=np.float32
                ),
            }
        )

    def _sentiment(self) -> np.ndarray:
        return _row(self._seed_val, self._t - self._lag, self._n, self._intensity)

    def _headline_list(self, sentiment: np.ndarray) -> list[str]:
        out: list[str] = []
        for j, s in enumerate(sentiment):
            if abs(float(s)) < _HEADLINE_THRESHOLD:
                continue
            tmpl = _POSITIVE_HEADLINE if s > 0.0 else _NEGATIVE_HEADLINE
            out.append(f"{self._symbols[j]}: {tmpl}")
        return out

    def _augment(self, obs: dict, info: dict) -> tuple[dict, dict]:
        sentiment = self._sentiment()
        obs = {**obs, "sentiment": sentiment}
        if self._headlines:
            info = {**info, "headlines": self._headline_list(sentiment)}
        return obs, info

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._t = 0
        self._seed_val = int(info.get("scenario_seed", self._seed_val))
        return self._augment(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        obs, info = self._augment(obs, info)
        return obs, reward, terminated, truncated, info
