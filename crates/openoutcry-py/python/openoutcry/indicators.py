"""Causal technical-indicator observation wrapper for :class:`OpenOutcryEnv`.

Every comparable trading env feeds the policy a stack of technical indicators, but
OpenOutcry's gym obs is deliberately near-raw (``closes``/``positions``/``cash``).
This wrapper adds an opt-in, config-listed indicator block — declared as a NAMED
indicator list (the freqtrade pattern), never hardcoded.

**Causality (leak-free by construction).** The wrapper does *not* peek at the wire
``close_history``; it accumulates the per-step ``closes`` it is handed into its own
rolling buffer (cleared on every episode ``reset``) and computes every indicator
from that buffer. The block emitted at bar ``t`` therefore depends only on closes at
bars ``0..t`` — a future bar can never reach back. This also needs no change to
``gym.py``: the only input is the obs the env already emits.

**Warmup.** Until an indicator's window has filled, it emits the documented sentinel
``0.0`` (never a future bar). The first observation of an episode has a one-bar
history, so all windowed indicators read ``0.0`` there.

Indicators are single-purpose numpy (no TA-Lib dep). Each contributes one scalar per
symbol, so the block is a Box of shape ``(n_symbols, len(indicators))`` added to the
Dict obs under the ``"indicators"`` key (column order == ``indicators`` order).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Default EWMA spans for MACD (fast/slow/signal) — the canonical 12/26/9 contract.
_MACD_FAST, _MACD_SLOW, _MACD_SIGNAL = 12, 26, 9


def _ema_series(series: np.ndarray, span: int) -> np.ndarray:
    """Causal EWMA over ``series`` (oldest-first), seeded at the first value."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(series, dtype=np.float64)
    out[0] = series[0]
    for i in range(1, series.size):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(series: np.ndarray, window: int) -> float:
    """Cutler's RSI (SMA-seeded, not Wilder smoothing): simple mean of the last
    ``window`` up/down moves. Deterministic and strictly window-bounded."""
    if series.size < window + 1:
        return 0.0
    deltas = np.diff(series[-(window + 1):])
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _sma(series: np.ndarray, window: int) -> float:
    if series.size < window:
        return 0.0
    return float(series[-window:].mean())


def _ema(series: np.ndarray, window: int) -> float:
    if series.size < window:
        return 0.0
    return float(_ema_series(series, window)[-1])


def _macd(series: np.ndarray, window: int) -> float:
    """MACD histogram (macd line - signal line), 12/26/9 EWMA. ``window`` ignored —
    MACD carries its own canonical spans."""
    if series.size < _MACD_SLOW:
        return 0.0
    macd_line = _ema_series(series, _MACD_FAST) - _ema_series(series, _MACD_SLOW)
    signal = _ema_series(macd_line, _MACD_SIGNAL)
    return float(macd_line[-1] - signal[-1])


def _bollinger(series: np.ndarray, window: int) -> float:
    """Bollinger %B over a ``window``/2σ band: 0 at the lower band, 1 at the upper."""
    if series.size < window:
        return 0.0
    w = series[-window:]
    mean = float(w.mean())
    sd = float(w.std())
    if sd == 0.0:
        return 0.0
    lower = mean - 2.0 * sd
    upper = mean + 2.0 * sd
    return float((series[-1] - lower) / (upper - lower))


def _realized_vol(series: np.ndarray, window: int) -> float:
    """Std of the last ``window`` simple returns."""
    if series.size < window + 1:
        return 0.0
    tail = series[-(window + 1):]
    rets = np.diff(tail) / tail[:-1]
    return float(np.std(rets))


def _return(series: np.ndarray, window: int) -> float:
    if series.size < 2:
        return 0.0
    return float(series[-1] / series[-2] - 1.0)


# The named indicator registry (freqtrade-style declarative list). Add an entry to
# expose a new indicator; the wrapper resolves names against this table.
INDICATORS: dict[str, Callable[[np.ndarray, int], float]] = {
    "rsi": _rsi,
    "sma": _sma,
    "ema": _ema,
    "macd": _macd,
    "bollinger": _bollinger,
    "realized_vol": _realized_vol,
    "return": _return,
}

DEFAULT_INDICATORS: tuple[str, ...] = (
    "rsi",
    "sma",
    "ema",
    "macd",
    "bollinger",
    "realized_vol",
    "return",
)


class CausalIndicatorObservation(gym.ObservationWrapper):
    """Append a causal technical-indicator block to a Dict observation.

    Reads the per-step ``closes`` (shape ``(n_symbols,)``) the base env emits,
    accumulates them into a per-episode rolling buffer, and computes each named
    indicator per symbol from that buffer — strictly past-and-present closes, so the
    block is leak-free. Adds ``obs["indicators"]`` as a Box of shape
    ``(n_symbols, len(indicators))`` (column ``j`` == ``indicators[j]``).
    """

    def __init__(
        self,
        env: gym.Env,
        indicators: tuple[str, ...] = DEFAULT_INDICATORS,
        window: int = 20,
    ) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "CausalIndicatorObservation expects a Dict observation space"
        )
        assert "closes" in env.observation_space.spaces, (
            "CausalIndicatorObservation requires a 'closes' obs key"
        )
        unknown = [name for name in indicators if name not in INDICATORS]
        if unknown:
            raise ValueError(f"unknown indicators: {unknown}")
        if window < 1:
            raise ValueError("window must be >= 1")

        self._names: tuple[str, ...] = tuple(indicators)
        self._fns = [INDICATORS[name] for name in self._names]
        self._window = int(window)
        self._n = int(env.observation_space.spaces["closes"].shape[-1])
        self._history: list[np.ndarray] = []

        out_spaces = dict(env.observation_space.spaces)
        out_spaces["indicators"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._n, len(self._names)),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(out_spaces)

    @property
    def indicator_names(self) -> tuple[str, ...]:
        return self._names

    def observation(self, obs: dict) -> dict:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        self._history.append(closes)
        hist = np.stack(self._history, axis=0)  # (T, n_symbols), oldest-first
        block = np.zeros((self._n, len(self._fns)), dtype=np.float64)
        for j, fn in enumerate(self._fns):
            for s in range(self._n):
                block[s, j] = fn(hist[:, s], self._window)
        out = dict(obs)
        out["indicators"] = block
        return out

    def reset(self, **kwargs):
        self._history = []
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


__all__ = [
    "CausalIndicatorObservation",
    "INDICATORS",
    "DEFAULT_INDICATORS",
]
