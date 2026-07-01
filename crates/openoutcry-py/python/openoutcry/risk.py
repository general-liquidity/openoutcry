"""Risk-termination and circuit-breaker wrappers over :class:`~openoutcry.gym.OpenOutcryEnv`.

The base env terminates only at bankruptcy (``nav <= 0``); every other episode end is a
truncation (the point-in-time window ran out). These wrappers add two configurable,
deterministic, leak-free risk overlays on top of that contract:

* :class:`DrawdownStopper` cuts the episode when equity falls a fixed fraction below its
  running peak (or initial) NAV — a deterministic liquidation proxy. It sets
  ``truncated``, **not** ``terminated``. A stop-out is an episode *cut*, not an absorbing
  MDP state: the world did not end, the operator pulled the position. The RL consequence is
  the standard bootstrapping rule — a value estimate bootstraps past a truncated step
  (there is a future, it was merely not observed) but not past a terminated one. Marking a
  risk stop as ``terminated`` would teach the critic that drawdown is a zero-future
  absorbing state, which is wrong and biases the value function.

* :class:`TurbulenceHalt` is a forced-flat circuit-breaker (FinRL pattern, re-derived
  point-in-time): when a trailing turbulence signal exceeds a threshold it overrides the
  action to flat (zeros) before stepping. It does **not** end the episode — the agent stays
  in the market flat and resumes once turbulence subsides.

* :class:`CrossSectionalDeleverage` is a *cross-sectional* circuit-breaker, distinct from
  the single-asset :class:`TurbulenceHalt`: when a large fraction of the universe is
  simultaneously oversold (per-symbol RSI below a floor) it vetoes new leverage into the
  oversold names — a broad-breadth "everything is selling off together" deleverage trigger.
  It does **not** end the episode; it forces the oversold subset (or all symbols) flat for
  that bar and surfaces ``info["deleverage_veto"]``.

The NAV wrappers read only past NAVs (``info["nav"]`` from prior steps); the cross-sectional
breaker reads only past-and-present ``closes`` (accumulated into a per-episode buffer cleared
on ``reset``, the :mod:`openoutcry.indicators` pattern). None can leak a future bar into a
present decision.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .indicators import _rsi

_EPS = 1e-8


class DrawdownStopper(gym.Wrapper):
    """Stop the episode out when NAV draws down past ``max_drawdown`` of its reference.

    The reference is the running peak NAV (``mode="peak"``) or the first NAV of the episode
    (``mode="initial"``). When ``nav <= (1 - max_drawdown) * reference`` the step is flagged
    ``truncated=True`` and ``info["stopped_out"]=True``. ``terminated`` is left to the base
    env (bankruptcy). Peak tracking is pure running-max over observed NAVs, so the stop-out
    step is a deterministic function of the NAV path.
    """

    def __init__(
        self, env: gym.Env, *, max_drawdown: float = 0.5, mode: str = "peak"
    ) -> None:
        super().__init__(env)
        if mode not in ("peak", "initial"):
            raise ValueError("mode must be 'peak' or 'initial'")
        if not 0.0 <= max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be in [0, 1]")
        self._max_drawdown = float(max_drawdown)
        self._mode = mode
        self._peak: Optional[float] = None
        self._initial: Optional[float] = None

    def reset(self, **kwargs: Any):
        self._peak = None
        self._initial = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        nav = info.get("nav")
        if nav is None:
            return obs, reward, bool(terminated), bool(truncated), info
        nav = float(nav)
        if self._initial is None:
            self._initial = nav
        if self._peak is None or nav > self._peak:
            self._peak = nav
        reference = self._peak if self._mode == "peak" else self._initial
        if nav <= (1.0 - self._max_drawdown) * reference:
            truncated = True
            info = dict(info)
            info["stopped_out"] = True
        return obs, reward, bool(terminated), bool(truncated), info


class TurbulenceHalt(gym.Wrapper):
    """Force the action flat when a trailing turbulence signal exceeds ``threshold``.

    Maintains a rolling buffer of the last ``window`` per-bar returns (from successive
    ``info["nav"]``). The turbulence signal is the squared standardized most-recent return
    over that trailing window — the single-asset reduction of the FinRL Mahalanobis
    turbulence ``(r-μ)Σ⁻¹(r-μ)ᵀ`` to ``((r-μ)/σ)²`` (no matrix inverse). It is computed from
    realized past returns only, so it is point-in-time. When the signal exceeds
    ``threshold`` the action handed to ``env.step`` is overridden to zeros and
    ``info["turbulence_halt"]=True`` is surfaced. This is a circuit-breaker, not a
    termination: the episode continues, flat, until turbulence subsides.
    """

    def __init__(
        self, env: gym.Env, *, window: int = 20, threshold: float = 3.0
    ) -> None:
        super().__init__(env)
        if window < 2:
            raise ValueError("window must be >= 2")
        self._window = int(window)
        self._threshold = float(threshold)
        self._returns: deque[float] = deque(maxlen=self._window)
        self._prev_nav: Optional[float] = None

    def reset(self, **kwargs: Any):
        self._returns.clear()
        self._prev_nav = None
        return self.env.reset(**kwargs)

    def _turbulence(self) -> Optional[float]:
        if len(self._returns) < 2:
            return None
        arr = np.asarray(self._returns, dtype=np.float64)
        std = float(arr.std())
        z = (float(arr[-1]) - float(arr.mean())) / (std + _EPS)
        return z * z

    def step(self, action):
        signal = self._turbulence()
        halt = signal is not None and signal > self._threshold
        executed = np.zeros_like(np.asarray(action)) if halt else action
        obs, reward, terminated, truncated, info = self.env.step(executed)
        nav = info.get("nav")
        if nav is not None:
            nav = float(nav)
            if self._prev_nav is not None and self._prev_nav != 0.0:
                self._returns.append(nav / self._prev_nav - 1.0)
            self._prev_nav = nav
        if halt:
            info = dict(info)
            info["turbulence_halt"] = True
        return obs, reward, bool(terminated), bool(truncated), info


class CrossSectionalDeleverage(gym.Wrapper):
    """Veto new leverage when the universe is *broadly* oversold — a cross-sectional breaker.

    Each bar the wrapper computes a per-symbol RSI (Cutler's, reused from
    :mod:`openoutcry.indicators`) from a causal per-episode buffer of the ``closes`` the base
    env emits, then counts how many symbols are simultaneously oversold (``rsi < oversold``).
    When that count is at least ``min_oversold`` **and** at least ``fraction`` of the universe,
    the breaker fires: the executed action is forced flat on the oversold subset (``scope=
    "subset"``) or on every symbol (``scope="all"``) before stepping, and
    ``info["deleverage_veto"]=True`` is surfaced along with ``info["deleverage_oversold"]``
    (the sorted oversold symbol indices) and ``info["deleverage_scope"]``.

    This is distinct from :class:`TurbulenceHalt`, a single-asset NAV-turbulence halt: the
    trigger here is *breadth* — many names selling off at once — the classic forced-deleverage
    condition, so it vetoes crowding into a market-wide washout rather than reacting to one
    portfolio's volatility.

    Leak-free by construction: the RSI at bar ``t`` is computed from the buffer of closes at
    bars ``0..t-1`` (seeded at ``reset`` with the first observation, extended after each step),
    so the veto decision never peeks at the bar it is about to step into. Until an RSI window
    has filled the breaker cannot fire (no symbol is scored oversold during warmup). A pure
    function of the close path, hence deterministic.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        rsi_window: int = 14,
        oversold: float = 30.0,
        fraction: float = 0.60,
        min_oversold: int = 3,
        scope: str = "subset",
    ) -> None:
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), (
            "CrossSectionalDeleverage expects a Dict observation space"
        )
        assert "closes" in env.observation_space.spaces, (
            "CrossSectionalDeleverage requires a 'closes' obs key"
        )
        if rsi_window < 2:
            raise ValueError("rsi_window must be >= 2")
        if not 0.0 < oversold < 100.0:
            raise ValueError("oversold must be in (0, 100)")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        if min_oversold < 1:
            raise ValueError("min_oversold must be >= 1")
        if scope not in ("subset", "all"):
            raise ValueError("scope must be 'subset' or 'all'")
        self._rsi_window = int(rsi_window)
        self._oversold = float(oversold)
        self._fraction = float(fraction)
        self._min_oversold = int(min_oversold)
        self._scope = scope
        self._n = int(env.observation_space.spaces["closes"].shape[-1])
        self._history: list[np.ndarray] = []

    def reset(self, **kwargs: Any):
        self._history = []
        obs, info = self.env.reset(**kwargs)
        self._history.append(
            np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        )
        return obs, info

    def _oversold_symbols(self) -> list[int]:
        """Indices of symbols whose causal RSI is below the oversold floor (empty on warmup)."""
        if len(self._history) < self._rsi_window + 1:
            return []
        hist = np.stack(self._history, axis=0)  # (T, n_symbols), oldest-first
        return [
            s
            for s in range(self._n)
            if _rsi(hist[:, s], self._rsi_window) < self._oversold
        ]

    def step(self, action):
        oversold = self._oversold_symbols()
        fires = (
            len(oversold) >= self._min_oversold
            and len(oversold) >= self._fraction * self._n
        )
        if fires and self._scope == "all":
            executed = np.zeros_like(np.asarray(action))
        elif fires:
            executed = np.array(action, copy=True)
            executed[oversold] = 0.0
        else:
            executed = action
        obs, reward, terminated, truncated, info = self.env.step(executed)
        self._history.append(
            np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        )
        if fires:
            info = dict(info)
            info["deleverage_veto"] = True
            info["deleverage_oversold"] = list(oversold)
            info["deleverage_scope"] = self._scope
        return obs, reward, bool(terminated), bool(truncated), info


__all__ = ["DrawdownStopper", "TurbulenceHalt", "CrossSectionalDeleverage"]
