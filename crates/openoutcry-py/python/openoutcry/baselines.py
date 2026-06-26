"""Reference baseline policies and a leaderboard runner for OpenOutcry.

A benchmark is only credible once it ships *numbers to beat*. These are the
trivial reference policies every entrant must clear: a do-nothing ``flat``, a
buy-and-hold-analog ``equal_weight_long``, and a one-step ``momentum`` tilt. They
are deliberately tiny and numpy-only so the baseline table is cheap to reproduce
and impossible to argue with.

The leaderboard ranks on the SharpeBench **deflated Sharpe** plus the **pass^k**
rate, never raw return. Raw return-rank is luck: a policy can top a single seed by
drawing a lucky path. Deflation discounts for the breadth of the search (the number
of policies tried becomes the ``n_trials`` deflation count), and pass^k demands the
edge survive on every run, not on average.

Each policy is a fresh instance per episode, so stateful tilts (``momentum`` carries
the previous closes) reset cleanly at the start of every seed.
"""

from __future__ import annotations

import json
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .gym import OpenOutcryEnv
from .openoutcry_py import score_run

Policy = Callable[[dict], np.ndarray]


# -- reference policies -----------------------------------------------------


class FlatPolicy:
    """Do nothing. Holds zero target weight every step (the null hypothesis)."""

    name = "flat"

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.zeros((n,), dtype=np.float32)


class EqualWeightLongPolicy:
    """Buy-and-hold analog: equal positive target weight across all symbols."""

    name = "equal_weight_long"

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.full((n,), 1.0 / n, dtype=np.float32)


class MomentumPolicy:
    """One-step momentum: weight each symbol by the sign of its last close change.

    Warms up to equal-weight-long on the first observation (no prior close to
    difference against). Weights are scaled by ``1/n`` so gross exposure stays
    bounded by 1 even when every symbol points the same way.
    """

    name = "momentum"

    def __init__(self) -> None:
        self._prev: Optional[np.ndarray] = None

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        n = closes.shape[0]
        if self._prev is None:
            self._prev = closes
            return np.full((n,), 1.0 / n, dtype=np.float32)
        sign = np.sign(closes - self._prev)
        self._prev = closes
        return (sign / n).astype(np.float32)


# -- causal covariance helpers ----------------------------------------------
#
# The obs carries only the *last* close per symbol, so the mean-variance policies
# accumulate their own trailing close buffer step by step (like ``MomentumPolicy``
# carries ``_prev``). Every covariance estimate is therefore CAUSAL: at step ``t`` it
# is built from closes observed at steps ``<= t`` only, never the full path.


def _returns_from_closes(history: np.ndarray) -> np.ndarray:
    """Simple per-step returns from a ``(T, n)`` close matrix → ``(T-1, n)``."""
    prev = history[:-1]
    safe = np.where(prev == 0.0, 1.0, prev)
    return history[1:] / safe - 1.0


def trailing_covariance(
    history: np.ndarray, lookback: int, ridge: float = 1e-6
) -> np.ndarray:
    """Ridge-regularized covariance of trailing returns over the last ``lookback`` bars.

    ``history`` is the accumulated ``(T, n)`` close matrix; only its tail is used, so the
    estimate at any step depends solely on data up to that step (causal by construction).
    Falls back to a scaled identity until two returns are available.
    """
    hist = np.asarray(history, dtype=np.float64)
    n = hist.shape[1]
    tail = hist[-(lookback + 1) :]
    rets = _returns_from_closes(tail)
    if rets.shape[0] < 2:
        return np.eye(n) * ridge
    cov = np.cov(rets, rowvar=False)
    cov = np.atleast_2d(cov).reshape(n, n)
    return cov + np.eye(n) * ridge


def trailing_mean(history: np.ndarray, lookback: int) -> np.ndarray:
    """Mean of trailing returns over the last ``lookback`` bars (causal, ``(n,)``)."""
    hist = np.asarray(history, dtype=np.float64)
    n = hist.shape[1]
    rets = _returns_from_closes(hist[-(lookback + 1) :])
    if rets.shape[0] < 1:
        return np.zeros((n,))
    return rets.mean(axis=0)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the long-only probability simplex (Duchi et al.).

    Closed-form and deterministic: sort, find the threshold, clip. Returns ``w >= 0`` with
    ``sum(w) == 1``.
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1] if np.any(cond) else n
    theta = cssv[cond][-1] / rho if np.any(cond) else (np.sum(v) - 1.0) / n
    return np.maximum(v - theta, 0.0)


class MinVariancePolicy:
    """Long-only minimum-variance portfolio on the trailing causal covariance.

    Accumulates closes step by step and solves ``min_w w'Σw`` over the probability
    simplex by projected gradient descent (deterministic, ``max_iter`` capped). Warms up
    to equal-weight-long until enough history exists. ``last_cov`` exposes the covariance
    used on the most recent step for causality checks.
    """

    name = "min_variance"

    def __init__(
        self, lookback: int = 60, max_iter: int = 50, min_history: int = 3
    ) -> None:
        self._lookback = lookback
        self._max_iter = max_iter
        self._min_history = min_history
        self._hist: list[np.ndarray] = []
        self.last_cov: Optional[np.ndarray] = None

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        n = closes.shape[0]
        self._hist.append(closes)
        if len(self._hist) < self._min_history:
            self.last_cov = None
            return np.full((n,), 1.0 / n, dtype=np.float32)
        cov = trailing_covariance(np.vstack(self._hist), self._lookback)
        self.last_cov = cov
        w = np.full((n,), 1.0 / n)
        step = 1.0 / (np.trace(cov) + 1e-12)
        for _ in range(self._max_iter):
            w = _project_simplex(w - step * (cov @ w))
        return w.astype(np.float32)


class MaxSharpePolicy:
    """Long-only maximum-Sharpe (tangency) portfolio on the trailing causal moments.

    Solves ``max_w (μ'w)/sqrt(w'Σw)`` over the simplex by projected gradient ascent on the
    Sharpe ratio (deterministic, ``max_iter`` capped). ``μ`` and ``Σ`` are trailing
    in-window estimates, so the policy is causal. Warms up to equal-weight-long.
    """

    name = "max_sharpe"

    def __init__(
        self, lookback: int = 60, max_iter: int = 50, min_history: int = 3
    ) -> None:
        self._lookback = lookback
        self._max_iter = max_iter
        self._min_history = min_history
        self._hist: list[np.ndarray] = []
        self.last_cov: Optional[np.ndarray] = None

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        n = closes.shape[0]
        self._hist.append(closes)
        if len(self._hist) < self._min_history:
            self.last_cov = None
            return np.full((n,), 1.0 / n, dtype=np.float32)
        stacked = np.vstack(self._hist)
        cov = trailing_covariance(stacked, self._lookback)
        mu = trailing_mean(stacked, self._lookback)
        self.last_cov = cov
        w = np.full((n,), 1.0 / n)
        step = 1.0 / (np.trace(cov) + 1e-12)
        for _ in range(self._max_iter):
            port_ret = float(mu @ w)
            sigma = float(np.sqrt(max(w @ cov @ w, 1e-18)))
            grad = mu / sigma - port_ret * (cov @ w) / (sigma**3)
            w = _project_simplex(w + step * grad)
        return w.astype(np.float32)


class KellyVolTargetPolicy:
    """Conservative-Kelly target weights scaled by an inverse-vol regime scalar.

    Per-symbol fractional Kelly ``μ/σ²`` is shrunk by ``kelly_fraction`` (default 0.25),
    then the whole vector is scaled by ``target_vol / realized_vol`` (capped) so exposure
    leans out in high-vol regimes and in when calm. Signed weights are clipped to
    ``[-max_weight, max_weight]`` and gross exposure is bounded to ``1``. Causal: all
    moments are trailing in-window estimates. Warms up flat.
    """

    name = "kelly_vol_target"

    def __init__(
        self,
        lookback: int = 60,
        kelly_fraction: float = 0.25,
        target_vol: float = 0.01,
        vol_cap: float = 2.0,
        max_weight: float = 1.0,
        min_history: int = 3,
    ) -> None:
        self._lookback = lookback
        self._kelly_fraction = kelly_fraction
        self._target_vol = target_vol
        self._vol_cap = vol_cap
        self._max_weight = max_weight
        self._min_history = min_history
        self._hist: list[np.ndarray] = []

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        n = closes.shape[0]
        self._hist.append(closes)
        if len(self._hist) < self._min_history:
            return np.zeros((n,), dtype=np.float32)
        stacked = np.vstack(self._hist)
        rets = _returns_from_closes(stacked[-(self._lookback + 1) :])
        mu = rets.mean(axis=0)
        var = rets.var(axis=0) + 1e-12
        kelly = self._kelly_fraction * (mu / var)
        realized_vol = float(rets.mean(axis=1).std()) + 1e-12
        vol_scalar = min(self._target_vol / realized_vol, self._vol_cap)
        w = np.clip(kelly * vol_scalar, -self._max_weight, self._max_weight)
        gross = float(np.abs(w).sum())
        if gross > 1.0:
            w = w / gross
        return w.astype(np.float32)


# Factories so every episode gets a fresh (state-reset) policy instance.
BASELINE_POLICIES: list[tuple[str, Callable[[], Policy]]] = [
    (FlatPolicy.name, FlatPolicy),
    (EqualWeightLongPolicy.name, EqualWeightLongPolicy),
    (MomentumPolicy.name, MomentumPolicy),
    (MinVariancePolicy.name, MinVariancePolicy),
    (MaxSharpePolicy.name, MaxSharpePolicy),
    (KellyVolTargetPolicy.name, KellyVolTargetPolicy),
]


# -- runner -----------------------------------------------------------------


def _make_env(
    n_symbols: int, n_days: int, seed: int, distribution_mode: str
) -> OpenOutcryEnv:
    """Build an env at ``seed``, requesting ``distribution_mode`` when the binding
    supports it. The ``distribution_mode`` tiering is added by a sibling stream; until
    it lands the kwarg raises ``TypeError`` and we fall back to the default scenario so
    the baseline runner stays green either way."""
    try:
        return OpenOutcryEnv(
            n_symbols=n_symbols,
            n_days=n_days,
            seed=seed,
            distribution_mode=distribution_mode,
        )
    except TypeError:
        return OpenOutcryEnv(n_symbols=n_symbols, n_days=n_days, seed=seed)


def _rollout_returns(env: OpenOutcryEnv, policy: Policy, max_steps: int) -> list[float]:
    obs, _ = env.reset()
    out: list[float] = []
    for _ in range(max_steps):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        out.append(float(reward))
        if bool(terminated) or bool(truncated):
            break
    return out


def run_baselines(
    *,
    n_symbols: int = 4,
    n_days: int = 120,
    seeds: Iterable[int] = range(8),
    distribution_mode: str = "calm",
    max_steps: int = 512,
    n_trials: Optional[int] = None,
) -> list[dict]:
    """Roll every reference policy over ``seeds`` and score it with SharpeBench.

    For each policy we record one episode's return series per seed, pool the series,
    and score the pool with the real ``score_run`` kernel for an aggregate deflated
    Sharpe; ``passed_k_rate`` is the fraction of seeds whose own series passes pass^k.
    ``n_trials`` defaults to the number of baseline policies — the honest declared
    in-sample search breadth, which deflates the Sharpe for multiple-comparison luck.

    Returns one row per policy: ``{policy, deflated_sharpe, passed_k_rate, mean_return}``.
    """
    seeds = list(seeds)
    trials = len(BASELINE_POLICIES) if n_trials is None else int(n_trials)
    rows: list[dict] = []
    for name, factory in BASELINE_POLICIES:
        pooled: list[float] = []
        passed: list[float] = []
        for s in seeds:
            policy = factory()
            env = _make_env(n_symbols, n_days, s, distribution_mode)
            returns = _rollout_returns(env, policy, max_steps)
            pooled.extend(returns)
            if len(returns) >= 2:
                comp = json.loads(score_run(returns, trials))
                passed.append(1.0 if comp.get("passed_k", False) else 0.0)
        composite = json.loads(score_run(pooled, trials)) if len(pooled) >= 2 else {}
        rows.append(
            {
                "policy": name,
                "deflated_sharpe": float(composite.get("deflated_sharpe", 0.0)),
                "passed_k_rate": float(np.mean(passed)) if passed else 0.0,
                "mean_return": float(np.mean(pooled)) if pooled else 0.0,
            }
        )
    return rows


def leaderboard_markdown(rows: Sequence[dict]) -> str:
    """Render baseline ``rows`` as a markdown table sorted by deflated Sharpe (desc).

    The sort key is deflated Sharpe and *only* deflated Sharpe — the social contract
    is that entrants are ranked on the deflated, process-checked number, never raw
    return. Mean return is shown for context, not for ranking.
    """
    ordered = sorted(rows, key=lambda r: r.get("deflated_sharpe", 0.0), reverse=True)
    lines = [
        "| Rank | Policy | Deflated Sharpe | pass^k rate | Mean return |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(ordered, start=1):
        lines.append(
            "| {rank} | {policy} | {ds:.4f} | {pk:.2f} | {mr:.6f} |".format(
                rank=i,
                policy=r.get("policy", "?"),
                ds=float(r.get("deflated_sharpe", 0.0)),
                pk=float(r.get("passed_k_rate", 0.0)),
                mr=float(r.get("mean_return", 0.0)),
            )
        )
    return "\n".join(lines)


__all__ = [
    "FlatPolicy",
    "EqualWeightLongPolicy",
    "MomentumPolicy",
    "MinVariancePolicy",
    "MaxSharpePolicy",
    "KellyVolTargetPolicy",
    "trailing_covariance",
    "trailing_mean",
    "BASELINE_POLICIES",
    "run_baselines",
    "leaderboard_markdown",
]
