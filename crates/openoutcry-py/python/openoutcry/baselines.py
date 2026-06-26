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


# Factories so every episode gets a fresh (state-reset) policy instance.
BASELINE_POLICIES: list[tuple[str, Callable[[], Policy]]] = [
    (FlatPolicy.name, FlatPolicy),
    (EqualWeightLongPolicy.name, EqualWeightLongPolicy),
    (MomentumPolicy.name, MomentumPolicy),
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
    "BASELINE_POLICIES",
    "run_baselines",
    "leaderboard_markdown",
]
