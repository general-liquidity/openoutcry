"""Reward-misspecification **negative controls** for OpenOutcry.

OpenOutcry's thesis is that the SharpeBench kernel (deflated Sharpe / ``pass^k`` /
process checks) PUNISHES naive, over-fit, churn-heavy strategies. This module makes that
thesis *falsifiable*: it ships a registry of deliberately-FLAWED reward functions and a
demonstration that proxy agents optimized for them score BELOW a clean baseline on the
scorer — high in-sample raw return, near-zero deflated Sharpe out-of-sample.

CRITICAL INVARIANT — these rewards are NEGATIVE CONTROLS for research only. They are NOT
valid scoring options. They MUST NEVER be registered into ``rewards.REWARD_SCHEMES`` and
MUST NEVER feed the scorer or the rank key. They re-introduce over-leverage / overfit /
churn / myopia *by design*. Importing this module does not mutate any production registry.

The proxy policies (:data:`MISSPECIFIED_PROXY_POLICIES`) STAND IN for trained agents: we
cannot run GRPO here, so each proxy is the greedy maximizer of its flawed reward (a
max-leverage book for ``raw_pnl``, a tiny-position book for ``win_rate``, a momentum
chaser for ``indicator_shaped``, a last-bar chaser for ``recency_biased``). That is an
honest stand-in — the wedge is structural (deflated Sharpe is scale-invariant, so leverage
inflates raw return without moving Sharpe; deflation then floors it).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .openoutcry_py import score_run

Policy = Callable[[dict], np.ndarray]
MakeEnv = Callable[[int], object]


# ---------------------------------------------------------------------------
# Misspecified reward functions — NEGATIVE CONTROLS. Bounded, pure over ``state``.
# Signature mirrors the production rewards so the rubric *could* route them — which is
# exactly why they must never be registered.
# ---------------------------------------------------------------------------

_RAW_PNL_GAIN = 50.0
_INDICATOR_WINDOW = 3
_RECENCY_DECAY = 0.6


def _returns_from_state(state: Optional[dict]) -> list[float]:
    return [float(r) for r in (state or {}).get("returns", []) or []]


def _net_weights_per_bar(state: Optional[dict]) -> list[float]:
    """Net signed weight per ``target_weights`` event (the agent's directional bet)."""
    out: list[float] = []
    for e in (state or {}).get("events", []) or []:
        if isinstance(e, dict) and e.get("event") == "target_weights":
            w = e.get("weights")
            if isinstance(w, (list, tuple)):
                out.append(float(sum(float(x) for x in w)))
    return out


def raw_pnl_unpenalized(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — gross PnL with no risk, cost, or drawdown penalty.

    ``tanh(GAIN * sum(returns))``: a high-gain reward on raw cumulative return that ignores
    volatility, turnover, and drawdown entirely, so it pays an agent to over-leverage and
    churn. Bounded in ``[-1, 1]``. NOT a valid scoring option.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    return float(np.tanh(_RAW_PNL_GAIN * float(np.sum(rets))))


def win_rate(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — fraction of positive bars, magnitude-blind.

    Rewards *how often* a bar is green, never *by how much*, so it pays an agent to harvest
    many tiny wins and accept rare catastrophic losses (the classic blow-up reward). Bounded
    in ``[0, 1]``. NOT a valid scoring option.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    return float(np.mean([1.0 if r > 0.0 else 0.0 for r in rets]))


def indicator_shaped(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    window: int = _INDICATOR_WINDOW,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — alignment with a naive short-window momentum signal.

    For each bar past ``window``, rewards the agent when its net position sign agrees with the
    sign of the trailing ``window``-bar return momentum — a classic in-sample-overfit reward
    that fits the agent to a single indicator instead of risk-adjusted edge. Reads the recorded
    ``target_weights`` events; vacuously ``0.0`` if the agent never declared a direction.
    Bounded in ``[0, 1]``. NOT a valid scoring option.
    """
    rets = _returns_from_state(state)
    nets = _net_weights_per_bar(state)
    if len(rets) <= window or not nets:
        return 0.0
    hits = 0.0
    count = 0
    for i in range(window, min(len(rets), len(nets))):
        mom = float(np.sign(np.sum(rets[i - window : i])))
        pos = float(np.sign(nets[i]))
        if mom != 0.0 and pos != 0.0:
            hits += 1.0 if mom == pos else 0.0
            count += 1
    return float(hits / count) if count else 0.0


def recency_biased(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    decay: float = _RECENCY_DECAY,
    **kwargs: Any,
) -> float:
    """NEGATIVE CONTROL — exponentially over-weights the most recent bars (myopic).

    Geometric decay places almost all weight on the last few bars, so an agent is rewarded for
    chasing the latest move and discounting the path that produced it. Bounded in ``[-1, 1]``
    via ``tanh`` of the recency-weighted mean return. NOT a valid scoring option.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    n = len(rets)
    w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    weighted = float(np.dot(w, np.asarray(rets, dtype=float)) / w.sum())
    return float(np.tanh(_RAW_PNL_GAIN * weighted))


MISSPECIFIED_REWARDS: dict[str, Callable[..., float]] = {
    "raw_pnl_unpenalized": raw_pnl_unpenalized,
    "win_rate": win_rate,
    "indicator_shaped": indicator_shaped,
    "recency_biased": recency_biased,
}


# ---------------------------------------------------------------------------
# Proxy policies — greedy maximizers of each flawed reward. STAND-INS for trained agents.
# ---------------------------------------------------------------------------


class MaxLeveragePolicy:
    """Greedy maximizer of ``raw_pnl_unpenalized``: full long on every symbol.

    Gross exposure ``= n`` (max per-symbol weight), so it harvests the most raw PnL the action
    space allows. Because deflated Sharpe is scale-invariant, the leverage that inflates raw
    return does NOT move the Sharpe — the wedge."""

    name = "max_leverage"

    def __init__(self, max_weight: float = 1.0) -> None:
        self._w = float(max_weight)

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.full((n,), self._w, dtype=np.float32)


class TinyPositionPolicy:
    """Greedy maximizer of ``win_rate``: a tiny constant long.

    Minimal exposure maximizes the fraction of green bars (slight drift wins often) while each
    win is negligible — high win_rate, near-zero risk-adjusted edge."""

    name = "tiny_position"

    def __init__(self, eps: float = 0.02) -> None:
        self._eps = float(eps)

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.full((n,), self._eps, dtype=np.float32)


class MomentumChasePolicy:
    """Greedy maximizer of ``indicator_shaped``: full-size last-move sign chase.

    Bets the full per-symbol weight in the direction of the last close change, so its position
    sign agrees with short-window momentum by construction. Warms up full long."""

    name = "momentum_chase"

    def __init__(self, max_weight: float = 1.0) -> None:
        self._w = float(max_weight)
        self._prev: Optional[np.ndarray] = None

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        n = closes.shape[0]
        if self._prev is None:
            self._prev = closes
            return np.full((n,), self._w, dtype=np.float32)
        sign = np.sign(closes - self._prev)
        self._prev = closes
        return (sign * self._w).astype(np.float32)


class RecencyChasePolicy:
    """Greedy maximizer of ``recency_biased``: bet on the single most recent move.

    Sizes each symbol by the sign of its last one-bar change at full weight and ignores all
    earlier history — maximally myopic. Warms up full long."""

    name = "recency_chase"

    def __init__(self, max_weight: float = 1.0) -> None:
        self._w = float(max_weight)
        self._prev: Optional[np.ndarray] = None

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        n = closes.shape[0]
        if self._prev is None:
            self._prev = closes
            return np.full((n,), self._w, dtype=np.float32)
        sign = np.sign(closes - self._prev)
        self._prev = closes
        return (sign * self._w).astype(np.float32)


def _clean_reference_policy() -> Policy:
    """An equal-weight-long book — buy-and-hold analog standing in for a clean
    (differential-Sharpe-trained) agent."""

    def _policy(obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return np.full((n,), 1.0 / n, dtype=np.float32)

    return _policy


MISSPECIFIED_PROXY_POLICIES: dict[str, Callable[[], Policy]] = {
    "raw_pnl_unpenalized": MaxLeveragePolicy,
    "win_rate": TinyPositionPolicy,
    "indicator_shaped": MomentumChasePolicy,
    "recency_biased": RecencyChasePolicy,
}


# ---------------------------------------------------------------------------
# Scoring the proxies through the REAL SharpeBench kernel
# ---------------------------------------------------------------------------


def _rollout_returns(env, policy: Policy, max_steps: int) -> list[float]:
    obs, _ = env.reset()
    out: list[float] = []
    for _ in range(max_steps):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        out.append(float(reward))
        if bool(terminated) or bool(truncated):
            break
    return out


def _score_policy(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    factory: Callable[[], Policy],
    max_steps: int,
    n_trials: int,
) -> dict:
    """Roll a fresh policy per seed, pool the return series, and score with ``score_run``."""
    pooled: list[float] = []
    passed: list[float] = []
    for s in seeds:
        returns = _rollout_returns(make_env_for_seed(s), factory(), max_steps)
        pooled.extend(returns)
        if len(returns) >= 2:
            comp = json.loads(score_run(returns, n_trials))
            passed.append(1.0 if comp.get("passed_k", False) else 0.0)
    composite = json.loads(score_run(pooled, n_trials)) if len(pooled) >= 2 else {}
    return {
        "deflated_sharpe": float(composite.get("deflated_sharpe", 0.0)),
        "passed_k": float(np.mean(passed)) if passed else 0.0,
        "mean_return": float(np.mean(pooled)) if pooled else 0.0,
    }


def misspecification_gap(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    *,
    clean_reward: str = "differential_sharpe",
    flawed_reward: str,
    policy: Optional[Callable[[], Policy]] = None,
    max_steps: int = 512,
    n_trials: int = 2,
) -> dict:
    """Score a clean reference vs a flawed-optimized proxy over the SAME seeds.

    ``policy`` is the clean reference policy factory (defaults to an equal-weight-long book
    standing in for a ``clean_reward``-trained agent); the flawed side is the greedy proxy for
    ``flawed_reward`` from :data:`MISSPECIFIED_PROXY_POLICIES`. Both are scored by the real
    ``score_run`` kernel and the deflated-Sharpe / mean-return gaps are reported. The proxies
    STAND IN for trained agents — we cannot run GRPO here, so we evaluate the reward's greedy
    maximizer instead.
    """
    if flawed_reward not in MISSPECIFIED_PROXY_POLICIES:
        raise ValueError(
            f"unknown flawed_reward {flawed_reward!r}; choose from "
            f"{sorted(MISSPECIFIED_PROXY_POLICIES)}"
        )
    seeds = list(seeds)
    clean_factory = policy or _clean_reference_policy
    flawed_factory = MISSPECIFIED_PROXY_POLICIES[flawed_reward]
    clean = _score_policy(make_env_for_seed, seeds, clean_factory, max_steps, n_trials)
    flawed = _score_policy(make_env_for_seed, seeds, flawed_factory, max_steps, n_trials)
    return {
        "clean_reward": clean_reward,
        "flawed_reward": flawed_reward,
        "clean": clean,
        "flawed": flawed,
        "gap_deflated_sharpe": clean["deflated_sharpe"] - flawed["deflated_sharpe"],
        "gap_mean_return": clean["mean_return"] - flawed["mean_return"],
        "proxy_is_stand_in": True,
    }


def demonstrate_punishment(
    make_env_for_seed: MakeEnv,
    seeds: Sequence[int],
    *,
    max_steps: int = 512,
    n_trials: Optional[int] = None,
) -> dict:
    """Run every flawed-reward proxy over ``seeds`` and score it with SharpeBench.

    Returns ``{reward_name: {deflated_sharpe, passed_k, mean_return}}`` — the falsifiable
    demonstration that proxies optimized for the misspecified rewards score poorly on deflated
    Sharpe (and fail ``pass^k``) despite, where applicable, a healthy raw mean return. The
    proxies STAND IN for trained agents. ``n_trials`` defaults to the proxy count — the honest
    declared in-sample search breadth, which deflates Sharpe for multiple-comparison luck.
    """
    seeds = list(seeds)
    trials = len(MISSPECIFIED_PROXY_POLICIES) if n_trials is None else int(n_trials)
    table: dict[str, dict] = {}
    for reward_name, factory in MISSPECIFIED_PROXY_POLICIES.items():
        table[reward_name] = _score_policy(
            make_env_for_seed, seeds, factory, max_steps, trials
        )
    return table


__all__ = [
    "MISSPECIFIED_REWARDS",
    "MISSPECIFIED_PROXY_POLICIES",
    "raw_pnl_unpenalized",
    "win_rate",
    "indicator_shaped",
    "recency_biased",
    "MaxLeveragePolicy",
    "TinyPositionPolicy",
    "MomentumChasePolicy",
    "RecencyChasePolicy",
    "misspecification_gap",
    "demonstrate_punishment",
]
