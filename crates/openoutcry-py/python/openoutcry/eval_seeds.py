"""A frozen, named, held-out eval-seed regression set for OpenOutcry.

Procgen ships ``EXPLORATION_LEVEL_SEEDS`` — a tiny dict of named, fixed, hard single
seeds forming a stable evaluation set. Ported here: :data:`EVAL_SEEDS` is a committed,
versioned set of named scenario seeds that form a never-trained-on regression suite.
A deterministic reference policy rolled over each seed produces pinned SharpeBench
scores; :func:`assert_no_regression` gates CI on any drift versus a committed snapshot.

This is the EVAL-level complement to the Rust ``generate_scenario`` golden-hash: that
asserts byte-identity of the generated market; this asserts the *scored* outcome of a
fixed policy on the held-out band is stable across generator/kernel changes.

Every seed lives at/above :data:`EVAL_SEED_BASE`, so the set is provably disjoint from
the train band ``[0, EVAL_SEED_BASE)``. Because the seeds are already absolute held-out
values, the env is constructed with ``mode="train"`` (seed offset 0) and the absolute
seed passed straight through — ``mode="eval"`` would add ``EVAL_SEED_BASE`` again and
double-offset the scenario.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

import numpy as np

from .baselines import EqualWeightLongPolicy, Policy
from .dataset import EVAL_SEED_BASE
from .gym import OpenOutcryEnv
from .openoutcry_py import score_run

SCHEMA_VERSION = "1"
EVAL_SET_VERSION = "openoutcry-eval-seeds-v1"

# Committed, named held-out seeds — a never-trained-on regression band. Offsets are
# spread so the scenarios are not near-duplicates; names are stable identifiers (the
# contract is the name→seed mapping, not the label semantics).
EVAL_SEEDS: dict[str, int] = {
    "held_out_00": EVAL_SEED_BASE + 0,
    "held_out_01": EVAL_SEED_BASE + 7,
    "held_out_02": EVAL_SEED_BASE + 13,
    "held_out_03": EVAL_SEED_BASE + 29,
    "held_out_04": EVAL_SEED_BASE + 101,
    "held_out_05": EVAL_SEED_BASE + 257,
    "held_out_06": EVAL_SEED_BASE + 1024,
    "held_out_07": EVAL_SEED_BASE + 4099,
}

# Disjointness + uniqueness guard: the committed set must never touch the train band.
for _name, _seed in EVAL_SEEDS.items():
    assert _seed >= EVAL_SEED_BASE, (
        f"eval seed {_name}={_seed} is below EVAL_SEED_BASE={EVAL_SEED_BASE}; "
        "named eval seeds must live in the held-out band"
    )
assert len(set(EVAL_SEEDS.values())) == len(EVAL_SEEDS), "eval seeds must be unique"

PolicyFactory = Callable[[], Policy]


def _make_env(
    n_symbols: int, n_days: int, seed: int, distribution_mode: str
) -> OpenOutcryEnv:
    try:
        return OpenOutcryEnv(
            n_symbols=n_symbols,
            n_days=n_days,
            seed=seed,
            distribution_mode=distribution_mode,
            mode="train",
        )
    except TypeError:
        return OpenOutcryEnv(n_symbols=n_symbols, n_days=n_days, seed=seed, mode="train")


def _rollout_returns(env: OpenOutcryEnv, policy: Policy, max_steps: int) -> list[float]:
    obs, _ = env.reset()
    out: list[float] = []
    for _ in range(max_steps):
        obs, reward, terminated, truncated, _info = env.step(policy(obs))
        out.append(float(reward))
        if bool(terminated) or bool(truncated):
            break
    return out


def evaluate_eval_set(
    *,
    n_symbols: int = 4,
    n_days: int = 120,
    distribution_mode: str = "calm",
    policy: Optional[PolicyFactory] = None,
    max_steps: int = 512,
    n_trials: Optional[int] = None,
) -> dict[str, dict]:
    """Roll a deterministic policy over every named eval seed and score each run.

    ``policy`` is a zero-arg factory yielding a fresh (state-reset) policy per seed,
    defaulting to :class:`EqualWeightLongPolicy` (an equal-weight long baseline). Each
    seed's return series is scored with the real SharpeBench kernel. ``n_trials``
    deflates for declared in-sample search breadth and defaults to the size of the eval
    set — pin it explicitly to keep a snapshot reproducible across set growth.

    Returns ``{name: {"deflated_sharpe", "passed_k", "mean_return"}}`` — the pinned
    regression snapshot. The result is byte-identical across calls (the regression-gate
    property), since the seeds, policy, kernel, and env are all deterministic.
    """
    factory: PolicyFactory = policy or EqualWeightLongPolicy
    trials = len(EVAL_SEEDS) if n_trials is None else int(n_trials)
    out: dict[str, dict] = {}
    for name, seed in EVAL_SEEDS.items():
        env = _make_env(n_symbols, n_days, seed, distribution_mode)
        returns = _rollout_returns(env, factory(), max_steps)
        comp = json.loads(score_run(returns, trials)) if len(returns) >= 2 else {}
        out[name] = {
            "deflated_sharpe": float(comp.get("deflated_sharpe", 0.0)),
            "passed_k": bool(comp.get("passed_k", False)),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
        }
    return out


def assert_no_regression(
    reference: dict[str, dict], current: dict[str, dict], *, tol: float = 1e-9
) -> None:
    """Raise ``AssertionError`` if ``current`` drifts from the committed ``reference``.

    The CI gate: float metrics must match within ``tol``; ``passed_k`` (the pass^k gate)
    must match exactly. A changed seed set is itself a regression (the contract moved).
    """
    if set(reference) != set(current):
        raise AssertionError(
            f"eval-seed set changed: reference={sorted(reference)} "
            f"current={sorted(current)}"
        )
    for name in reference:
        ref, cur = reference[name], current[name]
        for key in ("deflated_sharpe", "mean_return"):
            delta = abs(float(ref[key]) - float(cur[key]))
            if delta > tol:
                raise AssertionError(
                    f"regression at {name}.{key}: reference={ref[key]!r} "
                    f"current={cur[key]!r} (|Δ|={delta} > tol={tol})"
                )
        if bool(ref["passed_k"]) != bool(cur["passed_k"]):
            raise AssertionError(
                f"regression at {name}.passed_k: reference={ref['passed_k']!r} "
                f"current={cur['passed_k']!r}"
            )


__all__ = [
    "SCHEMA_VERSION",
    "EVAL_SET_VERSION",
    "EVAL_SEEDS",
    "evaluate_eval_set",
    "assert_no_regression",
]
