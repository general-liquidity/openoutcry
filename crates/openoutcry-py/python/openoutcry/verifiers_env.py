"""PrimeIntellect ``verifiers`` environment for OpenOutcry.

================================  AUTHORED-UNVERIFIED  ================================
The ``verifiers`` package is NOT installed in this build environment, so the code in
this module has NOT been executed end-to-end. It is authored against the documented
``verifiers`` API (``vf.Environment`` / ``vf.Rubric`` / module-level
``load_environment``) and the import is guarded — importing :mod:`openoutcry` never
fails just because ``verifiers`` is absent. The reward functions wire SharpeBench-style
metrics (deflated Sharpe / pass^k / process checks) as placeholders over the rollout
state; calibrate against the Rust scorers before relying on the numbers.
======================================================================================

Usage (once ``verifiers`` is installed)::

    import verifiers as vf
    env = vf.load_environment("openoutcry")   # or: from openoutcry.verifiers_env import load_environment

The rollout drives :class:`~openoutcry.gym.OpenOutcryEnv` and records the per-step
reward (portfolio return) into ``state['returns']``; the rubric then scores the run.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

try:  # pragma: no cover - exercised only when verifiers is installed
    import verifiers as vf

    _HAS_VERIFIERS = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    vf = None  # type: ignore[assignment]
    _HAS_VERIFIERS = False


# ---------------------------------------------------------------------------
# SharpeBench-style reward functions (placeholders — calibrate vs the Rust scorers)
# ---------------------------------------------------------------------------

def _returns_from_state(state: dict) -> list[float]:
    """Per-step portfolio returns recorded by the rollout, if any."""
    return list(state.get("returns", []) or [])


def deflated_sharpe_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    n_trials: int = 1,
    periods_per_year: float = 252.0,
    **kwargs: Any,
) -> float:
    """Deflated-Sharpe placeholder: annualized Sharpe shrunk for the number of
    in-sample trials (a crude multiple-testing haircut). Real deflation uses the
    variance of the trial Sharpes; this is a monotone stand-in."""
    returns = _returns_from_state(state or {})
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return 0.0
    sharpe = (mean / std) * math.sqrt(periods_per_year)
    # Haircut grows with log(n_trials): more search ⇒ more deflation.
    haircut = 1.0 / (1.0 + math.log1p(max(n_trials - 1, 0)))
    return sharpe * haircut


def pass_k_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    threshold: float = 0.0,
    **kwargs: Any,
) -> float:
    """pass^k placeholder: 1.0 iff the run's total return clears ``threshold``
    (the per-run indicator pass^k aggregates with mode='all' over k seeds)."""
    returns = _returns_from_state(state or {})
    if not returns:
        return 0.0
    total = 1.0
    for r in returns:
        total *= 1.0 + r
    return 1.0 if (total - 1.0) > threshold else 0.0


def process_check_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """Process-check placeholder: penalize block-severity events surfaced by the
    environment's per-step ``info`` (e.g. sim-exploitation guards). 1.0 = clean."""
    events: Sequence[dict] = state.get("events", []) if state else []
    bad = sum(1 for e in events if str(e.get("event", "")).endswith("violation"))
    return 1.0 if bad == 0 else max(0.0, 1.0 - 0.25 * bad)


def build_rubric():
    """Assemble the SharpeBench-style :class:`vf.Rubric`. Raises if verifiers
    is unavailable (callers should gate on :data:`_HAS_VERIFIERS`)."""
    if not _HAS_VERIFIERS:
        raise RuntimeError("verifiers is not installed; cannot build a Rubric")
    return vf.Rubric(
        funcs=[deflated_sharpe_reward, pass_k_reward, process_check_reward],
        weights=[1.0, 0.5, 0.5],
    )


def load_environment(
    n_symbols: int = 4,
    n_days: int = 120,
    seed: int = 0,
    **kwargs: Any,
):
    """``verifiers`` entry point. AUTHORED-UNVERIFIED — see module docstring.

    Builds a single-turn-per-bar environment whose dataset is the OpenOutcry
    synthetic market and whose rubric is the SharpeBench-style metric bundle.
    """
    if not _HAS_VERIFIERS:
        raise RuntimeError(
            "verifiers is not installed. Install PrimeIntellect 'verifiers' to load "
            "this environment; the rest of the openoutcry package works without it."
        )
    rubric = build_rubric()
    # The concrete Environment subclass / dataset wiring depends on the verifiers
    # release; a SingleTurnEnv over a market-observation dataset is the intended
    # shape. Left as the documented constructor call for the operator to confirm.
    return vf.SingleTurnEnv(  # type: ignore[attr-defined]
        dataset=kwargs.pop("dataset", None),
        rubric=rubric,
        **kwargs,
    )


__all__ = [
    "deflated_sharpe_reward",
    "pass_k_reward",
    "process_check_reward",
    "build_rubric",
    "load_environment",
]
