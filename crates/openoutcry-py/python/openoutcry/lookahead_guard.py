"""Leak-free at the agent/harness boundary.

OpenOutcry's leak-free guarantee is *structural* — the native env only ever emits a
point-in-time :class:`MarketObservation` (trailing closes up to the decision date), so a
policy literally cannot observe a future bar. :class:`LookaheadGuard` extends that
guarantee one layer out, to the agent harness, the way PrimeIntellect ``rlm`` refuses
broad git-history reads to stop a coding agent from reading the answer key. It is a
*refusal layer*: a deny-list of operations that would read future bars, the full
underlying series, the scenario answer key, or the scenario internals.

The pattern-matching here is **defense-in-depth behind the env's structural guarantee**,
not the guarantee itself — a harness that bypasses the guard still cannot make the env
leak. The point is to catch a misconfigured tool / policy that *tries* to, loudly, and to
keep legitimate analytics on PAST observations unobstructed.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, Optional

ENV_ALLOW_FULL_SERIES = "OPENOUTCRY_ALLOW_FULL_SERIES"


class LookaheadViolation(RuntimeError):
    """Raised when an operation would read beyond the current point-in-time bar."""


# Named operations that read future bars / the full series / the answer key / scenario
# internals. Mapping is op -> the reason it is refused (surfaced in the exception).
BLOCKED_OPERATIONS: dict[str, str] = {
    "read_dataset": "the raw underlying Dataset is the full series, i.e. the answer key",
    "read_full_series": "the full close series past the current bar is unobservable by construction",
    "read_future_bar": "bars after the current decision date are the labels the policy is graded on",
    "peek_next_bar": "the next bar is the realized outcome the decision is scored against",
    "slice_future": "slicing close_history past the current index leaks future returns",
    "read_answer_key": "the scenario answer key is never a policy input",
    "read_scenario_internals": "scenario-generator internals (windows, regimes) are hidden state",
    "scenario_seed_as_feature": "the scenario seed identifies the episode and trivializes generalization",
    "policy_out_of_band_input": "a policy may only consume the observation it is handed",
}

# Substring tells, so a novel op name still trips the guard (defense-in-depth).
_BLOCKED_SUBSTRINGS = (
    "future",
    "lookahead",
    "look_ahead",
    "answer_key",
    "full_series",
    "next_bar",
    "peek",
)


class LookaheadGuard:
    """Refuse operations that would read future / out-of-band data.

    ``allow_full_series`` defaults to ``None`` — the escape hatch is then read from the
    ``OPENOUTCRY_ALLOW_FULL_SERIES`` environment variable on every :meth:`check`, so a
    legitimate research run (``OPENOUTCRY_ALLOW_FULL_SERIES=1``) disables blocking without
    rebuilding the guard. Pass an explicit bool to pin the behaviour.
    """

    def __init__(self, *, allow_full_series: Optional[bool] = None) -> None:
        self._allow_override = allow_full_series

    @property
    def allow_full_series(self) -> bool:
        if self._allow_override is not None:
            return bool(self._allow_override)
        return os.environ.get(ENV_ALLOW_FULL_SERIES, "") == "1"

    def check(self, operation: str, **ctx: Any) -> None:
        """Raise :class:`LookaheadViolation` if ``operation`` would look ahead.

        ``ctx`` may carry ``index`` / ``current_index`` to bound a point-in-time slice:
        reading ``index`` strictly greater than the current bar is a violation. Analytics
        on a past index (``index <= current_index``) pass.
        """
        if self.allow_full_series:
            return
        op = str(operation).lower()
        if op in BLOCKED_OPERATIONS:
            raise LookaheadViolation(f"{operation}: {BLOCKED_OPERATIONS[op]}")
        for sub in _BLOCKED_SUBSTRINGS:
            if sub in op:
                raise LookaheadViolation(
                    f"{operation}: refused — matches lookahead pattern '{sub}'"
                )
        idx = ctx.get("index")
        cur = ctx.get("current_index")
        if idx is not None and cur is not None and int(idx) > int(cur):
            raise LookaheadViolation(
                f"{operation}: index {idx} is past the current point-in-time bar {cur}"
            )


def guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate ``fn`` so its name is checked against the guard before each call.

    Name a data-access helper after what it does (``read_future_bar``) and the guard
    refuses it; a past-only helper (``compute_sma_on_history``) passes. ``index`` /
    ``current_index`` kwargs are honoured for the point-in-time bound.
    """
    guard = LookaheadGuard()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        guard.check(fn.__name__, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


def wrap_policy(policy_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``policy_fn(observation) -> decision`` so it can ONLY see the observation it
    is handed. Extra positional/keyword inputs (a smuggled raw dataset, a future slice,
    the seed) are refused — the policy's only legitimate input is the provided point-in-
    time observation.
    """
    guard = LookaheadGuard()

    @functools.wraps(policy_fn)
    def wrapper(observation: Any, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            guard.check("policy_out_of_band_input")
        return policy_fn(observation)

    return wrapper


__all__ = [
    "LookaheadGuard",
    "LookaheadViolation",
    "BLOCKED_OPERATIONS",
    "ENV_ALLOW_FULL_SERIES",
    "guarded",
    "wrap_policy",
]
