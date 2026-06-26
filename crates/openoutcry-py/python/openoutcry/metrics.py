"""Cost-adjusted run-metrics for an OpenOutcry rollout.

:class:`RunMetrics` is a **diagnostic** block — separate from the env reward and from the
SharpeBench score, never a scored signal. It tracks the cheap, deterministic facts about a
run (step count, invalid decisions, agent-supplied token / byte / latency budgets, realized
return, max drawdown) so a leaderboard can rank **cost-adjusted, process-checked**
performance rather than raw Sharpe.

:func:`cost_adjusted_score` combines the authoritative SharpeBench composite with a bounded
efficiency penalty. The SharpeBench score stays authoritative; the penalty only ever shrinks
a positive score (and, symmetrically, makes a negative score worse), so a cheaper run with
the same edge ranks above an expensive one — but no amount of frugality can manufacture
edge that the kernel did not credit.

Determinism: this module never reads a wall clock. ``time_to_decision`` durations are
accepted as inputs (seconds), so a replay reproduces the same metrics.
"""

from __future__ import annotations

from typing import Optional


class RunMetrics:
    """Per-run diagnostic counters. Update via :meth:`record_step`; read via :meth:`to_dict`."""

    def __init__(self) -> None:
        self.steps = 0
        self.invalid_decisions = 0
        self.decision_durations: list[float] = []  # seconds, agent-supplied
        self.tokens = 0
        self.tool_response_bytes = 0
        self._navs: list[float] = []
        self.realized_return = 0.0
        self.max_drawdown = 0.0

    def record_step(
        self,
        *,
        reward: float = 0.0,
        nav: Optional[float] = None,
        invalid: bool = False,
        duration: Optional[float] = None,
        tokens: int = 0,
        tool_response_bytes: int = 0,
    ) -> None:
        self.steps += 1
        if invalid:
            self.invalid_decisions += 1
        if duration is not None:
            self.decision_durations.append(float(duration))
        self.tokens += int(tokens)
        self.tool_response_bytes += int(tool_response_bytes)
        if nav is not None:
            self._navs.append(float(nav))
        else:
            prev = self._navs[-1] if self._navs else 1.0
            self._navs.append(prev * (1.0 + float(reward)))
        self._recompute()

    def _recompute(self) -> None:
        if not self._navs:
            self.realized_return = 0.0
            self.max_drawdown = 0.0
            return
        first = self._navs[0] or 1.0
        self.realized_return = self._navs[-1] / first - 1.0
        peak = self._navs[0]
        mdd = 0.0
        for v in self._navs:
            peak = max(peak, v)
            if peak > 0.0:
                mdd = max(mdd, (peak - v) / peak)
        self.max_drawdown = mdd

    @property
    def time_to_decision(self) -> float:
        """Mean agent-supplied decision latency (seconds); ``0.0`` if none supplied."""
        return (
            sum(self.decision_durations) / len(self.decision_durations)
            if self.decision_durations
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "invalid_decisions": self.invalid_decisions,
            "time_to_decision": self.time_to_decision,
            "total_decision_seconds": sum(self.decision_durations),
            "tokens": self.tokens,
            "tool_response_bytes": self.tool_response_bytes,
            "realized_return": self.realized_return,
            "max_drawdown": self.max_drawdown,
        }


# Default per-unit cost weights. ``invalid`` is punished hardest (a malformed decision is a
# process failure); token/byte/latency are gentle so they only break ties between agents of
# comparable edge.
_DEFAULT_WEIGHTS = {
    "invalid": 1.0,
    "token": 1e-4,
    "byte": 1e-6,
    "time": 0.1,
}


def cost_adjusted_score(
    composite_score: dict,
    metrics: RunMetrics,
    *,
    base_key: str = "deflated_sharpe",
    weights: Optional[dict] = None,
) -> float:
    """Combine the authoritative SharpeBench composite with a bounded efficiency penalty.

    ``base = composite_score[base_key]`` (the deflated Sharpe the benchmark ranks on).
    ``cost`` is a per-step average of the weighted invalid-decision / token / byte / latency
    budgets; ``penalty = 1 / (1 + cost) ∈ (0, 1]``. The result is ``base * penalty``, so
    ``|result| ≤ |base|`` (bounded) and a higher cost monotonically pulls the magnitude
    toward zero. The SharpeBench score is authoritative — the penalty can only discount it.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    base = float(composite_score.get(base_key, 0.0)) if composite_score else 0.0

    steps = max(metrics.steps, 1)
    raw_cost = (
        w["invalid"] * metrics.invalid_decisions
        + w["token"] * metrics.tokens
        + w["byte"] * metrics.tool_response_bytes
        + w["time"] * sum(metrics.decision_durations)
    )
    cost = max(raw_cost, 0.0) / steps
    penalty = 1.0 / (1.0 + cost)  # ∈ (0, 1]
    return base * penalty


__all__ = [
    "RunMetrics",
    "cost_adjusted_score",
]
