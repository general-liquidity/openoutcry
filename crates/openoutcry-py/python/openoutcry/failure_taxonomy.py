"""Deterministic episode-failure taxonomy for OpenOutcry rollouts.

A trading episode can end in more than one way, and "did it end badly?" is not one bit —
a margin cascade that wiped equity, a plain bankruptcy, a drawdown stop-out, and a mandate
violation are distinct failures that call for distinct fixes. :func:`classify_episode_failure`
reduces an episode to a single :class:`FailureMode` over signals the env and its wrappers
**already surface at episode end** — the realized return series, the accumulated ``events``,
and the scenario :class:`~openoutcry.mandate.Mandate` — with **no new signals invented** and
**no LLM judge** (a subjective judge is a no-fit for a deterministic, leak-free market; the
verdict here has to be reproducible byte-for-byte).

Signals consumed (all pre-existing):

* **returns** — the per-bar realized-return series. Its running product is the NAV path, so
  a NAV that reaches ``<= 0`` is a bankruptcy, and the realized drawdown feeds the mandate
  drawdown check.
* **events** — the accumulated per-bar event dicts. The forced-liquidation cascade
  (:mod:`openoutcry.cascade`) emits ``margin_call`` / ``cascade_impact`` records carrying the
  breach NAV and each step's mark drop, so a cascade's survived-vs-wiped outcome is
  reconstructable from the exact fields it already emits. A ``stopped_out`` marker
  (:class:`~openoutcry.risk.DrawdownStopper`) is read when present. The ``target_weights``
  records feed the mandate structural / inventory checks.
* **mandate** — the per-scenario objective. Which mandate source was breached (structural /
  drawdown / inventory) is recovered by re-scoring against *isolated* single-source mandates
  through the existing Rust :func:`~openoutcry.mandate.mandate_breach` kernel — no
  reimplementation of the breach math.

Classification is worst-case-first (the :func:`~openoutcry.mandate.mandate_breach` ``max``
convention): a terminal capital outcome outranks a mandate-policy violation, and the more
specific cascade wipe outranks a generic bankruptcy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Union

from .mandate import Mandate, mandate_breach, validate_mandate


class FailureMode(str, Enum):
    """The terminal disposition of an episode (``str``-valued for plain-JSON traces)."""

    CLEAN = "clean"
    BANKRUPT = "bankrupt"
    STOPPED_OUT = "stopped_out"
    CASCADE_WIPED = "cascade_wiped"
    MANDATE_STRUCTURAL = "mandate_structural"
    MANDATE_DRAWDOWN = "mandate_drawdown"
    MANDATE_INVENTORY = "mandate_inventory"


def _nav_bankrupt(returns: Sequence[float]) -> bool:
    """True if the NAV path (running product of ``1 + r``) reaches ``<= 0`` — bankruptcy."""
    nav = 1.0
    for r in returns or []:
        nav *= 1.0 + float(r)
        if nav <= 0.0:
            return True
    return False


def _cascade_outcome(events: Sequence[dict]) -> str:
    """Reconstruct the forced-liquidation cascade outcome from its emitted events.

    Returns ``"none"`` (no cascade fired), ``"survived"`` (a cascade fired but equity held),
    or ``"wiped"`` (a cascade drove equity ``<= 0``). Each ``margin_call`` opens a chain at
    its breach NAV; the following ``cascade_impact`` mark drops subtract from it until the
    next ``margin_call`` — exactly the arithmetic :func:`openoutcry.cascade._run_cascade`
    performs, read back off the fields it already surfaces.
    """
    fired = False
    wiped = False
    cur: Optional[float] = None
    for e in events or []:
        name = str(e.get("event", ""))
        if name == "margin_call":
            fired = True
            cur = float(e.get("nav", 0.0))
        elif name == "cascade_impact" and cur is not None:
            cur -= float(e.get("mark_drop", 0.0))
            if cur <= 0.0:
                wiped = True
    if not fired:
        return "none"
    return "wiped" if wiped else "survived"


def _has_stop_out(events: Sequence[dict]) -> bool:
    """True if a drawdown stop-out is signalled — either a ``stopped_out`` event or the
    top-level ``stopped_out`` flag :class:`~openoutcry.risk.DrawdownStopper` sets, whichever
    the caller folded into the event stream."""
    for e in events or []:
        if str(e.get("event", "")) == "stopped_out":
            return True
        if e.get("stopped_out"):
            return True
    return False


def _mandate_failure(
    mandate: Union[Mandate, dict],
    returns: Sequence[float],
    events: Sequence[dict],
    *,
    tol: float,
) -> Optional[FailureMode]:
    """The dominant mandate-breach source, recovered via isolated single-source re-scoring.

    Re-scores the episode against three isolated mandates through the real
    :func:`~openoutcry.mandate.mandate_breach` kernel — one carrying only the structural
    style, one only the drawdown cap, one only the inventory cap — so the per-source breach
    is read straight from the kernel without reimplementing its math. The largest source
    strictly above ``tol`` wins; ties break structural > drawdown > inventory (the enum
    order). ``None`` when no source is breached.
    """
    m = Mandate.from_dict(mandate) if isinstance(mandate, dict) else mandate
    rets = [float(r) for r in returns or []]
    evs = list(events or [])

    candidates: list[tuple[float, FailureMode]] = []
    structural = mandate_breach(Mandate(style=m.style), rets, evs)
    candidates.append((structural, FailureMode.MANDATE_STRUCTURAL))
    if m.max_drawdown is not None:
        dd = mandate_breach(
            Mandate(style="unconstrained", max_drawdown=m.max_drawdown), rets, evs
        )
        candidates.append((dd, FailureMode.MANDATE_DRAWDOWN))
    if m.max_inventory is not None:
        inv = mandate_breach(
            Mandate(style="unconstrained", max_inventory=m.max_inventory), rets, evs
        )
        candidates.append((inv, FailureMode.MANDATE_INVENTORY))

    best_breach, best_mode = max(candidates, key=lambda c: c[0])
    return best_mode if best_breach > tol else None


def classify_episode_failure(
    returns: Sequence[float],
    events: Sequence[dict],
    mandate: Union[Mandate, dict, None] = None,
    *,
    mandate_tol: float = 0.0,
) -> FailureMode:
    """Classify one episode into a single :class:`FailureMode`, worst-case first.

    Precedence (most catastrophic first): a cascade that wiped equity
    (:attr:`FailureMode.CASCADE_WIPED`) outranks a plain bankruptcy
    (:attr:`FailureMode.BANKRUPT`), which outranks a stop-out — a drawdown stop marker or a
    *survived* cascade (:attr:`FailureMode.STOPPED_OUT`) — which outranks a mandate-policy
    breach (:attr:`FailureMode.MANDATE_STRUCTURAL` / :attr:`FailureMode.MANDATE_DRAWDOWN` /
    :attr:`FailureMode.MANDATE_INVENTORY`). An episode tripping none of these is
    :attr:`FailureMode.CLEAN`. Deterministic and pure; safe on empty inputs.
    """
    cascade = _cascade_outcome(events)
    if cascade == "wiped":
        return FailureMode.CASCADE_WIPED
    if _nav_bankrupt(returns):
        return FailureMode.BANKRUPT
    if cascade == "survived" or _has_stop_out(events):
        return FailureMode.STOPPED_OUT
    if validate_mandate(mandate):
        mode = _mandate_failure(mandate, returns, events, tol=mandate_tol)  # type: ignore[arg-type]
        if mode is not None:
            return mode
    return FailureMode.CLEAN


@dataclass(frozen=True)
class FailureRollup:
    """A suite-level tally of episode dispositions for the evaluation path.

    ``counts`` carries every :class:`FailureMode` value (zeros included) so the schema is
    stable across suites — a regression snapshot can diff it directly. ``clean_rate`` is the
    share of clean episodes, ``failure_rate`` its complement.
    """

    counts: dict[str, int]
    total: int
    clean: int
    failures: int
    clean_rate: float
    failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "total": self.total,
            "clean": self.clean,
            "failures": self.failures,
            "clean_rate": self.clean_rate,
            "failure_rate": self.failure_rate,
        }


def _as_mode(item: Any, *, mandate_tol: float) -> FailureMode:
    """Coerce a rollup item to a :class:`FailureMode`: an already-classified mode passes
    through; a mapping is classified from its ``returns`` / ``events`` / ``mandate`` keys; a
    ``(returns, events, mandate)`` sequence is classified positionally."""
    if isinstance(item, FailureMode):
        return item
    if isinstance(item, str):
        return FailureMode(item)
    if isinstance(item, dict):
        return classify_episode_failure(
            item.get("returns", []),
            item.get("events", []),
            item.get("mandate"),
            mandate_tol=mandate_tol,
        )
    returns, events, *rest = tuple(item)
    mandate = rest[0] if rest else None
    return classify_episode_failure(returns, events, mandate, mandate_tol=mandate_tol)


def rollup_failure_modes(
    episodes: Iterable[Any], *, mandate_tol: float = 0.0
) -> FailureRollup:
    """Tally :class:`FailureMode` over a suite of episodes for the evaluation path.

    Each item may be a pre-classified :class:`FailureMode` / its string value, a mapping with
    ``returns`` / ``events`` / ``mandate`` keys, or a ``(returns, events, mandate)`` sequence
    — so a caller can roll up raw rollouts directly or pre-classified verdicts. Deterministic:
    the same suite yields the same tally.
    """
    counts: dict[str, int] = {mode.value: 0 for mode in FailureMode}
    total = 0
    for item in episodes:
        mode = _as_mode(item, mandate_tol=mandate_tol)
        counts[mode.value] += 1
        total += 1
    clean = counts[FailureMode.CLEAN.value]
    failures = total - clean
    clean_rate = clean / total if total else 0.0
    failure_rate = failures / total if total else 0.0
    return FailureRollup(
        counts=counts,
        total=total,
        clean=clean,
        failures=failures,
        clean_rate=clean_rate,
        failure_rate=failure_rate,
    )


__all__ = [
    "FailureMode",
    "classify_episode_failure",
    "FailureRollup",
    "rollup_failure_modes",
]
