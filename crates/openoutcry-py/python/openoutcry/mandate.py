"""Per-scenario trading mandates — MiniGrid's ``Mission`` pattern for trading.

In MiniGrid's *Fetch* task each episode ships a per-episode objective ("pick up the
red key") and the agent is graded on satisfying *that* objective, not a fixed one —
fetching the wrong object is penalized even when it is otherwise competent behavior.
The trading analogue: each scenario draws a :class:`Mandate` (a sampled trading
objective — a style constraint, an optional drawdown cap, an optional benchmark) that
the episode is graded against. Two rollouts on the same market but different mandates
are held to different standards.

:func:`sample_mandate` is **deterministic and leak-free**: it derives the whole mandate
from the scenario ``seed`` (known at ``reset``), never from future bars. The mandate
round-trips through plain-JSON (:meth:`Mandate.to_dict` / :func:`mandate_from_dict`) so
it survives a trace/replay, and :func:`mandate_breach` is a pure, numpy-light penalty in
``[0, 1]`` (``0`` = clean, ``1`` = fully breached) the reward layer turns into ``1 -
breach`` — bounded, hence GRPO-safe.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Union

# The constraint families a scenario can draw. ``unconstrained`` is the permissive
# control (no structural breach); the others each carry a distinct structural rule.
STYLES = ("long_only", "market_neutral", "momentum", "unconstrained")

# Styles that require shorting — excluded when the env disallows shorts, so a sampled
# mandate is never unsatisfiable-by-construction.
_SHORT_REQUIRING = frozenset({"market_neutral"})

_DRAWDOWN_CAPS = (0.05, 0.10, 0.15, 0.20)

_EPS = 1e-9


@dataclass(frozen=True)
class Mandate:
    """A per-scenario objective the episode is graded on satisfying.

    ``style`` is the structural constraint; ``max_drawdown`` an optional realized-DD cap
    (a fraction, e.g. ``0.10`` = 10%); ``benchmark`` an optional symbol to beat (carried
    in the prompt text — informational, not breach-scored, since the breach checker has no
    per-symbol returns); ``text`` the human-readable rendering shown to the model.
    """

    style: str
    max_drawdown: Optional[float] = None
    benchmark: Optional[str] = None
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form for the dataset ``info`` dict / trace (scalars + null only)."""
        return {
            "style": self.style,
            "max_drawdown": self.max_drawdown,
            "benchmark": self.benchmark,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Mandate":
        return cls(
            style=str(d["style"]),
            max_drawdown=(None if d.get("max_drawdown") is None else float(d["max_drawdown"])),
            benchmark=(None if d.get("benchmark") is None else str(d["benchmark"])),
            text=str(d.get("text", "")),
        )


def _render_text(style: str, max_drawdown: Optional[float], benchmark: Optional[str]) -> str:
    base = {
        "long_only": "Long-only mandate: hold no short positions",
        "market_neutral": "Market-neutral mandate: keep net exposure near zero (balance longs and shorts)",
        "momentum": "Momentum mandate: lean into recent winners, cut losers",
        "unconstrained": "Unconstrained mandate: trade freely",
    }[style]
    clauses = [base]
    if max_drawdown is not None:
        clauses.append(f"keep max drawdown under {max_drawdown:.0%}")
    if benchmark is not None:
        clauses.append(f"aim to beat {benchmark}")
    return "; ".join(clauses) + "."


def sample_mandate(
    seed: int,
    n_symbols: int = 4,
    *,
    allow_short: bool = True,
) -> Mandate:
    """Deterministically draw a :class:`Mandate` from a scenario ``seed``.

    Leak-free: the draw depends only on ``seed`` (and the static ``n_symbols`` /
    ``allow_short`` env shape), so it is reproducible at ``reset`` and never peeks at
    future bars. When ``allow_short`` is ``False`` the short-requiring styles are dropped
    so the mandate stays satisfiable on a long-only market.
    """
    rng = random.Random(int(seed))
    styles = [s for s in STYLES if allow_short or s not in _SHORT_REQUIRING]
    style = rng.choice(styles)
    max_drawdown = rng.choice(_DRAWDOWN_CAPS) if rng.random() < 0.5 else None
    benchmark = (
        f"SYM{rng.randrange(max(1, int(n_symbols))):02d}" if rng.random() < 0.3 else None
    )
    text = _render_text(style, max_drawdown, benchmark)
    return Mandate(style=style, max_drawdown=max_drawdown, benchmark=benchmark, text=text)


def mandate_text(m: Union[Mandate, dict]) -> str:
    """The human-readable objective string (for the scenario prompt)."""
    return _as_mandate(m).text


def mandate_from_dict(d: dict[str, Any]) -> Mandate:
    """Reconstruct a :class:`Mandate` from its plain-JSON form (trace/replay)."""
    return Mandate.from_dict(d)


def validate_mandate(obj: Union[Mandate, dict, None]) -> bool:
    """True iff ``obj`` is a structurally valid mandate (round-trip / replay guard)."""
    if obj is None:
        return False
    try:
        m = _as_mandate(obj)
    except Exception:  # noqa: BLE001 - malformed payload is simply invalid
        return False
    if m.style not in STYLES:
        return False
    if m.max_drawdown is not None and not (0.0 < float(m.max_drawdown) <= 1.0):
        return False
    return True


def _as_mandate(m: Union[Mandate, dict]) -> Mandate:
    return m if isinstance(m, Mandate) else Mandate.from_dict(m)


def _weights_per_step(events: Any) -> list[list[float]]:
    """The per-step target-weight vectors the rollout recorded as events, if any.

    The multi-turn env appends a ``{"event": "target_weights", "weights": [...]}`` record
    each bar; the breach checker reads structural constraints off those. Events without a
    ``weights`` payload (real market events) are ignored.
    """
    out: list[list[float]] = []
    for e in events or []:
        if isinstance(e, dict) and "weights" in e:
            w = e.get("weights")
            if isinstance(w, (list, tuple)):
                out.append([float(x) for x in w])
    return out


def _max_drawdown(returns: list[float]) -> float:
    """Realized max drawdown of the per-bar return series, as a positive fraction."""
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        equity *= 1.0 + float(r)
        if equity > peak:
            peak = equity
        if peak > _EPS:
            dd = (peak - equity) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def mandate_breach(
    m: Union[Mandate, dict],
    returns: list[float],
    events: list[dict],
) -> float:
    """A bounded breach penalty in ``[0, 1]`` (0 = clean, 1 = fully breached).

    Two independent breach sources, combined by worst-case (``max``) so a clean structure
    with a blown drawdown — or vice versa — still scores the violation:

    * **structural** — a short under ``long_only`` (fraction of bars holding a short), or
      net exposure away from zero under ``market_neutral`` (mean ``|net| / gross``). Read
      off the per-step weight events. ``momentum`` / ``unconstrained`` carry no structural
      rule.
    * **drawdown** — realized max drawdown over the cap, normalized by the cap and
      saturated at 1.

    Pure and numpy-light; safe on empty inputs (returns 0).
    """
    mandate = _as_mandate(m)
    breaches: list[float] = []

    weights = _weights_per_step(events)
    if weights:
        if mandate.style == "long_only":
            short_steps = sum(1 for w in weights if w and min(w) < -_EPS)
            breaches.append(short_steps / len(weights))
        elif mandate.style == "market_neutral":
            nets = []
            for w in weights:
                gross = sum(abs(x) for x in w)
                nets.append(abs(sum(w)) / gross if gross > _EPS else 0.0)
            breaches.append(min(1.0, sum(nets) / len(nets)))

    if mandate.max_drawdown is not None and returns:
        cap = float(mandate.max_drawdown)
        mdd = _max_drawdown([float(r) for r in returns])
        if mdd > cap:
            breaches.append(min(1.0, (mdd - cap) / max(cap, _EPS)))

    if not breaches:
        return 0.0
    return float(min(1.0, max(breaches)))


__all__ = [
    "Mandate",
    "STYLES",
    "sample_mandate",
    "mandate_text",
    "mandate_breach",
    "mandate_from_dict",
    "validate_mandate",
]
