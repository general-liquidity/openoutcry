"""Versioned, leak-free rollout-trace recording for :class:`~openoutcry.gym.OpenOutcryEnv`.

A :class:`RolloutTraceWriter` records a run as an append-only JSONL of per-step records
plus a single run ``meta`` record carrying the config, scenario seeds, and the **real**
SharpeBench :func:`score_run` scores over the collected returns. The artifact is the
production-trace→eval substrate: :func:`load_trace` + :func:`trace_to_returns` feed a
RECORDED trace back into the SharpeBench process-checks / ``pass^k`` kernel offline, with
no live env spawn.

Leak-safety is the load-bearing invariant. A point-in-time backtest is only valid if the
trace contains the observation the agent actually saw at bar ``t`` and the decision it made
— never the future bars or the full underlying series. The writer therefore records only
caller-supplied per-step records and **rejects** a raw dataset / env handle (anything that
exposes a ``reset``/``step`` market surface or is a known dataset type); a leaked series
would let an offline re-scorer peek ahead and silently inflate the score.

Robustness follows the ``rlm`` rule: a missing field contributes zero and never cascades —
a malformed line is skipped on load, a missing reward reads as ``0.0``.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import numpy as np

# PUBLIC contract — bump on any breaking change to the record/meta shape.
SCHEMA_VERSION = "openoutcry.trace/1.0.0"

_STEP = "step"
_META = "meta"

# Type names that must never be serialized into a trace: a raw dataset / env handle
# carries the full (incl. future) series. Matched by class name so we don't import the
# native binding here.
_LEAKY_TYPE_NAMES = frozenset({"TradingEnv", "Dataset", "OpenOutcryEnv"})


def _is_leaky(value: Any) -> bool:
    """True if ``value`` is a raw dataset / env handle rather than a point-in-time record."""
    if type(value).__name__ in _LEAKY_TYPE_NAMES:
        return True
    return callable(getattr(value, "reset", None)) and callable(getattr(value, "step", None))


def _reject_leaky(name: str, value: Any) -> None:
    if _is_leaky(value):
        raise TypeError(
            f"refusing to record {name}={type(value).__name__!r}: a raw dataset/env handle "
            "would leak future bars into the trace; pass the point-in-time "
            f"{name} the agent actually saw, not the underlying series."
        )


def _jsonify(x: Any) -> Any:
    """Recursively coerce numpy arrays/scalars to JSON-native types; pass everything else."""
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _composite(returns: Sequence[float], n_trials: int) -> dict:
    """The real SharpeBench ``CompositeScore`` for a return series, ``{}`` for < 2 bars.

    Imported lazily so this module is importable (and ``py_compile``-able) without the
    native binding built.
    """
    rets = [float(r) for r in returns]
    if len(rets) < 2:
        return {}
    from .openoutcry_py import score_run

    return json.loads(score_run(rets, int(n_trials)))


class RolloutTraceWriter:
    """Append-only JSONL writer for one rollout.

    Per-step records are ``{"kind": "step", "step": int, "observation": ..., "decision":
    ..., "reward": float, "info": dict}``; the run is closed by a single ``{"kind": "meta",
    ...}`` record holding config, ``schema_version``, scenario seeds, and the SharpeBench
    scores over the collected returns. ``path=None`` keeps the trace in memory only.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        config: Optional[dict] = None,
        n_trials: int = 0,
    ) -> None:
        self.path = path
        self._config = dict(config or {})
        self._n_trials = int(n_trials)
        self._records: list[dict] = []
        self._seeds: set[int] = set()
        self._fh = open(path, "w", encoding="utf-8") if path is not None else None

    # -- recording ---------------------------------------------------------

    def record_step(
        self,
        *,
        step: int,
        observation: Any,
        decision: Any,
        reward: float,
        info: Optional[dict] = None,
    ) -> dict:
        """Append one point-in-time step. Rejects a raw dataset/env as observation/decision."""
        _reject_leaky("observation", observation)
        _reject_leaky("decision", decision)
        info = info or {}
        seed = info.get("scenario_seed")
        if seed is not None:
            try:
                self._seeds.add(int(seed))
            except (TypeError, ValueError):
                pass
        record = {
            "kind": _STEP,
            "step": int(step),
            "observation": _jsonify(observation),
            "decision": _jsonify(decision),
            "reward": float(reward),
            "info": _jsonify(info),
        }
        self._records.append(record)
        self._write(record)
        return record

    def returns(self) -> list[float]:
        """The realized per-bar return series collected so far."""
        return trace_to_returns(self._records)

    def finalize(self, meta_extra: Optional[dict] = None) -> dict:
        """Write and return the run ``meta`` record (config + seeds + SharpeBench scores)."""
        returns = self.returns()
        meta = {
            "kind": _META,
            "schema_version": SCHEMA_VERSION,
            "config": self._config,
            "n_trials": self._n_trials,
            "n_steps": len(self._records),
            "scenario_seeds": sorted(self._seeds),
            "scores": _composite(returns, self._n_trials),
        }
        if meta_extra:
            meta.update(meta_extra)
        self._write(meta)
        if self._fh is not None:
            self._fh.flush()
        return meta

    # -- file plumbing -----------------------------------------------------

    def _write(self, record: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RolloutTraceWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def load_trace(path: str) -> tuple[list[dict], dict]:
    """Read a JSONL trace into ``(step_records, meta)``. Malformed/blank lines are skipped."""
    records: list[dict] = []
    meta: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("kind") == _META:
                meta = obj
            else:
                records.append(obj)
    return records, meta


def trace_to_returns(records: Sequence[dict]) -> list[float]:
    """Reconstruct the realized per-bar return series from step records (in recorded order).

    A record missing ``reward`` contributes ``0.0`` rather than cascading a failure.
    """
    out: list[float] = []
    for rec in records:
        if not isinstance(rec, dict) or rec.get("kind") == _META:
            continue
        try:
            out.append(float(rec.get("reward", 0.0)))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "RolloutTraceWriter",
    "load_trace",
    "trace_to_returns",
]
