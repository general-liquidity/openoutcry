"""Tests for the deterministic episode-failure taxonomy.

Run from the crate dir::

    python -m pytest crates/openoutcry-py/tests/test_failure_taxonomy.py -q

The logic tests are pure: they hand :func:`classify_episode_failure` the exact
returns/events/mandate signals the env and its wrappers surface at episode end. The
live-binding test drives a real forced-liquidation cascade and classifies its output; it is
skipped when the native ``openoutcry`` module is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from openoutcry.failure_taxonomy import (
    FailureMode,
    classify_episode_failure,
    FailureRollup,
    rollup_failure_modes,
)
from openoutcry.mandate import Mandate


# -- event builders mirroring what the wrappers emit ------------------------

def _cascade_events(nav: float, mark_drops):
    """A cascade chain in the exact shape openoutcry.cascade emits."""
    out = [{"event": "margin_call", "nav": nav, "deficit": 0.1}]
    for k, drop in enumerate(mark_drops, start=1):
        out.append({"event": "forced_reduce", "fraction": min(1.0, k / len(mark_drops))})
        out.append({"event": "cascade_impact", "step": k, "mark_drop": float(drop)})
    return out


def _weights_event(weights):
    return {"event": "target_weights", "weights": [float(w) for w in weights]}


# -- single-signal classification -------------------------------------------

def test_clean_episode():
    assert classify_episode_failure([0.01, -0.005, 0.002], [], None) == FailureMode.CLEAN


def test_bankrupt_from_returns():
    # NAV path 1 -> 0.9 -> -0.18 crosses zero: bankruptcy.
    assert classify_episode_failure([-0.1, -1.2], []) == FailureMode.BANKRUPT


def test_cascade_wiped():
    # margin_call at nav=0.3, mark drops 0.2 + 0.2 drive equity below zero.
    events = _cascade_events(0.3, [0.2, 0.2])
    assert classify_episode_failure([-0.01, -0.01], events) == FailureMode.CASCADE_WIPED


def test_cascade_survived_is_stopped_out():
    # margin_call at nav=0.4, small drops leave equity positive -> a survived stop-out.
    events = _cascade_events(0.4, [0.02, 0.02])
    assert classify_episode_failure([-0.01, -0.01], events) == FailureMode.STOPPED_OUT


def test_stop_out_marker_event():
    assert classify_episode_failure([-0.01], [{"event": "stopped_out"}]) == FailureMode.STOPPED_OUT


def test_stop_out_flag_folded_into_events():
    # DrawdownStopper sets info["stopped_out"]=True; if the caller folds that dict in, it reads.
    assert classify_episode_failure([-0.01], [{"stopped_out": True, "nav": 0.5}]) == FailureMode.STOPPED_OUT


def test_mandate_structural_short_under_long_only():
    m = Mandate(style="long_only")
    events = [_weights_event([-0.5, 0.5])] * 3
    assert classify_episode_failure([0.0, 0.0, 0.0], events, m) == FailureMode.MANDATE_STRUCTURAL


def test_mandate_drawdown_cap_blown():
    m = Mandate(style="unconstrained", max_drawdown=0.05)
    # 10% realized drawdown, NAV stays positive (not bankrupt), unconstrained style (no struct).
    assert classify_episode_failure([-0.1], [], m) == FailureMode.MANDATE_DRAWDOWN


def test_mandate_inventory_cap_blown():
    m = Mandate(style="unconstrained", max_inventory=0.5)
    events = [_weights_event([0.8, 0.8])]  # gross 1.6 > 0.5 cap
    assert classify_episode_failure([0.0], events, m) == FailureMode.MANDATE_INVENTORY


def test_mandate_dict_form_accepted():
    m = {"style": "long_only", "max_drawdown": None, "max_inventory": None, "text": ""}
    events = [_weights_event([-0.5, 0.5])]
    assert classify_episode_failure([0.0], events, m) == FailureMode.MANDATE_STRUCTURAL


def test_no_mandate_no_breach_is_clean():
    events = [_weights_event([-0.9, 0.9])]  # a short, but no mandate to breach
    assert classify_episode_failure([0.001], events, None) == FailureMode.CLEAN


# -- precedence (worst-case first) ------------------------------------------

def test_cascade_wipe_outranks_bankrupt():
    events = _cascade_events(0.3, [0.2, 0.2])
    # returns also bankrupt, but the more specific cascade wipe wins.
    assert classify_episode_failure([-0.1, -1.2], events) == FailureMode.CASCADE_WIPED


def test_bankrupt_outranks_stop_out_and_mandate():
    m = Mandate(style="long_only")
    events = [{"event": "stopped_out"}, _weights_event([-1.0, 0.0])]
    assert classify_episode_failure([-0.1, -1.2], events, m) == FailureMode.BANKRUPT


def test_stop_out_outranks_mandate():
    m = Mandate(style="long_only")
    events = [{"event": "stopped_out"}, _weights_event([-1.0, 0.0])]
    assert classify_episode_failure([-0.01], events, m) == FailureMode.STOPPED_OUT


def test_mandate_source_priority_structural_wins_when_largest():
    # long_only fully shorted (structural breach 1.0) plus a blown DD cap; structural is the
    # larger source, so it is reported.
    m = Mandate(style="long_only", max_drawdown=0.05)
    events = [_weights_event([-1.0, 0.0])]
    assert classify_episode_failure([-0.1], events, m) == FailureMode.MANDATE_STRUCTURAL


def test_mandate_tol_suppresses_tiny_breach():
    # A single short bar out of 4 -> structural breach 0.25; a tol above it suppresses.
    m = Mandate(style="long_only")
    events = [_weights_event([-0.5, 0.5])] + [_weights_event([0.5, 0.5])] * 3
    assert classify_episode_failure([0.0] * 4, events, m) == FailureMode.MANDATE_STRUCTURAL
    assert classify_episode_failure([0.0] * 4, events, m, mandate_tol=0.5) == FailureMode.CLEAN


def test_deterministic():
    events = _cascade_events(0.3, [0.2, 0.2])
    a = classify_episode_failure([-0.01, -0.01], events)
    b = classify_episode_failure([-0.01, -0.01], events)
    assert a == b == FailureMode.CASCADE_WIPED


# -- suite-level rollup ------------------------------------------------------

def test_rollup_mixed_forms():
    episodes = [
        FailureMode.CLEAN,                                   # already-classified
        "clean",                                             # string value
        {"returns": [-0.1, -1.2], "events": []},             # dict -> bankrupt
        ([-0.01], [{"event": "stopped_out"}], None),         # tuple -> stopped_out
        {"returns": [-0.01, -0.01], "events": _cascade_events(0.3, [0.2, 0.2])},  # cascade wiped
    ]
    r = rollup_failure_modes(episodes)
    assert isinstance(r, FailureRollup)
    assert r.total == 5
    assert r.counts["clean"] == 2
    assert r.counts["bankrupt"] == 1
    assert r.counts["stopped_out"] == 1
    assert r.counts["cascade_wiped"] == 1
    assert r.clean == 2 and r.failures == 3
    assert r.clean_rate == pytest.approx(0.4)
    assert r.failure_rate == pytest.approx(0.6)
    # every mode is present in the schema (zeros included) for a stable regression snapshot.
    assert set(r.counts) == {m.value for m in FailureMode}


def test_rollup_empty_is_safe():
    r = rollup_failure_modes([])
    assert r.total == 0 and r.clean_rate == 0.0 and r.failure_rate == 0.0
    assert all(v == 0 for v in r.counts.values())


def test_rollup_to_dict_roundtrips_keys():
    r = rollup_failure_modes([FailureMode.CLEAN, FailureMode.BANKRUPT])
    d = r.to_dict()
    assert d["total"] == 2 and d["counts"]["bankrupt"] == 1
    assert set(d) == {"counts", "total", "clean", "failures", "clean_rate", "failure_rate"}


# -- live binding (skipped when the native module is absent) -----------------

openoutcry = pytest.importorskip("openoutcry")


def test_live_cascade_classifies_as_capital_failure():
    from openoutcry.cascade import LiquidationCascadeEnv

    base = openoutcry.OpenOutcryEnv(n_symbols=3, n_days=50, seed=1, distribution_mode="extreme")
    env = LiquidationCascadeEnv(base, maintenance_margin=0.999, cascade_steps=3)
    env.reset(seed=1)
    n = env.action_space.shape[0]
    a = np.full((n,), 1.0 / n, dtype=np.float32)
    returns: list[float] = []
    events: list[dict] = []
    fired = False
    for _ in range(40):
        _o, reward, terminated, truncated, info = env.step(a)
        returns.append(float(reward))
        events.extend(info.get("events", []) or [])
        if info["cascade"]["triggered"]:
            fired = True
        if terminated or truncated:
            break
    assert fired
    mode = classify_episode_failure(returns, events, None)
    assert mode in (
        FailureMode.CASCADE_WIPED,
        FailureMode.STOPPED_OUT,
        FailureMode.BANKRUPT,
    )
