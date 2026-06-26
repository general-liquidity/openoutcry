"""Tests for the rollout-trace schema (:mod:`openoutcry.trace`) and run-metrics
(:mod:`openoutcry.metrics`).

The env/score tests need the native ``openoutcry`` binding (for ``OpenOutcryEnv`` /
``score_run``) and skip when it isn't built. The pure-logic tests (round-trip, metrics,
leak rejection) load the two modules standalone when the binding is absent, so they run on a
machine without the Rust toolchain too.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

_PKGDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "python", "openoutcry")
)

try:
    import openoutcry as oo  # noqa: F401
    from openoutcry.trace import (
        SCHEMA_VERSION,
        RolloutTraceWriter,
        load_trace,
        trace_to_returns,
    )
    from openoutcry.metrics import RunMetrics, cost_adjusted_score

    HAS_BINDING = True
except Exception:  # noqa: BLE001 - binding not built; fall back to standalone modules
    HAS_BINDING = False
    _pkg = types.ModuleType("oo_standalone")
    _pkg.__path__ = [_PKGDIR]  # type: ignore[attr-defined]
    sys.modules.setdefault("oo_standalone", _pkg)

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            f"oo_standalone.{name}", os.path.join(_PKGDIR, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    _trace = _load("trace")
    _metrics = _load("metrics")
    SCHEMA_VERSION = _trace.SCHEMA_VERSION
    RolloutTraceWriter = _trace.RolloutTraceWriter
    load_trace = _trace.load_trace
    trace_to_returns = _trace.trace_to_returns
    RunMetrics = _metrics.RunMetrics
    cost_adjusted_score = _metrics.cost_adjusted_score


needs_binding = pytest.mark.skipif(
    not HAS_BINDING, reason="native openoutcry binding not built"
)


# --- env-driven round-trip ----------------------------------------------------


@needs_binding
def test_rollout_roundtrips_and_scores(tmp_path):
    from openoutcry import OpenOutcryEnv, score_run

    env = OpenOutcryEnv(n_symbols=3, n_days=40, seed=7)
    obs, info = env.reset(seed=7)
    weights = np.full(len(env.symbols), 1.0 / len(env.symbols), dtype=np.float32)

    path = str(tmp_path / "run.jsonl")
    writer = RolloutTraceWriter(path, config={"policy": "equal_weight"}, n_trials=0)

    recorded_returns: list[float] = []
    for t in range(8):
        nobs, reward, terminated, truncated, info = env.step(weights)
        writer.record_step(
            step=t, observation=obs, decision=weights, reward=reward, info=info
        )
        recorded_returns.append(float(reward))
        obs = nobs
        if terminated or truncated:
            break
    meta = writer.finalize()
    writer.close()

    records, loaded_meta = load_trace(path)
    assert len(records) == len(recorded_returns)
    assert loaded_meta["schema_version"] == SCHEMA_VERSION
    assert loaded_meta["scenario_seeds"] == [7]

    rebuilt = trace_to_returns(records)
    assert rebuilt == pytest.approx(recorded_returns)

    # finalize produced the real SharpeBench composite over the collected returns.
    assert meta["scores"] == loaded_meta["scores"]
    assert meta["scores"] == json.loads(score_run(rebuilt, 0))
    assert "deflated_sharpe" in meta["scores"]


@needs_binding
def test_observation_is_point_in_time_not_full_series(tmp_path):
    from openoutcry import OpenOutcryEnv

    env = OpenOutcryEnv(n_symbols=3, n_days=40, seed=1)
    obs, _ = env.reset(seed=1)
    path = str(tmp_path / "pit.jsonl")
    with RolloutTraceWriter(path) as writer:
        nobs, reward, _, _, info = env.step(
            np.zeros(len(env.symbols), dtype=np.float32)
        )
        writer.record_step(step=0, observation=obs, decision="hold", reward=reward, info=info)
        writer.finalize()

    records, _ = load_trace(path)
    rec_obs = records[0]["observation"]
    # The decoded obs is per-symbol point-in-time vectors, never the n_days series.
    assert set(rec_obs) == {"closes", "positions", "cash"}
    assert len(rec_obs["closes"]) == len(env.symbols)
    assert len(rec_obs["positions"]) == len(env.symbols)


# --- leak safety (no binding required) ---------------------------------------


def test_writer_rejects_raw_env_handle(tmp_path):
    class _FakeEnv:  # walks like an env: exposes reset/step → a leaky full-series handle
        def reset(self):  # pragma: no cover - never called
            return {}

        def step(self, a):  # pragma: no cover - never called
            return {}

    class Dataset:  # leaky by class name
        pass

    writer = RolloutTraceWriter(str(tmp_path / "leak.jsonl"))
    with pytest.raises(TypeError):
        writer.record_step(step=0, observation=_FakeEnv(), decision="hold", reward=0.0)
    with pytest.raises(TypeError):
        writer.record_step(step=0, observation={"ok": 1}, decision=Dataset(), reward=0.0)
    writer.close()


def test_trace_to_returns_tolerates_missing_reward():
    records = [
        {"kind": "step", "step": 0, "reward": 0.1},
        {"kind": "step", "step": 1},  # missing reward → 0.0, must not cascade
        {"kind": "step", "step": 2, "reward": -0.05},
    ]
    assert trace_to_returns(records) == pytest.approx([0.1, 0.0, -0.05])


def test_load_trace_skips_malformed_lines(tmp_path):
    path = tmp_path / "noisy.jsonl"
    path.write_text(
        '{"kind":"step","step":0,"reward":0.2}\n'
        "not json at all\n"
        "\n"
        '{"kind":"meta","schema_version":"x","scores":{}}\n',
        encoding="utf-8",
    )
    records, meta = load_trace(str(path))
    assert len(records) == 1
    assert meta["schema_version"] == "x"


# --- run-metrics (no binding required) ---------------------------------------


def test_runmetrics_to_dict_keys():
    m = RunMetrics()
    m.record_step(reward=0.1, invalid=False, duration=0.5, tokens=100, tool_response_bytes=2048)
    m.record_step(reward=-0.2, invalid=True, duration=1.5, tokens=50, tool_response_bytes=512)
    d = m.to_dict()
    assert set(d) == {
        "steps",
        "invalid_decisions",
        "time_to_decision",
        "total_decision_seconds",
        "tokens",
        "tool_response_bytes",
        "realized_return",
        "max_drawdown",
    }
    assert d["steps"] == 2
    assert d["invalid_decisions"] == 1
    assert d["tokens"] == 150
    assert d["tool_response_bytes"] == 2560
    assert d["time_to_decision"] == pytest.approx(1.0)
    assert 0.0 <= d["max_drawdown"] <= 1.0


def test_cost_adjusted_score_is_bounded_and_penalizes_cost():
    composite = {"deflated_sharpe": 2.0}

    cheap = RunMetrics()
    for _ in range(10):
        cheap.record_step(reward=0.0, tokens=10, duration=0.01)

    expensive = RunMetrics()
    for _ in range(10):
        expensive.record_step(
            reward=0.0, invalid=True, tokens=5000, tool_response_bytes=100000, duration=5.0
        )

    s_cheap = cost_adjusted_score(composite, cheap)
    s_exp = cost_adjusted_score(composite, expensive)

    # Authoritative score is never inflated by the penalty, only discounted.
    assert 0.0 < s_cheap <= composite["deflated_sharpe"]
    assert 0.0 < s_exp <= s_cheap
    assert s_exp < s_cheap  # higher cost ⇒ strictly lower cost-adjusted score


def test_cost_adjusted_score_missing_base_key_is_zero():
    assert cost_adjusted_score({}, RunMetrics()) == 0.0
    assert cost_adjusted_score({"other": 1.0}, RunMetrics()) == 0.0
