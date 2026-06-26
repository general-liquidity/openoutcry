"""Agent-boundary runtime: the lookahead guard and the MCP server.

The guard tests need no external deps. The MCP-server tests ``importorskip('mcp')`` so the
suite stays green without the optional SDK; a separate test asserts the import-absent
contract (``import openoutcry.mcp_server`` works; ``build_server`` raises cleanly).
"""

import json

import pytest

from openoutcry.lookahead_guard import (
    ENV_ALLOW_FULL_SERIES,
    LookaheadGuard,
    LookaheadViolation,
    guarded,
    wrap_policy,
)


# -- lookahead guard ---------------------------------------------------------

def test_blocked_named_ops_raise():
    guard = LookaheadGuard()
    for op in (
        "read_dataset",
        "read_full_series",
        "read_future_bar",
        "peek_next_bar",
        "read_answer_key",
        "scenario_seed_as_feature",
    ):
        with pytest.raises(LookaheadViolation):
            guard.check(op)


def test_blocked_substring_ops_raise():
    guard = LookaheadGuard()
    with pytest.raises(LookaheadViolation):
        guard.check("fetch_future_returns")
    with pytest.raises(LookaheadViolation):
        guard.check("policy_lookahead_hack")


def test_past_only_ops_pass():
    guard = LookaheadGuard()
    # named analytics on history are fine; a past index is fine.
    guard.check("compute_sma_on_history")
    guard.check("read_close_history", index=5, current_index=10)
    guard.check("read_close_history", index=10, current_index=10)


def test_future_index_slice_blocked():
    guard = LookaheadGuard()
    with pytest.raises(LookaheadViolation):
        guard.check("read_close_history", index=11, current_index=10)


def test_env_var_escape_hatch(monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_FULL_SERIES, "1")
    guard = LookaheadGuard()
    # blocking is disabled wholesale for legitimate research.
    guard.check("read_dataset")
    guard.check("read_future_bar", index=99, current_index=1)
    monkeypatch.setenv(ENV_ALLOW_FULL_SERIES, "0")
    with pytest.raises(LookaheadViolation):
        guard.check("read_dataset")


def test_explicit_allow_override():
    assert LookaheadGuard(allow_full_series=True).check("read_dataset") is None
    with pytest.raises(LookaheadViolation):
        LookaheadGuard(allow_full_series=False).check("read_dataset")


def test_guarded_decorator():
    @guarded
    def read_future_bar():
        return "leaked"

    @guarded
    def compute_sma_on_history(values):
        return sum(values) / len(values)

    with pytest.raises(LookaheadViolation):
        read_future_bar()
    assert compute_sma_on_history([1.0, 2.0, 3.0]) == 2.0


def test_wrap_policy_rejects_out_of_band_inputs():
    wrapped = wrap_policy(lambda obs: {"seen": obs})
    assert wrapped({"closes": [1.0]}) == {"seen": {"closes": [1.0]}}
    with pytest.raises(LookaheadViolation):
        wrapped({"closes": [1.0]}, "the_raw_dataset")
    with pytest.raises(LookaheadViolation):
        wrapped({"closes": [1.0]}, seed=42)


# -- mcp_server import-absent contract (runs even without mcp) ----------------

def test_mcp_server_imports_without_mcp():
    import openoutcry.mcp_server as ms

    assert hasattr(ms, "build_server")
    assert hasattr(ms, "main")
    if not ms._HAS_MCP:
        with pytest.raises(RuntimeError):
            ms.build_server()
        with pytest.raises(RuntimeError):
            ms.main()


def test_decision_to_weights_helper():
    import openoutcry.mcp_server as ms

    decision = json.dumps(
        {"orders": [{"symbol": "BBB", "action": "buy", "target_weight": 0.5}]}
    )
    weights = ms._decision_to_weights(decision, ["AAA", "BBB", "CCC"])
    assert weights.tolist() == [0.0, 0.5, 0.0]


# -- mcp server (skipped when mcp is absent) ---------------------------------

mcp = pytest.importorskip("mcp")


def _call(server, name, arguments):
    """Invoke a registered FastMCP tool and return its decoded string result."""
    import asyncio

    _content, result = asyncio.run(server.call_tool(name, arguments))
    return result["result"]


def test_build_server_constructs():
    from openoutcry.mcp_server import build_server

    server = build_server(env_kwargs={"n_symbols": 3, "n_days": 30, "seed": 1})
    assert server is not None
    assert server.name == "openoutcry"


def test_step_returns_structured_error_on_malformed_decision():
    from openoutcry.mcp_server import build_server

    server = build_server(env_kwargs={"n_symbols": 2, "n_days": 20, "seed": 2})
    _call(server, "reset", {})
    out = json.loads(_call(server, "step", {"decision_json": "{not valid json}"}))
    assert "error" in out  # structured content, not a raised exception


def test_reset_step_roundtrip():
    from openoutcry.mcp_server import build_server

    server = build_server(env_kwargs={"n_symbols": 3, "n_days": 30, "seed": 3})
    obs = json.loads(_call(server, "reset", {}))
    assert len(obs["symbols"]) == 3
    assert len(obs["closes"]) == 3

    symbols = obs["symbols"]
    decision = json.dumps(
        {"orders": [{"symbol": symbols[0], "action": "buy", "target_weight": 0.5}]}
    )
    result = json.loads(_call(server, "step", {"decision_json": decision}))
    assert "observation" in result
    assert "reward" in result and isinstance(result["reward"], float)
    assert "terminated" in result and "truncated" in result
    assert len(result["observation"]["closes"]) == 3


def test_spec_tool_exposes_schema_and_symbols():
    from openoutcry.mcp_server import build_server

    server = build_server(env_kwargs={"n_symbols": 2, "n_days": 20, "seed": 4})
    spec = json.loads(_call(server, "spec", {}))
    assert "decision_space" in spec
    assert "observation_space" in spec
    assert len(spec["symbols"]) == 2
