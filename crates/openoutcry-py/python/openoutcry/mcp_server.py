"""An MCP server exposing OpenOutcry's reset/step/spec over streamable HTTP.

A thin :class:`~mcp.server.fastmcp.FastMCP` server so any MCP-capable harness — e.g.
PrimeIntellect ``rlm``, which reads a standard ``mcpServers`` URL map over streamable
HTTP — can drive an :class:`~openoutcry.gym.OpenOutcryEnv` episode with zero glue.

``mcp`` is an OPTIONAL dependency (guarded exactly like ``verifiers_env`` guards
``verifiers``): ``import openoutcry.mcp_server`` succeeds without it; :func:`build_server`
/ :func:`main` raise a clear error if called without it. The rest of the package is
unaffected.

env/invalid-decision are encoded as **structured content**, never raised — ``rlm`` turns
a tool exception into a ``RuntimeError`` that aborts the rollout, so ``step`` returns an
``{"error": ...}`` object on a malformed decision instead of throwing.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from .gym import OpenOutcryEnv
from .openoutcry_py import validate_decision_json

try:  # pragma: no cover - exercised only when mcp is installed
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    FastMCP = None  # type: ignore[assignment, misc]
    _HAS_MCP = False


def _obs_payload(obs: dict, symbols: list[str]) -> dict:
    """A compact, JSON-safe observation (it re-enters the agent's context every step)."""
    return {
        "symbols": list(symbols),
        "closes": [float(x) for x in obs["closes"]],
        "positions": [float(x) for x in obs["positions"]],
        "cash": float(obs["cash"][0]),
    }


def _decision_to_weights(decision_json: str, symbols: list[str]) -> np.ndarray:
    """Map a wire-contract ``Decision`` JSON to a target-weight vector over ``symbols``.

    Caller is expected to have validated ``decision_json`` first; unknown symbols are
    ignored and unmentioned symbols default to a 0.0 (flat) weight.
    """
    d = json.loads(decision_json)
    by_symbol: dict[str, float] = {}
    for order in d.get("orders", []) or []:
        sym = order.get("symbol")
        if sym is not None:
            by_symbol[sym] = float(order.get("target_weight", 0.0))
    return np.array([by_symbol.get(s, 0.0) for s in symbols], dtype=np.float64)


def build_server(env_kwargs: Optional[dict] = None) -> "FastMCP":
    """Build a ``FastMCP('openoutcry')`` exposing ``reset`` / ``step`` / ``spec``.

    ``env_kwargs`` is forwarded to :class:`~openoutcry.gym.OpenOutcryEnv` (e.g.
    ``n_symbols``, ``n_days``, ``seed``, ``allow_short``). The episode env is created
    lazily and held for the lifetime of the server. Raises ``RuntimeError`` if ``mcp`` is
    not installed.
    """
    if not _HAS_MCP:
        raise RuntimeError(
            "mcp is not installed. Install the 'mcp' SDK to run the OpenOutcry MCP "
            "server; the rest of the openoutcry package works without it."
        )

    server = FastMCP("openoutcry")
    state: dict[str, Any] = {"env": None, "kwargs": dict(env_kwargs or {})}

    def _env() -> OpenOutcryEnv:
        if state["env"] is None:
            state["env"] = OpenOutcryEnv(**state["kwargs"])
        return state["env"]

    @server.tool()
    def reset() -> str:
        """(Re)create and reset the episode; return the initial observation JSON."""
        env = _env()
        obs, _info = env.reset()
        return json.dumps(_obs_payload(obs, env.symbols))

    @server.tool()
    def step(decision_json: str) -> str:
        """Advance one bar with a wire-contract ``Decision`` JSON.

        Returns ``{observation, reward, terminated, truncated, info}``; a malformed
        decision returns ``{"error": ...}`` (structured content, not an exception).
        """
        env = _env()
        if not validate_decision_json(decision_json):
            return json.dumps(
                {"error": "decision_json does not match the Decision wire contract; call spec()"}
            )
        weights = _decision_to_weights(decision_json, env.symbols)
        obs, reward, terminated, truncated, info = env.step(weights)
        return json.dumps(
            {
                "observation": _obs_payload(obs, env.symbols),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        )

    @server.tool()
    def spec() -> str:
        """The decision/observation wire schema plus the episode's symbol axis."""
        env = _env()
        return json.dumps(
            {
                "decision_space": env.decision_space(),
                "observation_space": env.observation_space_schema(),
                "symbols": env.symbols,
            }
        )

    return server


def main() -> None:
    """Run the server over streamable HTTP (``python -m openoutcry.mcp_server``)."""
    build_server().run(transport="streamable-http")


if __name__ == "__main__":  # pragma: no cover
    main()
