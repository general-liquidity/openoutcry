#!/usr/bin/env python3
"""Reference OpenOutcry agent (Python) — the simplest thing that honors the contract.

Transport: stdio. Reads one MarketObservation (JSON) per line on stdin and writes one
Decision (JSON) per line on stdout. Strategy: equal-weight buy-and-hold — the baseline
every real agent must beat. Fork it, replace ``decide``.

    python examples/reference-agent.py   # then feed it MarketObservation JSON lines
"""
import json
import sys


def decide(obs):
    """MarketObservation -> Decision. Replace this body with your strategy."""
    symbols = obs["symbols"]
    weight = 1.0 / max(len(symbols), 1)
    orders = [
        {"symbol": s["symbol"], "action": "buy", "target_weight": weight,
         "confidence": 0.5, "rationale": "equal-weight hold"}
        for s in symbols
    ]
    return {"orders": orders, "reasoning": "equal-weight buy-and-hold"}


def main():
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            decision = decide(json.loads(line))
        except Exception:
            # Any bad input degrades to an empty-orders hold — never crashes the harness.
            decision = {"orders": [], "reasoning": "parse error -> hold"}
        sys.stdout.write(json.dumps(decision) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
