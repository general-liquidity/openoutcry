#!/usr/bin/env node
// Reference OpenOutcry agent (TypeScript) — the simplest thing that honors the contract.
//
// Transport: stdio. Reads one MarketObservation (JSON) per line on stdin and writes
// one Decision (JSON) per line on stdout. Strategy: equal-weight buy-and-hold — the
// baseline every real agent must beat. Fork it, replace `decide`.
//
//   node examples/reference-agent.ts   # then feed it MarketObservation JSON lines

import { createInterface } from "node:readline";

interface Order { symbol: string; action: string; target_weight: number; confidence: number; rationale: string; }
interface Decision { orders: Order[]; reasoning: string; }

/** MarketObservation -> Decision. Replace this body with your strategy. */
function decide(obs: { symbols: { symbol: string }[] }): Decision {
  const weight = 1.0 / Math.max(obs.symbols.length, 1);
  const orders = obs.symbols.map((s) => ({
    symbol: s.symbol, action: "buy", target_weight: weight, confidence: 0.5, rationale: "equal-weight hold",
  }));
  return { orders, reasoning: "equal-weight buy-and-hold" };
}

const rl = createInterface({ input: process.stdin });
rl.on("line", (raw) => {
  const line = raw.trim();
  if (line === "") return;
  let decision: Decision;
  try {
    decision = decide(JSON.parse(line));
  } catch {
    // Any bad input degrades to an empty-orders hold — never crashes the harness.
    decision = { orders: [], reasoning: "parse error -> hold" };
  }
  process.stdout.write(JSON.stringify(decision) + "\n");
});
