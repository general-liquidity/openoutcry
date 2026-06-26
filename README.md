# OpenOutcry

**A leak-free, point-in-time environment for trading agents — and the language-agnostic contract they speak.**

OpenOutcry is the open-outcry trading floor for agents: the harness hands the agent a point-in-time
`Observation`, the agent returns a `Decision`, repeat. Look-ahead is *structurally impossible* (the
environment owns the time cursor and never hands out a future bar), and trajectories are
recompute-from-raw-decisions, so an agent cannot lie about its returns.

The strategic bet is **interface ownership**: if every trading agent in the open ecosystem conforms to
the OpenOutcry `Observation`/`Decision` contract, then [SharpeBench](https://crates.io/crates/sharpebench-core)
is the natural scorer and the whole funnel — env → trajectory → score → leaderboard — runs on one
standard. The interface *is* the product; the simulator is the credibility behind it.

## The agent contract (the standard)

An agent is just a program that reads an `Observation` and writes a `Decision` — in any language, over
stdio (newline-JSON) or HTTP (`POST /decide`):

```jsonc
// Observation (harness → agent)
{ "date": "2025-01-02", "cash": 1.0,
  "symbols": [{ "symbol": "AAPL", "close_history": [187.2, 188.0, 190.4] }],
  "portfolio": [] }

// Decision (agent → harness)
{ "orders": [{ "symbol": "AAPL", "action": "buy", "target_weight": 0.5 }] }
```

The wire shape is versioned (`CONTRACT_VERSION`), evolves **additively only** (new fields are optional
with defaults), and is pinned by published JSON Schemas + a conformance kit. See
[`GOVERNANCE.md`](./GOVERNANCE.md) and [`contract/`](./contract/).

## The Gym lifecycle

The same engine SharpeBench runs *closed* (`run_backtest`), OpenOutcry exposes *open* — the caller drives it:

```rust
use openoutcry::{TradingEnv, Dataset, CostModel, Window, BuyAndHold, Agent};

let data = Dataset::synthetic(4, 120, 1);
let mut env = TradingEnv::new(data, Window { start: 20, end: 120 }, CostModel::default(), 7);
let mut agent = BuyAndHold;
let mut obs = env.reset();
loop {
    let decision = agent.decide(&obs);
    let step = env.step(decision);   // -> { observation, reward, done, info }
    obs = step.observation;
    if step.done { break; }
}
```

Both stepping surfaces call one shared `step_once` body, so a trajectory the env produces is
**byte-identical** to the equivalent `run_backtest` (enforced by `env_step_matches_run_backtest`).

## env → SharpeBench score

Run with capture, hand the trajectory to a *separate verifier* that recomputes the submission from the
raw decisions + frozen data alone, then score:

```
cargo run -p openoutcry --example score-a-trajectory
```

(see [`examples/score-a-trajectory.rs`](./examples/score-a-trajectory.rs)). Tamper with the trajectory
and the honest replay recomputes to different returns — this is the trust hinge of the whole ecosystem.

## Distribution

OpenOutcry ships from one Rust engine to every surface, with a language-agnostic wire contract on top so
agents can be written in anything:

- **Rust** — `openoutcry` (this crate).
- **TypeScript / npm** — `@general-liquidity/openoutcry` (the engine compiled to WASM).
- **Python / PyPI** — `openoutcry`, with a `gymnasium.Env` adapter and a PrimeIntellect `verifiers`
  environment so it plugs into the RL-training stacks directly.

Reference agents in Rust, TypeScript, and Python double as the conformance smoke tests
([`examples/`](./examples/)).

## Status

Incubating inside the SharpeBench workspace (it depends on the published `sharpebench-sim` engine).
It graduates to its own repository at distribution time, consuming the engine as a versioned crate.
