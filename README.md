<!-- prettier-ignore -->
<div align="center">

# OpenOutcry

### The point-in-time environment for trading agents, and the contract they speak

*Whoever defines the agent interface owns the ecosystem. OpenOutcry is the leak-free trading floor every agent runs on, and the language-agnostic contract that makes any agent scorable.*

[![Crates.io](https://img.shields.io/crates/v/openoutcry?style=flat-square&logo=rust&color=DEA584&label=crates.io)](https://crates.io/crates/openoutcry)
[![npm](https://img.shields.io/npm/v/@general-liquidity/openoutcry?style=flat-square&logo=npm&color=CB3837)](https://www.npmjs.com/package/@general-liquidity/openoutcry)
[![PyPI](https://img.shields.io/pypi/v/openoutcry?style=flat-square&logo=pypi&logoColor=white&color=3776AB)](https://pypi.org/project/openoutcry/)
[![docs.rs](https://img.shields.io/docsrs/openoutcry?style=flat-square&logo=docsdotrs&label=docs.rs)](https://docs.rs/openoutcry)
[![CI](https://img.shields.io/github/actions/workflow/status/general-liquidity/openoutcry/ci.yml?style=flat-square&label=CI)](https://github.com/general-liquidity/openoutcry/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue?style=flat-square)](#license)
[![Unsafe](https://img.shields.io/badge/unsafe-forbidden-success?style=flat-square)](#architecture)

**[Why](#why) · [Quickstart](#quickstart) · [Train an agent](#train-an-agent) · [Surfaces](#use-it-from-anywhere) · [The contract](#the-agent-contract) · [Architecture](#architecture) · [Tech stack](#tech-stack)**

</div>

---

## Why

An eval is useless without an environment. A benchmark scores *trajectories*; something has to **produce** them. OpenOutcry is that producer: a leak-free, point-in-time market environment wrapped in a dead-simple, language-agnostic agent contract. **The harness sends an `Observation`, the agent returns a `Decision`, repeat.**

Two properties make it trustworthy rather than a toy:

1. **Look-ahead is structurally impossible.** The environment owns the time cursor and the data layer has *no API to read a future bar*, so an agent cannot peek by construction, not by policing.
2. **Trajectories are recompute-from-raw-decisions.** A run records only the agent's decisions; a separate verifier replays them against the frozen data to recompute a **byte-identical** result. A tampered trajectory recomputes differently, so an agent cannot lie about its returns.

The strategic bet is **interface ownership**: if every trading agent in the open ecosystem conforms to OpenOutcry's `Observation`/`Decision` contract, then [SharpeBench](https://crates.io/crates/sharpebench-core) is the natural scorer and the whole funnel (env, trajectory, score, leaderboard) runs on one standard. This is the OpenAI-Gym moment for trading agents. The interface *is* the product; the simulator is the credibility behind it.

> An agent is just a program that reads an observation and writes a decision, in any language. Conform to the contract, and you are scorable everywhere.

## Status: published, active (pre-1.0)

Published to **crates.io**, **npm**, and **PyPI** at **v0.1.0**, depending on the **published** `sharpebench-sim 0.0.7` engine (not a vendored copy). CI is green across four surfaces: Rust (`fmt`, `clippy -D warnings`, tests, a WASM target build), `cargo-deny`, the npm package, and the Python wheel (`maturin` + `pytest`).

Beyond the core `reset`/`step` lifecycle, the environment now ships a full **reinforcement-learning training surface**:

| Capability | What it is |
|:--|:--|
| **Procedural scenarios** | A seeded generator (`ScenarioSpec` / `generate_scenario`) using Procgen's integer-seed-interval model, with `Calm` / `Hard` / `Extreme` volatility-and-jump tiers and provably disjoint `train_test_split`, cross-runtime golden-hashed for byte-identical generation. |
| **Generalization gap** | `generalization_gap` measures train-vs-held-out deflated Sharpe over disjoint seed bands, turning "did it overfit" into one number scored by the SharpeBench kernel. |
| **`verifiers` training env** | A PrimeIntellect `verifiers` `MultiTurnEnv` that steps the market bar-by-bar, over a multi-row scenario `Dataset`, with an `XMLParser` decision protocol and a GRPO-safe bounded reward scored by the real SharpeBench `score_run` (deflated Sharpe, pass^k, process checks). |
| **Vectorized rollouts** | `VecTradingEnv` runs B scenario lanes in lockstep (rayon, structure-of-arrays JSON, gym3 same-step auto-reset), exposed as a `gymnasium.vector` env. |
| **Point-in-time-safe wrappers** | Causal normalize (no future-bar leak), `TimeLimit`, `FrameStack`, `RecordEpisodeStatistics`, plus a `check_env` conformance harness that *proves* seed-determinism. |
| **Harness integration** | An MCP server (`reset` / `step` / `spec` tools) so any MCP agent harness drives an episode with zero glue, a `LookaheadGuard` that refuses agent operations reading future data, versioned JSONL rollout traces that re-score offline through the SharpeBench kernel, and a cost-adjusted `RunMetrics` block for leaderboard ranking. |

**Not yet shipped:** the [PrimeIntellect](https://app.primeintellect.ai) Environments-Hub listing, multi-agent (shared-order-book) markets, and the Gordon conforming-agent adapter.

## Quickstart

```bash
cargo add openoutcry        # the Rust crate (re-exports the engine + the wire contract)
```

```rust
use openoutcry::{TradingEnv, Dataset, CostModel, Window, BuyAndHold, Agent};

let data = Dataset::synthetic(4, 120, 1);
let mut env = TradingEnv::new(data, Window { start: 20, end: 120 }, CostModel::default(), 7);
let mut agent = BuyAndHold;
let mut obs = env.reset();
loop {
    let decision = agent.decide(&obs);
    let step = env.step(decision);     // -> { observation, reward, done, info }
    obs = step.observation;
    if step.done { break; }
}
```

Both stepping surfaces (the open-loop `TradingEnv` and the closed-loop `run_backtest`) call one shared per-step body, so a trajectory the env produces is **byte-identical** to the equivalent backtest, enforced by a test.

```python
import gymnasium, openoutcry          # a first-class Gymnasium env
env = openoutcry.OpenOutcryEnv(n_symbols=4, n_days=120, seed=1)
obs, info = env.reset(seed=1)         # the seed selects the scenario
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Train an agent

The Python package is a PrimeIntellect `verifiers` environment, so an agent trains over a distribution of leak-free scenarios scored by the SharpeBench kernel:

```python
import openoutcry

# A multi-turn env that steps the market bar-by-bar over many seeded scenarios.
env = openoutcry.load_environment(n_windows=256, n_symbols=4, n_days=120)

# Measure overfitting directly: train vs a provably disjoint held-out seed band.
gap = openoutcry.generalization_gap(
    lambda seed: openoutcry.OpenOutcryEnv(n_symbols=4, n_days=120, seed=seed),
    n_train=64, n_test=64,
)
print(gap["gap_deflated_sharpe"])
```

A one-command [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) GRPO training config lives in [`examples/prime-rl/`](examples/prime-rl/); see [`docs/training.md`](docs/training.md) for the full loop (install, `vf-eval` baseline, `uv run rl`).

## Use it from anywhere

One Rust engine, scored identically across every surface, because they run the same code.

| Surface | Get it | What it is |
|:--|:--|:--|
| <img height="14" align="top" src="https://cdn.simpleicons.org/rust/DEA584" />&nbsp; **Rust crate** | `cargo add openoutcry` | The env, the procedural scenario generator, the batched `VecTradingEnv`, and the governed wire contract, re-exporting the leak-free engine. |
| <img height="14" align="top" src="https://cdn.simpleicons.org/pypi/3776AB" />&nbsp; **Python** | `pip install openoutcry` | A `gymnasium.Env` and `gymnasium.vector` adapter, the `verifiers` training environment, point-in-time-safe wrappers, rollout traces, and an MCP server, over the pyo3 binding. |
| <img height="14" align="top" src="https://cdn.simpleicons.org/npm/CB3837" />&nbsp; **npm** | `npm i @general-liquidity/openoutcry` | A typed JS/TS API over the engine compiled to WASM. |
| <img height="14" align="top" src="https://cdn.simpleicons.org/webassembly/654FF0" />&nbsp; **WASM** | `openoutcry-wasm` | The wasm-bindgen bridge the npm package and Gordon (Bun) embed. |

The agent itself can be written in **any** language: a conforming agent is a program that reads `MarketObservation` JSON (stdin or `POST /decide`) and writes `Decision` JSON. Reference agents in Rust, TypeScript, and Python double as the conformance smoke tests ([`crates/openoutcry/examples/`](crates/openoutcry/examples/)).

## The agent contract

The load-bearing standard. An `Observation` is point-in-time; a `Decision` is a set of target-weight orders:

```jsonc
// Observation (harness -> agent)
{ "date": "2025-01-02", "cash": 1.0,
  "symbols": [{ "symbol": "AAPL", "close_history": [187.2, 188.0, 190.4] }],
  "portfolio": [] }

// Decision (agent -> harness)
{ "orders": [{ "symbol": "AAPL", "action": "buy", "target_weight": 0.5 }] }
```

`CONTRACT_VERSION` tracks the wire shape and evolves **additively only** (new fields are optional with defaults), pinned by published JSON Schemas plus a conformance kit. The `validate_decision_json` boundary check rejects malformed decisions before they reach the engine. See [`crates/openoutcry/GOVERNANCE.md`](crates/openoutcry/GOVERNANCE.md) and [`crates/openoutcry/contract/`](crates/openoutcry/contract/).

## Architecture

A Rust [Cargo workspace](Cargo.toml), `#![forbid(unsafe_code)]`, that **depends on** the published SharpeBench engine rather than vendoring it, so the env and the benchmark cannot drift.

```
sharpebench-sim (published 0.0.7) ... the leak-free point-in-time engine
        |
   crates/openoutcry ......... the env, scenario generator, batched VecTradingEnv,
        |                       Gym reset/step, and the governed wire contract
        |-- crates/openoutcry-wasm   the engine as WASM (-> the npm package)
        |-- crates/openoutcry-py     pyo3 + gymnasium + verifiers + wrappers + MCP (maturin)
        +-- npm/openoutcry           the typed TS wrapper over the wasm
```

| Crate / package | Role |
|:--|:--|
| **`openoutcry`** | `TradingEnv` (`reset`/`step`), `VecTradingEnv` (batched), the procedural scenario generator, the `Scenario`/crisis-suite bundle, the re-exported wire contract plus scored `Run`, `CONTRACT_VERSION`, the conformance kit, and reference agents. |
| **`openoutcry-wasm`** | Pure JSON kernels (`run_baseline`, `replay_run`, `generate_scenario`, ...) plus wasm-bindgen exports, the identical engine for JS/TS. |
| **`openoutcry-py`** | A pyo3 extension exposing `TradingEnv` / `VecTradingEnv`, a `gymnasium.Env` and `gymnasium.vector` adapter, the `verifiers` training environment, wrappers, the generalization metric, rollout traces, and an MCP server (built by maturin). |
| **`@general-liquidity/openoutcry`** | The typed npm wrapper over the WASM kernel. |

## Tech stack

| Technology | Role |
|:--|:--|
| <img height="14" align="top" src="https://cdn.simpleicons.org/rust/DEA584" />&nbsp; [Rust](https://www.rust-lang.org) | The engine and env, pure `f64`, deterministic, no `unsafe`, batched with [rayon](https://github.com/rayon-rs/rayon) |
| <img height="14" align="top" src="https://cdn.simpleicons.org/webassembly/654FF0" />&nbsp; [WebAssembly](https://webassembly.org) | The engine for non-Rust hosts (`wasm-bindgen`) |
| <img height="14" align="top" src="https://cdn.simpleicons.org/typescript/3178C6" />&nbsp; [TypeScript](https://www.typescriptlang.org) | The typed npm package |
| <img height="14" align="top" src="https://cdn.simpleicons.org/python/3776AB" />&nbsp; [Python](https://www.python.org) | The pyo3 binding and Gymnasium adapter (built by [maturin](https://www.maturin.rs)) |
| <img height="14" align="top" src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/_static/img/gymnasium_black.svg" />&nbsp; [Gymnasium](https://gymnasium.farama.org) | The RL env standard the Python adapter conforms to (`reset`/`step`/spaces/`vector`) |
| <img height="14" align="top" src="https://github.com/PrimeIntellect-ai.png" />&nbsp; [Prime Intellect `verifiers`](https://github.com/PrimeIntellect-ai/verifiers) | The RLVR `Environment`/`Rubric` standard the training env conforms to |
| <img height="14" align="top" src="https://github.com/serde-rs.png" />&nbsp; [serde](https://serde.rs) | Deterministic JSON for the wire contract (`float_roundtrip` for byte-exact replay) |
| <img height="14" align="top" src="https://cdn.simpleicons.org/githubactions/2088FF" />&nbsp; GitHub Actions | CI: fmt, clippy, tests, wasm, cargo-deny, npm, maturin |

## Governance

The contract is governed in the open: additive-only evolution, a published deprecation window, and a conformance badge (see [`GOVERNANCE.md`](crates/openoutcry/GOVERNANCE.md)). Hosted by [General Liquidity](https://github.com/general-liquidity) to start; the credibility is the leak-free-by-construction substrate plus recompute-to-verify trajectories, not trust in the host. Gordon (GL's agent) conforms to the contract like any other entrant.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

---

<div align="center">
<sub><em>The trading floor every agent runs on.</em></sub>
</div>
