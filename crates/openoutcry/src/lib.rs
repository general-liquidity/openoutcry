//! # OpenOutcry — the trading-agent environment
//!
//! A leak-free, point-in-time market environment with a dead-simple, language-agnostic
//! agent contract: **the harness sends an [`MarketObservation`], the agent returns a
//! [`Decision`], repeat.** Look-ahead is structurally impossible — [`Dataset`] never hands
//! out a future bar — and trajectories are recompute-from-raw-decisions, so an agent cannot
//! lie about its returns.
//!
//! ## M0 — extraction scaffold
//!
//! This crate currently *re-exports* the point-in-time simulator that already lives in
//! `sharpebench-sim`, promoting it from "the benchmark's internal engine" to a standalone
//! environment whose public reason to exist is "run an agent in a market." The Gym-style
//! [`reset`]/[`step`] lifecycle, the frozen `CONTRACT_VERSION`, the WASM/npm + Python
//! surfaces, and the ecosystem wiring land in later milestones (see
//! `specs/SPEC-trading-agent-env.md`). v0.1 depends on `sharpebench-sim`; the `gl-sim-core`
//! extraction is a follow-up refactor, not a blocker.
//!
//! [`reset`]: https://gymnasium.farama.org
//! [`step`]: https://gymnasium.farama.org
#![forbid(unsafe_code)]

// --- The frozen wire-contract version (the standard OpenOutcry governs) --------------------

pub mod contract;
pub use contract::CONTRACT_VERSION;

// --- Point-in-time simulator surface (extraction from `sharpebench-sim`) ------------------

pub use sharpebench_sim::{
    // Trajectory capture + replay-recompute (the tamper-evidence path).
    replay_run,
    replay_submission,
    // The single-backtest engine + its window type.
    run_backtest,
    run_backtest_capture,
    // Walk-forward out-of-sample windows + regime tagging.
    tag_regime,
    walk_forward,
    // In-process reference/baseline agents + the trait they implement.
    Agent,
    BuyAndHold,
    // Leak-free data model + execution cost model.
    CostModel,
    CostProfile,
    Dataset,
    // External transports — a conforming agent is just a program that reads observations
    // (stdin / `POST /decide`) and writes decisions.
    ExternalAgent,
    HoldAgent,
    HttpAgent,
    Momentum,
    RandomAgent,
    Regime,
    // The Gym-style open-loop environment: `reset()` / `step()` over the same engine,
    // plus the named crisis-suite/scenario bundle.
    Scenario,
    StepInfo,
    StepResult,
    TeamAgent,
    TradingEnv,
    Window,
};

// --- The language-agnostic wire contract (the standard OpenOutcry governs) -----------------

pub use sharpebench_protocol::{
    Action, AgentTrajectory, Decision, DecisionStep, MarketObservation, Order, PositionState,
    RunTrajectory, SymbolSnapshot,
};

// --- The scored output (so callers read returns/trace without a second dependency) --------

pub use sharpebench_core::Run;
