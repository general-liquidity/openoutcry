//! M0 smoke tests — prove the re-exported OpenOutcry surface is usable end-to-end from
//! outside the crate: the engine runs an agent in a market, and the language-agnostic wire
//! contract round-trips through the public types.

use openoutcry::{
    run_backtest, BuyAndHold, CostModel, Dataset, Decision, MarketObservation, Window,
};

/// The engine surface: run a baseline agent over a synthetic point-in-time dataset and get
/// per-period returns + a decision trace back.
#[test]
fn reexported_engine_runs_a_backtest() {
    let data = Dataset::synthetic(4, 120, 11);
    let mut agent = BuyAndHold;
    let run = run_backtest(
        &data,
        &mut agent,
        Window {
            start: 20,
            end: 120,
        },
        1,
        CostModel::default(),
    );
    assert_eq!(run.returns.len(), 100);
    assert!(!run.trace.events.is_empty());
}

/// The wire contract: the language-agnostic observation/decision JSON deserializes through
/// the re-exported protocol types — including the legacy decision shape (no `confidence` /
/// `rationale`), proving the additive-only `#[serde(default)]` discipline survives re-export.
#[test]
fn reexported_wire_contract_parses() {
    let obs_json = r#"{
        "date": "2025-01-02",
        "cash": 1.0,
        "symbols": [{ "symbol": "AAPL", "close_history": [187.2, 188.0, 190.4] }],
        "portfolio": []
    }"#;
    let obs: MarketObservation = serde_json::from_str(obs_json).expect("observation parses");
    assert_eq!(obs.symbols.len(), 1);

    let legacy_decision =
        r#"{ "orders": [{ "symbol": "AAPL", "action": "buy", "target_weight": 0.5 }] }"#;
    let decision: Decision = serde_json::from_str(legacy_decision).expect("legacy decision parses");
    assert_eq!(decision.orders.len(), 1);
}
