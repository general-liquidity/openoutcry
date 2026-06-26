//! Conformance kit — the executable definition of "conforms to the OpenOutcry Agent
//! Interface v1.0". Every fixture under `contract/conformance/` is replayed through the
//! in-process reference agent ([`BuyAndHold`]) and the resulting [`Decision`] is checked
//! for well-formedness against the wire contract:
//!
//! - it deserializes through the protocol types,
//! - every `action` is one of the four enum variants,
//! - every `target_weight` is finite, and
//! - every order `symbol` is a subset of the symbols the observation actually offered.
//!
//! Fixtures that carry a `legacy_decision` (no `confidence` / `rationale` / `reasoning`)
//! also assert the additive-only `#[serde(default)]` discipline: the legacy shape still
//! parses and is itself well-formed. New fixtures are auto-discovered from the directory.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use openoutcry::{Action, Agent, BuyAndHold, Decision, MarketObservation, CONTRACT_VERSION};

/// The frozen wire version this kit certifies against. A bump here is a deliberate,
/// reviewed event (see `GOVERNANCE.md`), not an accident of editing.
#[test]
fn contract_version_is_frozen() {
    assert_eq!(CONTRACT_VERSION, "1.0");
}

/// Assert a decision is well-formed against the observed universe.
fn assert_well_formed(decision: &Decision, observed: &BTreeSet<String>, ctx: &str) {
    for order in &decision.orders {
        assert!(
            matches!(
                order.action,
                Action::Buy | Action::Sell | Action::Hold | Action::Close
            ),
            "{ctx}: action outside the enum",
        );
        assert!(
            order.target_weight.is_finite(),
            "{ctx}: non-finite target_weight for {}",
            order.symbol,
        );
        assert!(
            observed.contains(&order.symbol),
            "{ctx}: order on unobserved symbol {}",
            order.symbol,
        );
        assert!(
            (0.0..=1.0).contains(&order.confidence),
            "{ctx}: confidence {} out of [0, 1] for {}",
            order.confidence,
            order.symbol,
        );
    }
}

#[test]
fn conformance_fixtures_yield_well_formed_decisions() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contract/conformance");
    let mut fixtures = 0usize;

    for entry in fs::read_dir(&dir).expect("conformance directory exists") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = fs::read_to_string(&path).expect("fixture readable");
        let v: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: invalid JSON: {e}"));

        // Parse the observation through the protocol types (leg 1 of the badge).
        let obs: MarketObservation = serde_json::from_value(v["observation"].clone())
            .unwrap_or_else(|e| panic!("{name}: observation does not parse: {e}"));
        let observed: BTreeSet<String> = obs.symbols.iter().map(|s| s.symbol.clone()).collect();

        // Run the in-process reference agent over the observation (leg 2).
        let mut agent = BuyAndHold;
        let decision = agent.decide(&obs);
        assert_well_formed(&decision, &observed, &name);
        assert_eq!(
            decision.orders.len(),
            obs.symbols.len(),
            "{name}: BuyAndHold should weight every observed symbol",
        );

        // A fixture may also pin the legacy decision shape — it must still parse and be
        // well-formed, proving the additive-only discipline survives (leg 3).
        if let Some(legacy) = v.get("legacy_decision") {
            let parsed: Decision = serde_json::from_value(legacy.clone())
                .unwrap_or_else(|e| panic!("{name}: legacy_decision does not parse: {e}"));
            assert_well_formed(&parsed, &observed, &name);
        }

        fixtures += 1;
    }

    assert!(
        fixtures >= 4,
        "expected the full conformance kit, found {fixtures}"
    );
}
