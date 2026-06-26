//! The frozen wire contract version.

/// The version of the **OpenOutcry Agent Interface** — the JSON wire shape an
/// agent and the harness exchange ([`MarketObservation`](crate::MarketObservation)
/// in, [`Decision`](crate::Decision) out).
///
/// This is **not** the crate / npm / PyPI package version. It tracks only the
/// *shape* of the contract and moves independently: shipping bug fixes, new
/// baselines, or extra re-exports bumps the package version but leaves
/// `CONTRACT_VERSION` untouched.
///
/// It bumps **only** on a breaking wire change. Additive changes — a new field
/// that is optional-with-default, so every existing agent keeps parsing — do
/// **not** bump it (that is the whole additive-only discipline; see
/// `GOVERNANCE.md`). Removing or retyping a field is a major bump and requires a
/// parallel `…V2` namespace rather than mutating this surface in place.
///
/// Implementers in any language target this version: passing the conformance kit
/// earns "conforms to the OpenOutcry Agent Interface v1.0".
pub const CONTRACT_VERSION: &str = "1.0";
