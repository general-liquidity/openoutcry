//! WASM façade over [`openoutcry`] — the trading-agent environment (leak-free
//! point-in-time simulator + reference baselines + the recompute-to-verify path) —
//! so TypeScript/Bun (the published `@general-liquidity/openoutcry` npm package)
//! drives the **identical** engine as the Rust harness, and an in-process baseline
//! run can never drift from its npm twin.
//!
//! Every entry point is a pure JSON-in / JSON-out function with a host-testable
//! `*_json` core and, under `wasm32`, a `wasm-bindgen` export of the same name.
//! There is exactly one implementation of the engine math; this only marshals.
//!
//! ## JSON shapes
//!
//! * **Dataset** — `{ "dates": [str], "closes": { sym: [f64] }, "dividends": { sym: [f64] } }`
//!   (the leak-free price panel; `dividends` may be omitted).
//! * **Run** (every backtest's output) —
//!   `{ "returns": [f64], "trace": { "events": [..] }, "confidences": [f64], "outcomes": [bool], "cost": f64 }`.
//! * **CostModel** — `{ fee_bps, slippage_bps, impact_bps, financing_bps, max_participation }`,
//!   all optional; omitted fields fall back to the engine default.
//! * **Window** — `{ "start": usize, "end": usize }` (half-open `[start, end)`).
#![forbid(unsafe_code)]

use openoutcry::scenario_gen::generate_scenario;
use openoutcry::{
    replay_run, run_backtest, tag_regime, walk_forward, Agent, BuyAndHold, CostModel, Dataset,
    HoldAgent, Momentum, RandomAgent, Regime, RunTrajectory, ScenarioSpec, Window,
};
use serde::Deserialize;

/// Parse an optional config blob: blank → `T::default()`.
fn parse_or_default<T: serde::de::DeserializeOwned + Default>(json: &str) -> Result<T, String> {
    if json.trim().is_empty() {
        Ok(T::default())
    } else {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

// --- Cost model: a serde mirror that defaults field-by-field to `CostModel::default()` ----

/// JSON view of [`CostModel`]; every field is optional and falls back to the engine
/// default, so `{}` (or a blank string) is the realistic default cost model.
#[derive(Deserialize, Default)]
struct CostsInput {
    fee_bps: Option<f64>,
    slippage_bps: Option<f64>,
    impact_bps: Option<f64>,
    financing_bps: Option<f64>,
    max_participation: Option<f64>,
}

impl CostsInput {
    fn into_model(self) -> CostModel {
        let d = CostModel::default();
        CostModel {
            fee_bps: self.fee_bps.unwrap_or(d.fee_bps),
            slippage_bps: self.slippage_bps.unwrap_or(d.slippage_bps),
            impact_bps: self.impact_bps.unwrap_or(d.impact_bps),
            financing_bps: self.financing_bps.unwrap_or(d.financing_bps),
            max_participation: self.max_participation.unwrap_or(d.max_participation),
            trf_cost: d.trf_cost,
        }
    }
}

fn parse_costs(json: &str) -> Result<CostModel, String> {
    Ok(parse_or_default::<CostsInput>(json)?.into_model())
}

// --- Dataset construction -----------------------------------------------------------------

/// Synthetic-dataset parameters. Defaults give a 4-symbol × 120-day mildly
/// momentum-autocorrelated panel (`seed = 0`).
#[derive(Deserialize)]
struct SyntheticParams {
    #[serde(default = "d_symbols")]
    n_symbols: usize,
    #[serde(default = "d_days")]
    n_days: usize,
    #[serde(default)]
    seed: u64,
}

fn d_symbols() -> usize {
    4
}
fn d_days() -> usize {
    120
}

impl Default for SyntheticParams {
    fn default() -> Self {
        Self {
            n_symbols: d_symbols(),
            n_days: d_days(),
            seed: 0,
        }
    }
}

impl SyntheticParams {
    fn build(&self) -> Dataset {
        Dataset::synthetic(self.n_symbols, self.n_days, self.seed)
    }
}

/// Where a run's price data comes from: either `synthetic` params OR raw `csv`
/// text (`date,symbol,close[,dividend]`). If both are absent, a default synthetic
/// panel is used; `csv` takes precedence when present.
#[derive(Deserialize, Default)]
struct DatasetSource {
    #[serde(default)]
    synthetic: Option<SyntheticParams>,
    #[serde(default)]
    csv: Option<String>,
}

impl DatasetSource {
    fn build(self) -> Result<Dataset, String> {
        match self.csv {
            Some(text) => Dataset::from_csv(&text),
            None => Ok(self.synthetic.unwrap_or_default().build()),
        }
    }
}

/// JSON view of [`Window`] (half-open `[start, end)`).
#[derive(Deserialize)]
struct WindowInput {
    start: usize,
    end: usize,
}

impl From<WindowInput> for Window {
    fn from(w: WindowInput) -> Self {
        Window {
            start: w.start,
            end: w.end,
        }
    }
}

// --- run_baseline -------------------------------------------------------------------------

/// Config for [`run_baseline_json`].
#[derive(Deserialize)]
struct BaselineConfig {
    /// `"buy_and_hold" | "hold" | "momentum" | "random"`.
    agent: String,
    #[serde(default)]
    dataset: DatasetSource,
    /// Defaults to `{ start: 20, end: dataset.len() }` (a 20-bar warm-up).
    #[serde(default)]
    window: Option<WindowInput>,
    /// Execution seed (governs slippage noise; also seeds the `random` agent).
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    costs: CostsInput,
    /// Trailing window for the `momentum` baseline (default 10).
    #[serde(default)]
    momentum_lookback: Option<usize>,
}

/// 20-bar warm-up so the trailing-history features have data (matches the engine's
/// `LOOKBACK`); clamped to the dataset so a tiny panel still yields a valid window.
const DEFAULT_WARMUP: usize = 20;

fn build_agent(
    name: &str,
    seed: u64,
    momentum_lookback: Option<usize>,
) -> Result<Box<dyn Agent>, String> {
    match name {
        "buy_and_hold" => Ok(Box::new(BuyAndHold)),
        "hold" => Ok(Box::new(HoldAgent)),
        "momentum" => Ok(Box::new(Momentum {
            lookback: momentum_lookback.unwrap_or(10),
        })),
        "random" => Ok(Box::new(RandomAgent::new(seed))),
        other => Err(format!(
            "unknown baseline agent {other:?} (expected buy_and_hold | hold | momentum | random)"
        )),
    }
}

/// Run a named in-process baseline over a dataset for a window + seed + costs,
/// returning the [`openoutcry::Run`] JSON (per-period returns + decision trace +
/// per-step confidences/outcomes).
pub fn run_baseline_json(config_json: &str) -> Result<String, String> {
    let cfg: BaselineConfig = serde_json::from_str(config_json).map_err(|e| e.to_string())?;
    let data = cfg.dataset.build()?;
    let window: Window = match cfg.window {
        Some(w) => w.into(),
        None => Window {
            start: DEFAULT_WARMUP.min(data.len().saturating_sub(1)),
            end: data.len(),
        },
    };
    let costs = cfg.costs.into_model();
    let mut agent = build_agent(&cfg.agent, cfg.seed, cfg.momentum_lookback)?;
    let run = run_backtest(&data, agent.as_mut(), window, cfg.seed, costs);
    serde_json::to_string(&run).map_err(|e| e.to_string())
}

// --- replay_run (recompute-to-verify) -----------------------------------------------------

/// Replay a captured [`RunTrajectory`]'s raw decisions through the identical engine
/// to regenerate its [`openoutcry::Run`] — the tamper-evidence path. The `Run` is
/// byte-identical to the one captured alongside the trajectory iff the `dataset` and
/// `costs` match what it was captured against.
pub fn replay_run_json(
    dataset_json: &str,
    trajectory_json: &str,
    costs_json: &str,
) -> Result<String, String> {
    let data: Dataset = serde_json::from_str(dataset_json).map_err(|e| e.to_string())?;
    let traj: RunTrajectory = serde_json::from_str(trajectory_json).map_err(|e| e.to_string())?;
    let costs = parse_costs(costs_json)?;
    let run = replay_run(&data, &traj, costs);
    serde_json::to_string(&run).map_err(|e| e.to_string())
}

// --- helpers ------------------------------------------------------------------------------

/// Build a deterministic synthetic [`Dataset`] from `{n_symbols, n_days, seed}` →
/// Dataset JSON. Blank input uses the defaults.
pub fn dataset_synthetic_json(params_json: &str) -> Result<String, String> {
    let params: SyntheticParams = parse_or_default(params_json)?;
    serde_json::to_string(&params.build()).map_err(|e| e.to_string())
}

/// The named adversarial stress suite (flash-crash, whipsaw, …) for a given seed →
/// JSON array of `{ "name": str, "dataset": Dataset }`. Input `{ "seed": u64 }`
/// (blank → seed 0).
pub fn stress_suite_json(params_json: &str) -> Result<String, String> {
    #[derive(Deserialize, Default)]
    struct SeedInput {
        #[serde(default)]
        seed: u64,
    }
    let input: SeedInput = parse_or_default(params_json)?;
    let suite: Vec<serde_json::Value> = Dataset::stress_suite(input.seed)
        .into_iter()
        .map(|(name, dataset)| serde_json::json!({ "name": name, "dataset": dataset }))
        .collect();
    serde_json::to_string(&suite).map_err(|e| e.to_string())
}

/// Generate disjoint walk-forward out-of-sample windows → JSON array of
/// `{ "start": usize, "end": usize }`. Input `{ n_days, warmup, test, step }`.
pub fn walk_forward_json(params_json: &str) -> Result<String, String> {
    #[derive(Deserialize)]
    struct WfParams {
        n_days: usize,
        warmup: usize,
        test: usize,
        step: usize,
    }
    let p: WfParams = serde_json::from_str(params_json).map_err(|e| e.to_string())?;
    let windows: Vec<serde_json::Value> = walk_forward(p.n_days, p.warmup, p.test, p.step)
        .into_iter()
        .map(|w| serde_json::json!({ "start": w.start, "end": w.end }))
        .collect();
    serde_json::to_string(&windows).map_err(|e| e.to_string())
}

/// Tag a window's coarse market regime → `{ "regime": "bull" | "bear" | "chop" }`.
/// Input `{ "dataset": Dataset, "window": { start, end } }`.
pub fn tag_regime_json(input_json: &str) -> Result<String, String> {
    #[derive(Deserialize)]
    struct TagInput {
        dataset: Dataset,
        window: WindowInput,
    }
    let input: TagInput = serde_json::from_str(input_json).map_err(|e| e.to_string())?;
    let regime = tag_regime(&input.dataset, input.window.into());
    let label = match regime {
        Regime::Bull => "bull",
        Regime::Bear => "bear",
        Regime::Chop => "chop",
    };
    serde_json::to_string(&serde_json::json!({ "regime": label })).map_err(|e| e.to_string())
}

/// Generate a Procgen-style procedural scenario → Dataset JSON. Input
/// `{ "spec": ScenarioSpec, "seed": u64 }`; a blank `spec` uses the defaults
/// (mild 4×120 Calm). Deterministic: identical `(spec, seed)` ⇒ byte-identical JSON
/// across every runtime (the cross-runtime generalization-reproducibility guarantee).
pub fn generate_scenario_json(input_json: &str) -> Result<String, String> {
    #[derive(Deserialize, Default)]
    struct GenInput {
        #[serde(default)]
        spec: ScenarioSpec,
        #[serde(default)]
        seed: u64,
    }
    let input: GenInput = parse_or_default(input_json)?;
    serde_json::to_string(&generate_scenario(&input.spec, input.seed)).map_err(|e| e.to_string())
}

/// The wasm-bindgen exports. Each returns the result JSON, or a `{"error":"..."}`
/// JSON object on failure (never throws across the boundary).
#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::wasm_bindgen;

    fn wrap(r: Result<String, String>) -> String {
        match r {
            Ok(s) => s,
            Err(e) => format!(
                "{{\"error\":{}}}",
                serde_json::to_string(&e).unwrap_or_default()
            ),
        }
    }

    #[wasm_bindgen]
    pub fn run_baseline(config_json: &str) -> String {
        wrap(super::run_baseline_json(config_json))
    }

    #[wasm_bindgen]
    pub fn replay_run(dataset_json: &str, trajectory_json: &str, costs_json: &str) -> String {
        wrap(super::replay_run_json(
            dataset_json,
            trajectory_json,
            costs_json,
        ))
    }

    #[wasm_bindgen]
    pub fn dataset_synthetic(params_json: &str) -> String {
        wrap(super::dataset_synthetic_json(params_json))
    }

    #[wasm_bindgen]
    pub fn stress_suite(params_json: &str) -> String {
        wrap(super::stress_suite_json(params_json))
    }

    #[wasm_bindgen]
    pub fn walk_forward(params_json: &str) -> String {
        wrap(super::walk_forward_json(params_json))
    }

    #[wasm_bindgen]
    pub fn tag_regime(input_json: &str) -> String {
        wrap(super::tag_regime_json(input_json))
    }

    #[wasm_bindgen]
    pub fn generate_scenario(input_json: &str) -> String {
        wrap(super::generate_scenario_json(input_json))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_baseline_buy_and_hold_produces_a_run() {
        let cfg = r#"{"agent":"buy_and_hold","dataset":{"synthetic":{"n_symbols":4,"n_days":120,"seed":11}},"seed":1}"#;
        let out = run_baseline_json(cfg).expect("run");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        // 120 days with a 20-bar warm-up → 100 returns.
        assert_eq!(v["returns"].as_array().unwrap().len(), 100);
        assert!(!v["trace"]["events"].as_array().unwrap().is_empty());
    }

    #[test]
    fn run_baseline_defaults_and_each_agent_name() {
        for agent in ["buy_and_hold", "hold", "momentum", "random"] {
            let cfg = format!(r#"{{"agent":"{agent}","seed":3}}"#);
            let out = run_baseline_json(&cfg).expect("run");
            let v: serde_json::Value = serde_json::from_str(&out).unwrap();
            // Default synthetic = 120 days, 20-bar warm-up.
            assert_eq!(
                v["returns"].as_array().unwrap().len(),
                100,
                "agent {agent} should produce 100 returns"
            );
        }
    }

    #[test]
    fn run_baseline_from_csv() {
        let csv = "date,symbol,close\\n2025-01-01,AAA,10\\n2025-01-02,AAA,11\\n2025-01-03,AAA,12\\n2025-01-04,AAA,13";
        let cfg = format!(
            r#"{{"agent":"buy_and_hold","dataset":{{"csv":"{csv}"}},"window":{{"start":1,"end":4}},"seed":0}}"#
        );
        let out = run_baseline_json(&cfg).expect("csv run");
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["returns"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn run_baseline_rejects_unknown_agent() {
        assert!(run_baseline_json(r#"{"agent":"nope"}"#).is_err());
    }

    #[test]
    fn replay_recomputes_the_same_run() {
        // Capture a baseline run's trajectory through the real engine, then replay it
        // via the JSON kernel and assert byte-identical returns.
        use openoutcry::{run_backtest_capture, Dataset as Ds, Window as W};
        let data = Ds::synthetic(4, 120, 11);
        let window = W {
            start: 20,
            end: 120,
        };
        let costs = CostModel::default();
        let (direct, traj) =
            run_backtest_capture(&data, &mut Momentum::default(), window, 3, costs);
        let dataset_json = serde_json::to_string(&data).unwrap();
        let traj_json = serde_json::to_string(&traj).unwrap();
        let out = replay_run_json(&dataset_json, &traj_json, "").expect("replay");
        let replayed: openoutcry::Run = serde_json::from_str(&out).unwrap();
        assert_eq!(
            serde_json::to_string(&direct).unwrap(),
            serde_json::to_string(&replayed).unwrap(),
            "replay must reproduce the captured run byte-for-byte"
        );
    }

    #[test]
    fn dataset_synthetic_is_deterministic_json() {
        let a = dataset_synthetic_json(r#"{"n_symbols":3,"n_days":40,"seed":99}"#).unwrap();
        let b = dataset_synthetic_json(r#"{"n_symbols":3,"n_days":40,"seed":99}"#).unwrap();
        assert_eq!(a, b);
        let v: serde_json::Value = serde_json::from_str(&a).unwrap();
        assert_eq!(v["dates"].as_array().unwrap().len(), 40);
        assert_eq!(v["closes"].as_object().unwrap().len(), 3);
    }

    #[test]
    fn stress_suite_and_walk_forward_and_regime() {
        let s = stress_suite_json(r#"{"seed":1}"#).unwrap();
        let sv: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(sv.as_array().unwrap().len(), 2);
        assert_eq!(sv[0]["name"], "flash_crash");

        let w = walk_forward_json(r#"{"n_days":200,"warmup":20,"test":60,"step":60}"#).unwrap();
        let wv: serde_json::Value = serde_json::from_str(&w).unwrap();
        assert_eq!(wv.as_array().unwrap().len(), 3);
        assert_eq!(wv[0]["start"], 20);

        let ds = dataset_synthetic_json(r#"{"n_symbols":2,"n_days":120,"seed":7}"#).unwrap();
        let tag_in = format!(r#"{{"dataset":{ds},"window":{{"start":0,"end":120}}}}"#);
        let r = tag_regime_json(&tag_in).unwrap();
        let rv: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(["bull", "bear", "chop"].contains(&rv["regime"].as_str().unwrap()));
    }

    #[test]
    fn bad_json_is_an_error_not_a_panic() {
        assert!(run_baseline_json("not json").is_err());
        assert!(replay_run_json("{}", "not json", "").is_err());
        assert!(walk_forward_json("{}").is_err());
    }

    /// Cross-language equivalence: the `run_baseline` JSON kernel (which the
    /// wasm-bindgen export wraps verbatim, and which wasm runs with deterministic
    /// IEEE-754 f64) must reproduce the **native** engine's `Run` for the same
    /// fixture — so a trajectory produced in the browser/Bun is byte-identical to
    /// one produced in Rust or Python (all three drive the one engine).
    #[test]
    fn wasm_kernel_matches_native_engine() {
        use openoutcry::{run_backtest, BuyAndHold, Dataset as Ds, Window as W};

        let data = Ds::synthetic(4, 120, 11);
        let window = W {
            start: 20,
            end: 120,
        };
        let native = run_backtest(&data, &mut BuyAndHold, window, 1, CostModel::default());

        let cfg = r#"{"agent":"buy_and_hold","dataset":{"synthetic":{"n_symbols":4,"n_days":120,"seed":11}},"window":{"start":20,"end":120},"seed":1}"#;
        let kernel: openoutcry::Run =
            serde_json::from_str(&run_baseline_json(cfg).expect("kernel run")).unwrap();

        assert_eq!(
            native.returns, kernel.returns,
            "wasm kernel must reproduce the native engine's returns byte-for-byte"
        );
        assert_eq!(
            native.trace.events, kernel.trace.events,
            "wasm kernel must reproduce the native engine's trace"
        );
    }

    /// The procedural-scenario JSON kernel (wrapped verbatim by the wasm-bindgen
    /// export) must reproduce the native [`generate_scenario`] byte-for-byte, and its
    /// FNV-1a fingerprint must equal the committed golden value — so a published
    /// generalization number computed in the browser/Bun is reproducible in Rust.
    #[test]
    fn generate_scenario_kernel_matches_native_and_golden() {
        use openoutcry::{generate_scenario, DistributionMode, ScenarioSpec};

        let spec = ScenarioSpec {
            distribution_mode: DistributionMode::Calm,
            n_symbols: 4,
            n_days: 120,
            ..ScenarioSpec::default()
        };
        let native = serde_json::to_string(&generate_scenario(&spec, 7)).unwrap();

        let input = format!(
            r#"{{"spec":{},"seed":7}}"#,
            serde_json::to_string(&spec).unwrap()
        );
        let kernel = generate_scenario_json(&input).expect("kernel scenario");

        assert_eq!(
            native, kernel,
            "wasm scenario kernel must reproduce the native generator byte-for-byte"
        );

        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in kernel.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        assert_eq!(
            h, 0xb7cf_976c_7121_9c52,
            "cross-runtime golden fingerprint drifted from the openoutcry crate's pin"
        );
    }
}
