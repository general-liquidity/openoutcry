//! pyo3 bindings for OpenOutcry — the leak-free, point-in-time trading-agent
//! environment. The binding exchanges the wire-contract JSON at the boundary
//! (observations and decisions are JSON strings), which keeps the surface robust
//! and identical to the language-agnostic protocol any external agent speaks.

use openoutcry::vec_env::AutoresetMode;
use openoutcry::{
    generate_scenario, CostModel, Dataset, Decision, DistributionMode, LaneConfig, ScenarioSpec,
    TradingEnv as CoreEnv, VecTradingEnv as CoreVecEnv, Window,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;
use sharpebench_core::{score_agent, AgentSubmission, Run, ScoreConfig, Trace};

/// Parse the wire `distribution_mode` label, rejecting unknown tiers with a `ValueError`.
fn parse_distribution_mode(mode: &str) -> PyResult<DistributionMode> {
    match mode {
        "calm" => Ok(DistributionMode::Calm),
        "hard" => Ok(DistributionMode::Hard),
        "extreme" => Ok(DistributionMode::Extreme),
        other => Err(PyValueError::new_err(format!(
            "unknown distribution_mode {other:?} (expected calm | hard | extreme)"
        ))),
    }
}

/// Build the synthetic dataset for a tier: `Calm` is the mild panel; `Hard`/`Extreme`
/// post-process that same seeded panel (see `openoutcry::generate_scenario`).
fn build_dataset(n_symbols: usize, n_days: usize, seed: u64, mode: DistributionMode) -> Dataset {
    match mode {
        DistributionMode::Calm => Dataset::synthetic(n_symbols, n_days, seed),
        m => generate_scenario(
            &ScenarioSpec {
                n_symbols,
                n_days,
                distribution_mode: m,
                ..ScenarioSpec::default()
            },
            seed,
        ),
    }
}

/// Parse the wire `autoreset_mode` label, rejecting unknown modes with a `ValueError`.
fn parse_autoreset_mode(mode: &str) -> PyResult<AutoresetMode> {
    AutoresetMode::from_label(mode).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown autoreset_mode {mode:?} (expected next_step | same_step | disabled)"
        ))
    })
}

/// A Gym-style, steppable, leak-free trading environment.
///
/// Construct over a deterministic synthetic dataset (default) or, via the
/// [`from_csv`](Self::from_csv) classmethod, over a frozen long-format CSV.
/// `reset` returns the first observation as a JSON string; `step` takes a
/// decision JSON string and returns `(observation_json, reward, done, info_json)`.
#[pyclass(name = "TradingEnv")]
pub struct PyTradingEnv {
    inner: CoreEnv,
    seed: u64,
}

fn build_window(start: Option<usize>, end: Option<usize>, len: usize) -> Window {
    Window {
        start: start.unwrap_or(0),
        end: end.unwrap_or(len),
    }
}

fn build_costs(
    fee_bps: Option<f64>,
    slippage_bps: Option<f64>,
    impact_bps: Option<f64>,
    financing_bps: Option<f64>,
    max_participation: Option<f64>,
) -> CostModel {
    let d = CostModel::default();
    CostModel {
        fee_bps: fee_bps.unwrap_or(d.fee_bps),
        slippage_bps: slippage_bps.unwrap_or(d.slippage_bps),
        impact_bps: impact_bps.unwrap_or(d.impact_bps),
        financing_bps: financing_bps.unwrap_or(d.financing_bps),
        max_participation: max_participation.unwrap_or(d.max_participation),
    }
}

#[pymethods]
impl PyTradingEnv {
    /// Build an environment over a synthetic dataset of `n_symbols` × `n_days`,
    /// seeded by `seed`. The window defaults to the full series; costs default to
    /// [`CostModel::default`] unless overridden.
    #[new]
    #[pyo3(signature = (
        n_symbols = 4,
        n_days = 120,
        seed = 0,
        window_start = None,
        window_end = None,
        fee_bps = None,
        slippage_bps = None,
        impact_bps = None,
        financing_bps = None,
        max_participation = None,
        distribution_mode = "calm",
        exec_seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_symbols: usize,
        n_days: usize,
        seed: u64,
        window_start: Option<usize>,
        window_end: Option<usize>,
        fee_bps: Option<f64>,
        slippage_bps: Option<f64>,
        impact_bps: Option<f64>,
        financing_bps: Option<f64>,
        max_participation: Option<f64>,
        distribution_mode: &str,
        exec_seed: Option<u64>,
    ) -> PyResult<Self> {
        let mode = parse_distribution_mode(distribution_mode)?;
        let data = build_dataset(n_symbols, n_days, seed, mode);
        let window = build_window(window_start, window_end, data.len());
        if window.start >= window.end || window.end > data.len() {
            return Err(PyValueError::new_err(format!(
                "invalid window [{}, {}) over {} bars",
                window.start,
                window.end,
                data.len()
            )));
        }
        let costs = build_costs(
            fee_bps,
            slippage_bps,
            impact_bps,
            financing_bps,
            max_participation,
        );
        Ok(PyTradingEnv {
            inner: CoreEnv::new(data, window, costs, exec_seed.unwrap_or(seed)),
            seed,
        })
    }

    /// Build an environment over a frozen long-format CSV (`date,symbol,close[,dividend]`).
    #[classmethod]
    #[pyo3(signature = (
        csv_text,
        seed = 0,
        window_start = None,
        window_end = None,
        fee_bps = None,
        slippage_bps = None,
        impact_bps = None,
        financing_bps = None,
        max_participation = None,
        exec_seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_csv(
        _cls: &Bound<'_, PyType>,
        csv_text: &str,
        seed: u64,
        window_start: Option<usize>,
        window_end: Option<usize>,
        fee_bps: Option<f64>,
        slippage_bps: Option<f64>,
        impact_bps: Option<f64>,
        financing_bps: Option<f64>,
        max_participation: Option<f64>,
        exec_seed: Option<u64>,
    ) -> PyResult<Self> {
        let data = Dataset::from_csv(csv_text).map_err(PyValueError::new_err)?;
        let window = build_window(window_start, window_end, data.len());
        if window.start >= window.end || window.end > data.len() {
            return Err(PyValueError::new_err(format!(
                "invalid window [{}, {}) over {} bars",
                window.start,
                window.end,
                data.len()
            )));
        }
        let costs = build_costs(
            fee_bps,
            slippage_bps,
            impact_bps,
            financing_bps,
            max_participation,
        );
        Ok(PyTradingEnv {
            inner: CoreEnv::new(data, window, costs, exec_seed.unwrap_or(seed)),
            seed,
        })
    }

    /// The seed that generated this environment's scenario — out-of-band provenance,
    /// never a feature in the observation. The Python wrapper threads this into the
    /// `info` dict so a trajectory can be tied back to its generating seed.
    #[getter]
    fn scenario_seed(&self) -> u64 {
        self.seed
    }

    /// Reset to the start of the window; return the first point-in-time
    /// observation as a wire-format JSON string.
    fn reset(&mut self) -> PyResult<String> {
        let obs = self.inner.reset();
        serde_json::to_string(&obs).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Apply `decision_json` (a wire-format `Decision`) at the current bar and
    /// advance one step. Returns `(observation_json, reward, done, info_json)`,
    /// where `info_json` carries the post-step NAV and this step's process events.
    fn step(&mut self, decision_json: &str) -> PyResult<(String, f64, bool, String)> {
        let decision: Decision = serde_json::from_str(decision_json)
            .map_err(|e| PyValueError::new_err(format!("invalid decision JSON: {e}")))?;
        let res = self.inner.step(decision);
        let observation = serde_json::to_string(&res.observation)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let info = serde_json::json!({
            "nav": res.info.nav,
            "events": res.info.events,
        });
        Ok((observation, res.reward, res.done, info.to_string()))
    }
}

/// A vectorized, batched bank of `B` independent [`PyTradingEnv`]-equivalent lanes —
/// gym3's "vectorized-first" design. Each lane is a distinct synthetic scenario (one
/// seed per lane); the batch steps them together and recycles finished lanes per the
/// selected `autoreset_mode` (`next_step` default | `same_step` | `disabled`), so the
/// batch never stalls. Returns **structure-of-arrays batched JSON** (one call per batch,
/// not `B` separate calls).
#[pyclass(name = "VecTradingEnv")]
pub struct PyVecTradingEnv {
    inner: CoreVecEnv,
}

fn observations_to_json(
    observations: &[openoutcry::MarketObservation],
) -> PyResult<Vec<serde_json::Value>> {
    observations
        .iter()
        .map(|o| serde_json::to_value(o).map_err(|e| PyValueError::new_err(e.to_string())))
        .collect()
}

#[pymethods]
impl PyVecTradingEnv {
    /// Build a batch over `seeds` (one synthetic `n_symbols` × `n_days` lane per seed).
    /// All lanes share the window and cost overrides; window defaults to the full
    /// series and costs to [`CostModel::default`].
    #[new]
    #[pyo3(signature = (
        seeds,
        n_symbols = 4,
        n_days = 120,
        window_start = None,
        window_end = None,
        fee_bps = None,
        slippage_bps = None,
        impact_bps = None,
        financing_bps = None,
        max_participation = None,
        distribution_mode = "calm",
        exec_seed = None,
        autoreset_mode = "next_step",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        seeds: Vec<u64>,
        n_symbols: usize,
        n_days: usize,
        window_start: Option<usize>,
        window_end: Option<usize>,
        fee_bps: Option<f64>,
        slippage_bps: Option<f64>,
        impact_bps: Option<f64>,
        financing_bps: Option<f64>,
        max_participation: Option<f64>,
        distribution_mode: &str,
        exec_seed: Option<u64>,
        autoreset_mode: &str,
    ) -> PyResult<Self> {
        if seeds.is_empty() {
            return Err(PyValueError::new_err("seeds must be non-empty"));
        }
        let mode = parse_distribution_mode(distribution_mode)?;
        let reset_mode = parse_autoreset_mode(autoreset_mode)?;
        let window = match (window_start, window_end) {
            (None, None) => None,
            _ => {
                let w = Window {
                    start: window_start.unwrap_or(0),
                    end: window_end.unwrap_or(n_days),
                };
                if w.start >= w.end || w.end > n_days {
                    return Err(PyValueError::new_err(format!(
                        "invalid window [{}, {}) over {} bars",
                        w.start, w.end, n_days
                    )));
                }
                Some(w)
            }
        };
        let costs = build_costs(
            fee_bps,
            slippage_bps,
            impact_bps,
            financing_bps,
            max_participation,
        );
        let configs: Vec<LaneConfig> = seeds
            .iter()
            .map(|&seed| LaneConfig {
                n_symbols,
                n_days,
                seed,
                exec_seed,
                distribution_mode: mode,
                window,
                costs,
            })
            .collect();
        Ok(PyVecTradingEnv {
            inner: CoreVecEnv::from_configs(&configs).with_autoreset_mode(reset_mode),
        })
    }

    /// The number of lanes (`B`).
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.len()
    }

    /// The per-lane generating seeds (out-of-band provenance, never a feature).
    #[getter]
    fn scenario_seeds(&self) -> Vec<u64> {
        self.inner.seeds().to_vec()
    }

    /// The auto-reset mode applied to finished lanes (`next_step | same_step | disabled`).
    #[getter]
    fn autoreset_mode(&self) -> &'static str {
        self.inner.autoreset_mode().label()
    }

    /// Reset every lane; return `{ "n": B, "observations": [..] }` as a JSON string.
    fn reset_batch(&mut self) -> PyResult<String> {
        let observations = observations_to_json(&self.inner.reset_batch())?;
        let out = serde_json::json!({ "n": observations.len(), "observations": observations });
        Ok(out.to_string())
    }

    /// Step every lane. `decisions_json` is a JSON array of exactly `B` wire-format
    /// `Decision`s (`decisions[i]` drives lane `i`). Returns the structure-of-arrays
    /// batch as a JSON string:
    /// `{ "n", "observations", "rewards", "terminated", "truncated", "first", "infos",
    /// "final_obs", "final_info" }`. Recycling follows `autoreset_mode`: under `same_step`
    /// a finished lane's `observations[i]` is the reset t0 (`first[i]` true) and its terminal
    /// obs/info ride in `final_obs[i]`/`final_info[i]`; under `next_step` the terminal step is
    /// returned verbatim and the reset surfaces on the following step (`final_obs`/`final_info`
    /// are `null`).
    fn step_batch(&mut self, decisions_json: &str) -> PyResult<String> {
        let decisions: Vec<Decision> = serde_json::from_str(decisions_json)
            .map_err(|e| PyValueError::new_err(format!("invalid decisions JSON: {e}")))?;
        if decisions.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} decisions, got {}",
                self.inner.len(),
                decisions.len()
            )));
        }
        let step = self.inner.step_batch(&decisions);
        let observations = observations_to_json(&step.observations)?;
        let infos: Vec<serde_json::Value> = step
            .infos
            .iter()
            .map(|i| serde_json::json!({ "nav": i.nav, "events": i.events }))
            .collect();
        let mut final_obs = Vec::with_capacity(step.final_obs.len());
        for o in &step.final_obs {
            final_obs.push(match o {
                Some(obs) => {
                    serde_json::to_value(obs).map_err(|e| PyValueError::new_err(e.to_string()))?
                }
                None => serde_json::Value::Null,
            });
        }
        let final_info: Vec<serde_json::Value> = step
            .final_info
            .iter()
            .map(|i| match i {
                Some(info) => serde_json::json!({ "nav": info.nav, "events": info.events }),
                None => serde_json::Value::Null,
            })
            .collect();
        let out = serde_json::json!({
            "n": self.inner.len(),
            "observations": observations,
            "rewards": step.rewards,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "first": step.first,
            "infos": infos,
            "final_obs": final_obs,
            "final_info": final_info,
        });
        Ok(out.to_string())
    }
}

/// Score a sequence of per-period returns with the **same SharpeBench kernel** the
/// benchmark uses — the real deflated Sharpe / PSR / pass^k / process verdict, not a
/// Python reimplementation. `n_trials` folds in the agent's declared in-sample search
/// budget (more search ⇒ more deflation). Returns the `CompositeScore` as a JSON
/// string. This is what lets the `verifiers` rubric reward be *calibrated* rather than
/// approximate.
#[pyfunction]
#[pyo3(signature = (returns, n_trials = 0))]
fn score_run(returns: Vec<f64>, n_trials: u32) -> PyResult<String> {
    let outcomes: Vec<bool> = returns.iter().map(|r| *r > 0.0).collect();
    let confidences = vec![0.5_f64; returns.len()];
    let run = Run {
        returns,
        trace: Trace::default(),
        confidences,
        outcomes,
        cost: 0.0,
    };
    let submission = AgentSubmission {
        agent_id: "verifiers-rollout".to_string(),
        runs: vec![run],
        in_sample_trials: n_trials,
        candidates: Vec::new(),
    };
    let score = score_agent(&submission, &ScoreConfig::default());
    serde_json::to_string(&score).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Whether `decision_json` deserializes to the wire-contract [`Decision`] type — the
/// boundary `contains()` for actions, so a caller can validate an agent's output
/// against the action space without stepping the environment.
#[pyfunction]
fn validate_decision_json(decision_json: &str) -> bool {
    serde_json::from_str::<Decision>(decision_json).is_ok()
}

/// The `openoutcry_py` native module (imported as `openoutcry.openoutcry_py`).
#[pymodule]
fn openoutcry_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTradingEnv>()?;
    m.add_class::<PyVecTradingEnv>()?;
    m.add_function(wrap_pyfunction!(score_run, m)?)?;
    m.add_function(wrap_pyfunction!(validate_decision_json, m)?)?;
    m.add(
        "__doc__",
        "Native pyo3 bindings for the OpenOutcry trading-agent environment.",
    )?;
    Ok(())
}
