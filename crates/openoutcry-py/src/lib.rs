//! pyo3 bindings for OpenOutcry — the leak-free, point-in-time trading-agent
//! environment. The binding exchanges the wire-contract JSON at the boundary
//! (observations and decisions are JSON strings), which keeps the surface robust
//! and identical to the language-agnostic protocol any external agent speaks.

use openoutcry::exec_noise::{perturb as core_perturb_action, ExecNoise};
use openoutcry::lob_market::{OrderBook, OrderKind, Side};
use openoutcry::market::{MarketClearing, MarketParams};
use openoutcry::vec_env::AutoresetMode;
use openoutcry::{
    generate_scenario, CostModel, Dataset, Decision, DistributionMode, LaneConfig, Mandate,
    ScenarioSpec, TradingEnv as CoreEnv, VecTradingEnv as CoreVecEnv, Window,
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
        "cointegrated_pairs" => Ok(DistributionMode::CointegratedPairs),
        "regime_shift" => Ok(DistributionMode::RegimeShift),
        other => Err(PyValueError::new_err(format!(
            "unknown distribution_mode {other:?} (expected calm | hard | extreme | \
             cointegrated_pairs | regime_shift)"
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
        trf_cost: d.trf_cost,
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

    /// An O(1) snapshot of the mutable sim state (cursor + book) as JSON — the native
    /// checkpoint that replaces replay-from-decisions. Pair with [`restore_state`].
    fn clone_state(&self) -> PyResult<String> {
        let state = self.inner.clone_state();
        serde_json::to_string(&state).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Restore the env to a snapshot produced by [`clone_state`] in O(1) (no replay).
    fn restore_state(&mut self, state_json: &str) -> PyResult<()> {
        let state: openoutcry::EnvState = serde_json::from_str(state_json)
            .map_err(|e| PyValueError::new_err(format!("invalid env state: {e}")))?;
        self.inner.restore_state(state);
        Ok(())
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

/// Deterministically sample the scenario mandate from `seed`, returned as wire JSON
/// (`{"style","max_drawdown","benchmark","text"}`). Byte-identical to the Rust/WASM core.
#[pyfunction]
#[pyo3(signature = (seed, n_symbols = 4, allow_short = true))]
fn sample_mandate_json(seed: u64, n_symbols: usize, allow_short: bool) -> PyResult<String> {
    let m = openoutcry::mandate::sample_mandate(seed, n_symbols, allow_short);
    serde_json::to_string(&m).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The bounded breach penalty in `[0, 1]` for a mandate (wire JSON) over the recorded
/// per-bar `returns` and per-bar target-`weights` vectors. `0` = clean, `1` = fully breached.
#[pyfunction]
fn mandate_breach(mandate_json: &str, returns: Vec<f64>, weights: Vec<Vec<f64>>) -> PyResult<f64> {
    let m: Mandate = serde_json::from_str(mandate_json)
        .map_err(|e| PyValueError::new_err(format!("invalid mandate JSON: {e}")))?;
    Ok(openoutcry::mandate::mandate_breach(&m, &returns, &weights))
}

/// Perturb a `requested` action vector into a realized one via the deterministic Rust
/// `exec_noise` core — the trading analog of ALE sticky actions. With probability
/// `delay_prob` the `previous` action is returned (the order lands a bar late); otherwise
/// each weight gets bounded multiplicative uniform jitter scaled by `slippage_bps`. The
/// draw is keyed on `(seed, step_index)`, so it is byte-reproducible across runtimes.
#[pyfunction]
#[pyo3(signature = (seed, step_index, requested, previous, delay_prob = 0.0, slippage_bps = 0.0))]
fn perturb_action(
    seed: u64,
    step_index: u64,
    requested: Vec<f64>,
    previous: Vec<f64>,
    delay_prob: f64,
    slippage_bps: f64,
) -> Vec<f64> {
    core_perturb_action(
        seed,
        step_index,
        &requested,
        &previous,
        &ExecNoise {
            delay_prob,
            slippage_bps,
        },
    )
}

/// An endogenous price-impact **shared-book market** (M2): `N` agents trade one book per
/// symbol and their aggregate flow moves the cleared price (Kyle permanent + Almgren-
/// Chriss temporary impact). Distinct from the competition surface — here one agent's
/// orders move the price the others see. JSON at the boundary: `reset_market()` returns
/// the initial per-agent observations + market metadata; `step_market(orders_json)`
/// clears one bar and returns the per-agent fills/rewards/observations.
#[pyclass(name = "PyMarketClearing")]
pub struct PyMarketClearing {
    inner: MarketClearing,
    params: MarketParams,
    seed: u64,
}

#[pymethods]
impl PyMarketClearing {
    /// Synthetic `n_symbols` × `n_days` panel seeded by `seed`, for `n_agents` agents each
    /// starting with `capital` cash, under Kyle/Almgren-Chriss coefficients.
    #[new]
    #[pyo3(signature = (
        n_symbols = 4,
        n_days = 120,
        seed = 0,
        n_agents = 2,
        capital = 1.0,
        kyle_lambda = 0.1,
        eta = 0.05,
        volume_scale = 1.0,
        vol_scale = 0.0,
        distribution_mode = "calm",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_symbols: usize,
        n_days: usize,
        seed: u64,
        n_agents: usize,
        capital: f64,
        kyle_lambda: f64,
        eta: f64,
        volume_scale: f64,
        vol_scale: f64,
        distribution_mode: &str,
    ) -> PyResult<Self> {
        if n_agents < 1 {
            return Err(PyValueError::new_err("n_agents must be >= 1"));
        }
        let mode = parse_distribution_mode(distribution_mode)?;
        let data = build_dataset(n_symbols, n_days, seed, mode);
        let inner = MarketClearing::from_dataset(&data, n_agents, capital);
        let params = MarketParams {
            lambda: kyle_lambda,
            eta,
            volume_scale,
            vol_scale,
        };
        Ok(PyMarketClearing {
            inner,
            params,
            seed,
        })
    }

    #[getter]
    fn scenario_seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.symbols().to_vec()
    }

    #[getter]
    fn num_agents(&self) -> usize {
        self.inner.n_agents()
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.is_done()
    }

    /// Pre-trade per-agent observations + market metadata as a JSON string.
    fn reset_market(&self) -> PyResult<String> {
        let observations = observations_to_json(&self.inner.initial_observations())?;
        let out = serde_json::json!({
            "symbols": self.inner.symbols(),
            "n_agents": self.inner.n_agents(),
            "n_bars": self.inner.n_bars(),
            "start_bar": self.inner.start_bar(),
            "cursor": self.inner.cursor(),
            "capital": self.inner.capital(),
            "observations": observations,
        });
        Ok(out.to_string())
    }

    /// Clear one bar. `orders_json` is a JSON array of exactly `n_agents` target-weight
    /// vectors, each length `n_symbols` (canonical agent order, sorted symbol order).
    fn step_market(&mut self, orders_json: &str) -> PyResult<String> {
        let agent_orders: Vec<Vec<f64>> = serde_json::from_str(orders_json)
            .map_err(|e| PyValueError::new_err(format!("invalid orders JSON: {e}")))?;
        if agent_orders.len() != self.inner.n_agents() {
            return Err(PyValueError::new_err(format!(
                "expected {} agent order vectors, got {}",
                self.inner.n_agents(),
                agent_orders.len()
            )));
        }
        let n_sym = self.inner.symbols().len();
        if let Some(bad) = agent_orders.iter().position(|o| o.len() != n_sym) {
            return Err(PyValueError::new_err(format!(
                "agent {bad} order vector has {} weights, expected {n_sym}",
                agent_orders[bad].len()
            )));
        }
        let result = self.inner.step(&agent_orders, &self.params);
        let mut value =
            serde_json::to_value(&result).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let serde_json::Value::Object(ref mut map) = value {
            map.insert("cursor".to_string(), serde_json::json!(self.inner.cursor()));
        }
        Ok(value.to_string())
    }
}

fn parse_side(s: &str) -> PyResult<Side> {
    match s {
        "buy" => Ok(Side::Buy),
        "sell" => Ok(Side::Sell),
        other => Err(PyValueError::new_err(format!(
            "unknown side {other:?} (expected buy | sell)"
        ))),
    }
}

/// Parse one flat order JSON object into an `(agent, OrderKind)` tuple. Shape:
/// `{agent, kind: "limit"|"market"|"cancel"|"modify", side?, price_tick?, qty?, id?, new_qty?}`.
fn parse_order(v: &serde_json::Value) -> PyResult<(usize, OrderKind)> {
    let bad = |m: &str| PyValueError::new_err(format!("invalid order: {m}"));
    let agent = v["agent"].as_u64().ok_or_else(|| bad("agent"))? as usize;
    let kind = v["kind"].as_str().ok_or_else(|| bad("kind"))?;
    let order = match kind {
        "limit" => OrderKind::Limit {
            side: parse_side(v["side"].as_str().ok_or_else(|| bad("side"))?)?,
            price_tick: v["price_tick"].as_i64().ok_or_else(|| bad("price_tick"))?,
            qty: v["qty"].as_u64().ok_or_else(|| bad("qty"))?,
        },
        "market" => OrderKind::Market {
            side: parse_side(v["side"].as_str().ok_or_else(|| bad("side"))?)?,
            qty: v["qty"].as_u64().ok_or_else(|| bad("qty"))?,
        },
        "cancel" => OrderKind::Cancel {
            id: v["id"].as_u64().ok_or_else(|| bad("id"))?,
        },
        "modify" => OrderKind::Modify {
            id: v["id"].as_u64().ok_or_else(|| bad("id"))?,
            new_qty: v["new_qty"].as_u64().ok_or_else(|| bad("new_qty"))?,
        },
        other => return Err(bad(&format!("unknown kind {other:?}"))),
    };
    Ok((agent, order))
}

fn ladder_json(book: &OrderBook, levels: usize) -> serde_json::Value {
    serde_json::to_value(book.depth_ladder(levels)).unwrap_or(serde_json::Value::Null)
}

/// A deterministic integer-tick **limit-order-book** matching engine (M3): price-time
/// priority, market/limit/cancel/modify, partial fills, and a depth-ladder observation
/// (`mid` / `microprice` / `queue_imbalance`). JSON at the boundary so the tape is
/// byte-identical across runtimes. `step_book` applies a batch of agent orders in
/// canonical order and returns the resulting fills + post-step ladder.
#[pyclass(name = "PyOrderBook")]
pub struct PyOrderBook {
    inner: OrderBook,
    levels: usize,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    #[pyo3(signature = (tick_size = 0.01, levels = 10))]
    fn new(tick_size: f64, levels: usize) -> Self {
        PyOrderBook {
            inner: OrderBook::new(tick_size),
            levels,
        }
    }

    /// Clear the book and return the (empty) depth-ladder snapshot as JSON.
    fn reset_book(&mut self) -> String {
        self.inner = OrderBook::new(self.inner.tick_size());
        serde_json::json!({ "ladder": ladder_json(&self.inner, self.levels) }).to_string()
    }

    /// Apply a JSON array of flat agent orders (canonical order, price-time priority) and
    /// return `{ "fills": [Fill, ...], "ladder": LadderSnapshot }` as JSON.
    fn step_book(&mut self, orders_json: &str) -> PyResult<String> {
        let arr: Vec<serde_json::Value> = serde_json::from_str(orders_json)
            .map_err(|e| PyValueError::new_err(format!("invalid orders JSON: {e}")))?;
        let orders: Vec<(usize, OrderKind)> = arr
            .iter()
            .map(parse_order)
            .collect::<PyResult<Vec<_>>>()?;
        let fills = self.inner.step(&orders);
        let out = serde_json::json!({
            "fills": serde_json::to_value(&fills).map_err(|e| PyValueError::new_err(e.to_string()))?,
            "ladder": ladder_json(&self.inner, self.levels),
        });
        Ok(out.to_string())
    }

    /// The current depth-ladder snapshot as JSON (without stepping).
    fn ladder(&self) -> String {
        ladder_json(&self.inner, self.levels).to_string()
    }

    /// Single-price call-auction uncross over the current book (read-only): the batch
    /// open/close clearing the continuous book lacks. Returns
    /// `{ "clearing_tick": i64, "matched_qty": u64 }` at the volume-maximizing price, or
    /// `null` when nothing crosses. Does not mutate the book or the fill tape.
    fn uncross(&self) -> String {
        match self.inner.uncross() {
            Some((clearing_tick, matched_qty)) => {
                serde_json::json!({ "clearing_tick": clearing_tick, "matched_qty": matched_qty })
                    .to_string()
            }
            None => "null".to_string(),
        }
    }

    /// Read-only walk-the-book cost of filling `qty` on `side` (`"buy"` sweeps asks, `"sell"`
    /// sweeps bids) against current depth. Returns the `SweepCost` as JSON
    /// (`{ "avg_px_tick", "slippage_ticks", "filled_qty" }`) without mutating the book.
    fn sweep_cost(&self, side: &str, qty: u64) -> PyResult<String> {
        let cost = self.inner.sweep_cost(parse_side(side)?, qty);
        serde_json::to_string(&cost).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// The `openoutcry_py` native module (imported as `openoutcry.openoutcry_py`).
#[pymodule]
fn openoutcry_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTradingEnv>()?;
    m.add_class::<PyVecTradingEnv>()?;
    m.add_class::<PyMarketClearing>()?;
    m.add_class::<PyOrderBook>()?;
    m.add_function(wrap_pyfunction!(score_run, m)?)?;
    m.add_function(wrap_pyfunction!(validate_decision_json, m)?)?;
    m.add_function(wrap_pyfunction!(sample_mandate_json, m)?)?;
    m.add_function(wrap_pyfunction!(mandate_breach, m)?)?;
    m.add_function(wrap_pyfunction!(perturb_action, m)?)?;
    m.add(
        "__doc__",
        "Native pyo3 bindings for the OpenOutcry trading-agent environment.",
    )?;
    Ok(())
}
