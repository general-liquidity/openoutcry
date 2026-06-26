//! pyo3 bindings for OpenOutcry — the leak-free, point-in-time trading-agent
//! environment. The binding exchanges the wire-contract JSON at the boundary
//! (observations and decisions are JSON strings), which keeps the surface robust
//! and identical to the language-agnostic protocol any external agent speaks.

use openoutcry::{CostModel, Dataset, Decision, TradingEnv as CoreEnv, Window};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

/// A Gym-style, steppable, leak-free trading environment.
///
/// Construct over a deterministic synthetic dataset (default) or, via the
/// [`from_csv`](Self::from_csv) classmethod, over a frozen long-format CSV.
/// `reset` returns the first observation as a JSON string; `step` takes a
/// decision JSON string and returns `(observation_json, reward, done, info_json)`.
#[pyclass(name = "TradingEnv")]
pub struct PyTradingEnv {
    inner: CoreEnv,
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
    ) -> PyResult<Self> {
        let data = Dataset::synthetic(n_symbols, n_days, seed);
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
            inner: CoreEnv::new(data, window, costs, seed),
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
            inner: CoreEnv::new(data, window, costs, seed),
        })
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

/// The `openoutcry_py` native module (imported as `openoutcry.openoutcry_py`).
#[pymodule]
fn openoutcry_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTradingEnv>()?;
    m.add(
        "__doc__",
        "Native pyo3 bindings for the OpenOutcry trading-agent environment.",
    )?;
    Ok(())
}
