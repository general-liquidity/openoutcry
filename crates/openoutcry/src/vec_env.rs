//! Vectorized, batched environment — gym3's "vectorized-first" design ported to the
//! trading env. A [`VecTradingEnv`] holds `B` independent [`TradingEnv`] lanes (each a
//! distinct seed/scenario/window) and steps them as a batch.
//!
//! The batch never stalls: when a lane finishes (blew up or ran out of bars) it is
//! reset **in place** and the observation it returns this step is already the new
//! episode's t0, with `first = true` (gym3 same-step auto-reset). The other lanes keep
//! stepping. There is exactly one engine — each lane is a real [`TradingEnv`], so the
//! batch math is the scalar math (B = 1 is just a single-lane batch).
//!
//! Determinism is structural: every lane owns its own seeded book/RNG, dataset, and
//! cursor, with no cross-lane shared mutable state, so the rayon-parallel step is
//! byte-identical to a serial step (asserted by `parallel_matches_serial`). On
//! `wasm32`, where threads do not build, the same loop runs serially.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::{CostModel, Dataset, Decision, MarketObservation, StepInfo, TradingEnv, Window};

/// Per-lane construction config: a synthetic panel of `n_symbols` × `n_days` seeded by
/// `seed`, over an optional `window` (full series when `None`) under `costs`.
#[derive(Clone, Debug)]
pub struct LaneConfig {
    pub n_symbols: usize,
    pub n_days: usize,
    pub seed: u64,
    pub window: Option<Window>,
    pub costs: CostModel,
}

impl LaneConfig {
    /// A lane over a synthetic `n_symbols` × `n_days` panel seeded by `seed`, with the
    /// full window and default costs (matching the scalar `TradingEnv` defaults).
    pub fn new(n_symbols: usize, n_days: usize, seed: u64) -> Self {
        Self {
            n_symbols,
            n_days,
            seed,
            window: None,
            costs: CostModel::default(),
        }
    }

    fn build(&self) -> TradingEnv {
        let data = Dataset::synthetic(self.n_symbols, self.n_days, self.seed);
        let window = self.window.unwrap_or(Window {
            start: 0,
            end: data.len(),
        });
        TradingEnv::new(data, window, self.costs, self.seed)
    }
}

/// The result of one batched step (structure-of-arrays, length `B`).
///
/// `terminated[i]` means lane `i` blew up (NAV ≤ 0); `truncated[i]` means it ran out of
/// bars; `first[i]` means a new episode just started this step (lane `i` was
/// auto-reset, so `observations[i]` is the new episode's t0). `rewards`/`infos` always
/// describe the step that just executed.
pub struct BatchStep {
    pub observations: Vec<MarketObservation>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub first: Vec<bool>,
    pub infos: Vec<StepInfo>,
}

/// One lane's per-step output before it is transposed into [`BatchStep`]'s SoA layout.
struct LaneOutcome {
    observation: MarketObservation,
    reward: f64,
    terminated: bool,
    truncated: bool,
    first: bool,
    info: StepInfo,
}

/// Step a single lane and apply same-step auto-reset. When the lane finishes, it is
/// reset in place and the returned observation is the new episode's t0 (`first = true`);
/// the terminal observation is dropped (only its reward/flags/info carry forward), so
/// no tail of the prior episode leaks into the next.
fn step_lane(env: &mut TradingEnv, decision: &Decision) -> LaneOutcome {
    let res = env.step(decision.clone());
    let terminated = res.info.nav <= 0.0;
    let truncated = res.done;
    if terminated || truncated {
        let observation = env.reset();
        LaneOutcome {
            observation,
            reward: res.reward,
            terminated,
            truncated,
            first: true,
            info: res.info,
        }
    } else {
        LaneOutcome {
            observation: res.observation,
            reward: res.reward,
            terminated,
            truncated,
            first: false,
            info: res.info,
        }
    }
}

impl BatchStep {
    fn from_outcomes(outcomes: Vec<LaneOutcome>) -> Self {
        let mut step = BatchStep {
            observations: Vec::with_capacity(outcomes.len()),
            rewards: Vec::with_capacity(outcomes.len()),
            terminated: Vec::with_capacity(outcomes.len()),
            truncated: Vec::with_capacity(outcomes.len()),
            first: Vec::with_capacity(outcomes.len()),
            infos: Vec::with_capacity(outcomes.len()),
        };
        for o in outcomes {
            step.observations.push(o.observation);
            step.rewards.push(o.reward);
            step.terminated.push(o.terminated);
            step.truncated.push(o.truncated);
            step.first.push(o.first);
            step.infos.push(o.info);
        }
        step
    }
}

/// A batch of `B` independent [`TradingEnv`] lanes stepped together.
pub struct VecTradingEnv {
    envs: Vec<TradingEnv>,
    seeds: Vec<u64>,
}

impl VecTradingEnv {
    /// Build a batch from per-lane configs. Each lane is an independent synthetic env.
    pub fn from_configs(configs: &[LaneConfig]) -> Self {
        let envs = configs.iter().map(LaneConfig::build).collect();
        let seeds = configs.iter().map(|c| c.seed).collect();
        VecTradingEnv { envs, seeds }
    }

    /// Build a batch from pre-constructed lanes (the most general entry point). `seeds`
    /// is out-of-band provenance threaded into `info`; it must match `envs` in length.
    pub fn from_envs(envs: Vec<TradingEnv>, seeds: Vec<u64>) -> Self {
        assert_eq!(
            envs.len(),
            seeds.len(),
            "envs and seeds must have equal length"
        );
        VecTradingEnv { envs, seeds }
    }

    /// The number of lanes (`B`).
    pub fn len(&self) -> usize {
        self.envs.len()
    }

    /// Whether the batch has zero lanes.
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// The per-lane generating seeds (out-of-band provenance, never a feature).
    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    /// Reset every lane to the start of its window; return each lane's first
    /// observation (in lane order).
    pub fn reset_batch(&mut self) -> Vec<MarketObservation> {
        self.envs.iter_mut().map(TradingEnv::reset).collect()
    }

    /// Step every lane with its decision (`decisions[i]` drives lane `i`) and apply
    /// same-step auto-reset. `decisions.len()` must equal the lane count.
    pub fn step_batch(&mut self, decisions: &[Decision]) -> BatchStep {
        assert_eq!(
            decisions.len(),
            self.envs.len(),
            "decisions length must equal the lane count"
        );

        #[cfg(not(target_arch = "wasm32"))]
        let outcomes: Vec<LaneOutcome> = self
            .envs
            .par_iter_mut()
            .zip(decisions.par_iter())
            .map(|(env, decision)| step_lane(env, decision))
            .collect();

        #[cfg(target_arch = "wasm32")]
        let outcomes: Vec<LaneOutcome> = self
            .envs
            .iter_mut()
            .zip(decisions.iter())
            .map(|(env, decision)| step_lane(env, decision))
            .collect();

        BatchStep::from_outcomes(outcomes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, Order, TradingEnv as ScalarEnv};

    fn configs(seeds: &[u64]) -> Vec<LaneConfig> {
        seeds.iter().map(|&s| LaneConfig::new(4, 60, s)).collect()
    }

    /// A flat target-weight decision over a known symbol axis — deterministic so the
    /// engine math (not the agent) is what the tests compare.
    fn decision_for(obs: &MarketObservation, weight: f64) -> Decision {
        let orders = obs
            .symbols
            .iter()
            .map(|s| Order {
                symbol: s.symbol.clone(),
                action: if weight > 0.0 {
                    Action::Buy
                } else {
                    Action::Hold
                },
                target_weight: weight,
                confidence: 0.5,
                rationale: String::new(),
            })
            .collect();
        Decision {
            orders,
            reasoning: String::new(),
        }
    }

    fn decisions_for(obs: &[MarketObservation], weight: f64) -> Vec<Decision> {
        obs.iter().map(|o| decision_for(o, weight)).collect()
    }

    #[test]
    fn b1_is_byte_identical_to_scalar_engine() {
        let seed = 11;
        let mut scalar = ScalarEnv::new(
            Dataset::synthetic(4, 60, seed),
            Window { start: 0, end: 60 },
            CostModel::default(),
            seed,
        );
        let mut batch = VecTradingEnv::from_configs(&configs(&[seed]));

        let mut s_obs = scalar.reset();
        let b_obs = batch.reset_batch();
        assert_eq!(
            serde_json::to_string(&s_obs).unwrap(),
            serde_json::to_string(&b_obs[0]).unwrap(),
            "reset observation must match the scalar env byte-for-byte"
        );

        // Step in lockstep over the whole episode; the engine math (reward, NAV, events,
        // and every non-terminal observation) must be byte-identical to the scalar env.
        loop {
            let s_dec = decision_for(&s_obs, 0.25);
            let b_dec = decisions_for(std::slice::from_ref(&s_obs), 0.25);
            let s_res = scalar.step(s_dec);
            let b_res = batch.step_batch(&b_dec);

            assert_eq!(s_res.reward, b_res.rewards[0], "reward divergence");
            assert_eq!(s_res.info.nav, b_res.infos[0].nav, "nav divergence");
            assert_eq!(
                s_res.info.events, b_res.infos[0].events,
                "per-step events divergence"
            );
            assert_eq!(s_res.done, b_res.truncated[0], "truncation divergence");

            if s_res.done {
                // The scalar env clamps to the terminal bar; the batch auto-resets and
                // surfaces a fresh t0 — the one intentional, documented difference.
                assert!(b_res.first[0], "finished lane must flag first=true");
                break;
            }
            assert!(!b_res.first[0], "mid-episode lane must not flag first");
            assert_eq!(
                serde_json::to_string(&s_res.observation).unwrap(),
                serde_json::to_string(&b_res.observations[0]).unwrap(),
                "non-terminal observation must match the scalar env byte-for-byte"
            );
            s_obs = s_res.observation;
        }
    }

    #[test]
    fn parallel_matches_serial() {
        let cfgs = configs(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut par = VecTradingEnv::from_configs(&cfgs);
        let mut ser = VecTradingEnv::from_configs(&cfgs);

        let mut par_obs = par.reset_batch();
        let _ = ser.reset_batch();

        for _ in 0..120 {
            let decs = decisions_for(&par_obs, 0.5);
            let par_step = par.step_batch(&decs);

            // The serial reference: the same per-lane work, looped, no rayon.
            let ser_step = BatchStep::from_outcomes(
                ser.envs
                    .iter_mut()
                    .zip(decs.iter())
                    .map(|(env, decision)| step_lane(env, decision))
                    .collect(),
            );

            assert_eq!(par_step.rewards, ser_step.rewards);
            assert_eq!(par_step.terminated, ser_step.terminated);
            assert_eq!(par_step.truncated, ser_step.truncated);
            assert_eq!(par_step.first, ser_step.first);
            for (a, b) in par_step.infos.iter().zip(ser_step.infos.iter()) {
                assert_eq!(a.nav, b.nav);
                assert_eq!(a.events, b.events);
            }
            for (a, b) in par_step
                .observations
                .iter()
                .zip(ser_step.observations.iter())
            {
                assert_eq!(
                    serde_json::to_string(a).unwrap(),
                    serde_json::to_string(b).unwrap()
                );
            }
            par_obs = par_step.observations;
        }
    }

    #[test]
    fn auto_reset_keeps_the_batch_running() {
        // Two short lanes that exhaust their windows quickly; after each finishes the
        // batch must keep producing observations and flag `first` on the reset step.
        let cfgs = vec![LaneConfig::new(3, 25, 1), LaneConfig::new(3, 25, 2)];
        let mut env = VecTradingEnv::from_configs(&cfgs);
        let mut obs = env.reset_batch();

        let mut resets = [0usize; 2];
        for _ in 0..120 {
            let decs = decisions_for(&obs, 0.1);
            let step = env.step_batch(&decs);
            assert_eq!(step.observations.len(), 2, "batch never stalls");
            for (lane, &first) in step.first.iter().enumerate() {
                if first {
                    resets[lane] += 1;
                    // A reset lane's observation is a fresh t0 — its NAV is back to the
                    // starting capital (never the blown-up / terminal value).
                    assert!(step.infos[lane].nav.is_finite());
                }
            }
            obs = step.observations;
        }
        assert!(
            resets[0] > 1 && resets[1] > 1,
            "each lane should auto-reset multiple times over 120 steps, got {resets:?}"
        );
    }

    #[test]
    fn auto_reset_observation_is_a_fresh_t0() {
        // A finished lane's reset observation must equal a brand-new env's reset obs
        // for the same seed (no leak of the prior episode tail).
        let seed = 7;
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, seed)]);
        let mut obs = env.reset_batch();
        let reference = {
            let mut fresh = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, seed)]);
            serde_json::to_string(&fresh.reset_batch()[0]).unwrap()
        };

        loop {
            let decs = decisions_for(&obs, 0.0);
            let step = env.step_batch(&decs);
            if step.first[0] {
                assert_eq!(
                    serde_json::to_string(&step.observations[0]).unwrap(),
                    reference,
                    "reset observation must equal a fresh env's t0"
                );
                break;
            }
            obs = step.observations;
        }
    }
}
