//! Vectorized, batched environment — gym3's "vectorized-first" design ported to the
//! trading env. A [`VecTradingEnv`] holds `B` independent [`TradingEnv`] lanes (each a
//! distinct seed/scenario/window) and steps them as a batch.
//!
//! Auto-reset is selectable via [`AutoresetMode`]: the default `NextStep` returns a
//! finished lane's terminal step verbatim and resets it on the *following* step
//! (Gymnasium 1.x default); `SameStep` resets in place and surfaces the terminal
//! obs/info through `final_obs`/`final_info` (gym3's `first` flag); `Disabled` never
//! auto-resets. The other lanes keep stepping regardless. There is exactly one engine —
//! each lane is a real [`TradingEnv`], so the batch math is the scalar math (B = 1 is
//! just a single-lane batch).
//!
//! Determinism is structural: every lane owns its own seeded book/RNG, dataset, and
//! cursor, with no cross-lane shared mutable state, so the rayon-parallel step is
//! byte-identical to a serial step (asserted by `parallel_matches_serial`). On
//! `wasm32`, where threads do not build, the same loop runs serially.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::{
    generate_scenario, CostModel, Dataset, Decision, DistributionMode, MarketObservation,
    ScenarioSpec, StepInfo, TradingEnv, Window,
};

/// Per-lane construction config: a panel of `n_symbols` × `n_days` under
/// `distribution_mode`, seeded by `seed`, over an optional `window` (full series when
/// `None`) under `costs`. `seed` drives the *scenario* (the price path); `exec_seed`
/// (when set) independently seeds *execution noise*, so the two streams can be split.
#[derive(Clone, Debug)]
pub struct LaneConfig {
    pub n_symbols: usize,
    pub n_days: usize,
    pub seed: u64,
    pub exec_seed: Option<u64>,
    pub distribution_mode: DistributionMode,
    pub window: Option<Window>,
    pub costs: CostModel,
}

impl LaneConfig {
    /// A `Calm` lane over a synthetic `n_symbols` × `n_days` panel seeded by `seed`, with
    /// the full window, default costs, and execution noise seeded by `seed` (matching the
    /// scalar `TradingEnv` defaults).
    pub fn new(n_symbols: usize, n_days: usize, seed: u64) -> Self {
        Self {
            n_symbols,
            n_days,
            seed,
            exec_seed: None,
            distribution_mode: DistributionMode::Calm,
            window: None,
            costs: CostModel::default(),
        }
    }

    fn build(&self) -> TradingEnv {
        let data = match self.distribution_mode {
            DistributionMode::Calm => Dataset::synthetic(self.n_symbols, self.n_days, self.seed),
            mode => generate_scenario(
                &ScenarioSpec {
                    n_symbols: self.n_symbols,
                    n_days: self.n_days,
                    distribution_mode: mode,
                    ..ScenarioSpec::default()
                },
                self.seed,
            ),
        };
        let window = self.window.unwrap_or(Window {
            start: 0,
            end: data.len(),
        });
        TradingEnv::new(
            data,
            window,
            self.costs,
            self.exec_seed.unwrap_or(self.seed),
        )
    }
}

/// How a finished lane is recycled, mirroring `gymnasium.vector.AutoresetMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoresetMode {
    /// Return the terminal step verbatim; reset on the *following* step with reward 0,
    /// `terminated`/`truncated` false, and `first = true` (Gymnasium 1.x default).
    #[default]
    NextStep,
    /// Reset in place this step; surface the terminal obs/info via `final_obs`/`final_info`
    /// and flag `first = true` (gym3's same-step convention).
    SameStep,
    /// Never auto-reset; a finished lane stays at its terminal bar.
    Disabled,
}

impl AutoresetMode {
    /// Parse the wire label (`"next_step" | "same_step" | "disabled"`).
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "next_step" => Some(Self::NextStep),
            "same_step" => Some(Self::SameStep),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }

    /// The wire label for this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::NextStep => "next_step",
            Self::SameStep => "same_step",
            Self::Disabled => "disabled",
        }
    }
}

/// The result of one batched step (structure-of-arrays, length `B`).
///
/// `terminated[i]` means lane `i` blew up (NAV ≤ 0); `truncated[i]` means it ran out of
/// bars; `first[i]` means a new episode just started this step (lane `i` was
/// auto-reset, so `observations[i]` is the new episode's t0). `rewards`/`infos` always
/// describe the step that just executed. Under [`AutoresetMode::SameStep`], a lane that
/// finished this step carries its terminal obs/info in `final_obs[i]`/`final_info[i]`
/// (both `None` otherwise).
pub struct BatchStep {
    pub observations: Vec<MarketObservation>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub first: Vec<bool>,
    pub infos: Vec<StepInfo>,
    pub final_obs: Vec<Option<MarketObservation>>,
    pub final_info: Vec<Option<StepInfo>>,
}

/// One lane's per-step output before it is transposed into [`BatchStep`]'s SoA layout.
struct LaneOutcome {
    observation: MarketObservation,
    reward: f64,
    terminated: bool,
    truncated: bool,
    first: bool,
    info: StepInfo,
    final_obs: Option<MarketObservation>,
    final_info: Option<StepInfo>,
}

/// Step a single lane under `mode`, threading the lane's `pending_reset` flag (only
/// `NextStep` uses it). When a lane finishes:
/// - `NextStep` returns its terminal step verbatim and arms `pending_reset`; the *next*
///   call resets it (reward 0, flags false, `first = true`), ignoring that step's decision.
/// - `SameStep` resets in place and stashes the terminal obs/info in `final_obs`/`final_info`
///   (`first = true`), so no tail of the prior episode leaks into the next observation.
/// - `Disabled` returns the terminal step verbatim and never resets.
fn step_lane(
    env: &mut TradingEnv,
    decision: &Decision,
    pending_reset: &mut bool,
    mode: AutoresetMode,
) -> LaneOutcome {
    if mode == AutoresetMode::NextStep && *pending_reset {
        *pending_reset = false;
        let observation = env.reset();
        let info = StepInfo {
            nav: observation.cash,
            events: Vec::new(),
        };
        return LaneOutcome {
            observation,
            reward: 0.0,
            terminated: false,
            truncated: false,
            first: true,
            info,
            final_obs: None,
            final_info: None,
        };
    }

    let res = env.step(decision.clone());
    let terminated = res.info.nav <= 0.0;
    let truncated = res.done;
    let ended = terminated || truncated;

    match mode {
        AutoresetMode::SameStep if ended => {
            let final_info = StepInfo {
                nav: res.info.nav,
                events: res.info.events.clone(),
            };
            let observation = env.reset();
            LaneOutcome {
                observation,
                reward: res.reward,
                terminated,
                truncated,
                first: true,
                info: res.info,
                final_obs: Some(res.observation),
                final_info: Some(final_info),
            }
        }
        _ => {
            if ended && mode == AutoresetMode::NextStep {
                *pending_reset = true;
            }
            LaneOutcome {
                observation: res.observation,
                reward: res.reward,
                terminated,
                truncated,
                first: false,
                info: res.info,
                final_obs: None,
                final_info: None,
            }
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
            final_obs: Vec::with_capacity(outcomes.len()),
            final_info: Vec::with_capacity(outcomes.len()),
        };
        for o in outcomes {
            step.observations.push(o.observation);
            step.rewards.push(o.reward);
            step.terminated.push(o.terminated);
            step.truncated.push(o.truncated);
            step.first.push(o.first);
            step.infos.push(o.info);
            step.final_obs.push(o.final_obs);
            step.final_info.push(o.final_info);
        }
        step
    }
}

/// A batch of `B` independent [`TradingEnv`] lanes stepped together.
pub struct VecTradingEnv {
    envs: Vec<TradingEnv>,
    seeds: Vec<u64>,
    mode: AutoresetMode,
    pending_reset: Vec<bool>,
}

impl VecTradingEnv {
    /// Build a batch from per-lane configs. Each lane is an independent synthetic env.
    pub fn from_configs(configs: &[LaneConfig]) -> Self {
        let envs: Vec<TradingEnv> = configs.iter().map(LaneConfig::build).collect();
        let seeds = configs.iter().map(|c| c.seed).collect();
        let pending_reset = vec![false; envs.len()];
        VecTradingEnv {
            envs,
            seeds,
            mode: AutoresetMode::default(),
            pending_reset,
        }
    }

    /// Build a batch from pre-constructed lanes (the most general entry point). `seeds`
    /// is out-of-band provenance threaded into `info`; it must match `envs` in length.
    pub fn from_envs(envs: Vec<TradingEnv>, seeds: Vec<u64>) -> Self {
        assert_eq!(
            envs.len(),
            seeds.len(),
            "envs and seeds must have equal length"
        );
        let pending_reset = vec![false; envs.len()];
        VecTradingEnv {
            envs,
            seeds,
            mode: AutoresetMode::default(),
            pending_reset,
        }
    }

    /// Select the auto-reset mode (builder form). Defaults to [`AutoresetMode::NextStep`].
    pub fn with_autoreset_mode(mut self, mode: AutoresetMode) -> Self {
        self.mode = mode;
        self
    }

    /// The auto-reset mode this batch applies to finished lanes.
    pub fn autoreset_mode(&self) -> AutoresetMode {
        self.mode
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
    /// observation (in lane order). Clears any pending deferred reset.
    pub fn reset_batch(&mut self) -> Vec<MarketObservation> {
        self.pending_reset.iter_mut().for_each(|p| *p = false);
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

        let mode = self.mode;

        #[cfg(not(target_arch = "wasm32"))]
        let outcomes: Vec<LaneOutcome> = self
            .envs
            .par_iter_mut()
            .zip(decisions.par_iter())
            .zip(self.pending_reset.par_iter_mut())
            .map(|((env, decision), pending)| step_lane(env, decision, pending, mode))
            .collect();

        #[cfg(target_arch = "wasm32")]
        let outcomes: Vec<LaneOutcome> = self
            .envs
            .iter_mut()
            .zip(decisions.iter())
            .zip(self.pending_reset.iter_mut())
            .map(|((env, decision), pending)| step_lane(env, decision, pending, mode))
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
        let mut batch = VecTradingEnv::from_configs(&configs(&[seed]))
            .with_autoreset_mode(AutoresetMode::SameStep);

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

            // The serial reference: the same per-lane work, looped, no rayon. Both
            // batches use the default mode, so the serial loop mirrors it exactly.
            let ser_step = BatchStep::from_outcomes(
                ser.envs
                    .iter_mut()
                    .zip(decs.iter())
                    .zip(ser.pending_reset.iter_mut())
                    .map(|((env, decision), pending)| {
                        step_lane(env, decision, pending, AutoresetMode::NextStep)
                    })
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

    fn fresh_t0(seed: u64, n_symbols: usize, n_days: usize) -> String {
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(n_symbols, n_days, seed)]);
        serde_json::to_string(&env.reset_batch()[0]).unwrap()
    }

    #[test]
    fn next_step_defers_reset_to_following_step() {
        // The default mode: a finished lane returns its terminal step verbatim, and the
        // reset surfaces on the *next* step (reward 0, flags false, first=true).
        let t0 = fresh_t0(7, 3, 22);
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, 7)]);
        assert_eq!(env.autoreset_mode(), AutoresetMode::NextStep);
        let mut obs = env.reset_batch();
        loop {
            let step = env.step_batch(&decisions_for(&obs, 0.0));
            if step.truncated[0] || step.terminated[0] {
                assert!(
                    !step.first[0],
                    "next_step must not reset on the ending step"
                );
                assert!(step.final_obs[0].is_none());
                assert_ne!(
                    serde_json::to_string(&step.observations[0]).unwrap(),
                    t0,
                    "ending step returns the terminal obs, not a fresh t0"
                );
                let next = env.step_batch(&decisions_for(&step.observations, 0.0));
                assert!(next.first[0], "next_step resets on the following step");
                assert_eq!(next.rewards[0], 0.0);
                assert!(!next.terminated[0] && !next.truncated[0]);
                assert_eq!(
                    serde_json::to_string(&next.observations[0]).unwrap(),
                    t0,
                    "the deferred-reset obs must equal a fresh t0"
                );
                break;
            }
            obs = step.observations;
        }
    }

    #[test]
    fn same_step_surfaces_final_obs_and_info() {
        // same_step resets in place; the terminal obs/info ride along in final_obs/info.
        let t0 = fresh_t0(7, 3, 22);
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, 7)])
            .with_autoreset_mode(AutoresetMode::SameStep);
        let mut obs = env.reset_batch();
        loop {
            let step = env.step_batch(&decisions_for(&obs, 0.0));
            if step.first[0] {
                assert_eq!(
                    serde_json::to_string(&step.observations[0]).unwrap(),
                    t0,
                    "same_step primary obs is the fresh t0"
                );
                let final_obs = step.final_obs[0].as_ref().expect("final_obs present");
                assert_ne!(
                    serde_json::to_string(final_obs).unwrap(),
                    t0,
                    "final_obs is the terminal obs, not t0"
                );
                assert!(step.final_info[0].is_some(), "final_info present");
                break;
            }
            assert!(step.final_obs[0].is_none());
            obs = step.observations;
        }
    }

    #[test]
    fn disabled_never_resets() {
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, 7)])
            .with_autoreset_mode(AutoresetMode::Disabled);
        let mut obs = env.reset_batch();
        let mut ended = false;
        for _ in 0..80 {
            let step = env.step_batch(&decisions_for(&obs, 0.0));
            ended |= step.truncated[0];
            assert!(!step.first[0], "disabled never flags first");
            assert!(step.final_obs[0].is_none());
            obs = step.observations;
        }
        assert!(ended, "the lane should exhaust its window within 80 steps");
    }

    #[test]
    fn split_exec_seed_keeps_dataset_changes_noise() {
        // Same scenario seed ⇒ identical price path; distinct exec_seed ⇒ distinct
        // execution noise, so realized rewards diverge once the lane trades.
        let base = LaneConfig::new(3, 60, 5);
        let cfg_a = LaneConfig {
            exec_seed: Some(1000),
            ..base.clone()
        };
        let cfg_b = LaneConfig {
            exec_seed: Some(2000),
            ..base
        };
        let mut a = VecTradingEnv::from_configs(std::slice::from_ref(&cfg_a));
        let mut b = VecTradingEnv::from_configs(std::slice::from_ref(&cfg_b));
        let mut obs_a = a.reset_batch();
        let mut obs_b = b.reset_batch();
        assert_eq!(
            serde_json::to_string(&obs_a[0]).unwrap(),
            serde_json::to_string(&obs_b[0]).unwrap(),
            "identical scenario seed ⇒ identical t0 price path"
        );
        let mut rewards_a = Vec::new();
        let mut rewards_b = Vec::new();
        for _ in 0..30 {
            let sa = a.step_batch(&decisions_for(&obs_a, 0.5));
            let sb = b.step_batch(&decisions_for(&obs_b, 0.5));
            rewards_a.push(sa.rewards[0]);
            rewards_b.push(sb.rewards[0]);
            obs_a = sa.observations;
            obs_b = sb.observations;
        }
        assert_ne!(
            rewards_a, rewards_b,
            "distinct exec seeds must perturb execution noise"
        );
    }

    #[test]
    fn hard_distribution_diverges_from_calm() {
        // The amplify transform preserves the first bar, so t0 matches; divergence shows
        // up along the path. Compare reward trajectories over the same scenario seed.
        let calm = LaneConfig::new(3, 60, 9);
        let hard = LaneConfig {
            distribution_mode: DistributionMode::Hard,
            ..calm.clone()
        };
        let mut c = VecTradingEnv::from_configs(std::slice::from_ref(&calm));
        let mut h = VecTradingEnv::from_configs(std::slice::from_ref(&hard));
        let mut obs_c = c.reset_batch();
        let mut obs_h = h.reset_batch();
        let mut rewards_c = Vec::new();
        let mut rewards_h = Vec::new();
        for _ in 0..40 {
            let sc = c.step_batch(&decisions_for(&obs_c, 0.5));
            let sh = h.step_batch(&decisions_for(&obs_h, 0.5));
            rewards_c.push(sc.rewards[0]);
            rewards_h.push(sh.rewards[0]);
            obs_c = sc.observations;
            obs_h = sh.observations;
        }
        assert_ne!(
            rewards_c, rewards_h,
            "hard tier must post-process the panel away from calm"
        );
    }
}
