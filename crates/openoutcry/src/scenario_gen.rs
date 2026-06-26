//! Seeded procedural scenario generation — Procgen's `(start_level, num_levels)`
//! integer-seed-interval model, ported to the trading environment.
//!
//! A scenario is a **pure deterministic function of one `u64` seed**: the same
//! `(ScenarioSpec, seed)` always yields a byte-identical [`Dataset`]. Train/test
//! generalization is governed exactly the way Procgen governs it — by splitting the
//! seed *interval*, not the data — so an agent provably never trains on a test seed.
//!
//! `Calm` is the mild [`Dataset::synthetic`] panel; `Hard` / `Extreme` post-process
//! that same seeded panel — amplifying each bar's volatility around the symbol's mean
//! return and injecting seeded jumps. The transform uses only mul/add/div/`max` (no
//! `ln`/`exp`, which differ across libm implementations), so a generated panel is
//! byte-identical across Rust, WASM, and Python, and `n_symbols` / `n_days` are
//! honored by every tier.

use serde::{Deserialize, Serialize};

use crate::Dataset;

/// How adversarial a generated scenario is. The discrete difficulty tier maps each
/// seed onto a different family of the existing leak-free generators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistributionMode {
    /// Mild, momentum-autocorrelated synthetic panel ([`Dataset::synthetic`]).
    #[default]
    Calm,
    /// The seeded panel with amplified volatility and occasional jumps.
    Hard,
    /// The seeded panel with high volatility and frequent, larger jumps.
    Extreme,
    /// Consecutive symbol pairs share a common-trend random walk, so `y - beta*x`
    /// is a genuinely mean-reverting (stationary AR(1)) spread — a real
    /// cointegrated structure for a market-neutral mandate to exploit. The plain
    /// synthetic panel generates each symbol independently, so it has none.
    #[serde(rename = "cointegrated_pairs")]
    CointegratedPairs,
    /// A momentum→reversion regime change: a trending segment (high drift,
    /// persistent momentum) spliced to a mean-reverting/whipsaw segment, with the
    /// changepoint location drawn from the seed.
    #[serde(rename = "regime_shift")]
    RegimeShift,
}

/// A reproducible scenario family: a seed interval `[start_level, start_level +
/// num_levels)` (`num_levels == 0` ⇒ unbounded `[start_level, u64::MAX)`), the panel
/// dimensions, and the difficulty tier. `Default` is the mild 4×120 Calm family over
/// the unbounded interval (matching the synthetic façade defaults).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioSpec {
    pub start_level: u64,
    /// Size of the legal seed interval; `0` means unbounded.
    pub num_levels: u64,
    pub n_symbols: usize,
    pub n_days: usize,
    pub distribution_mode: DistributionMode,
}

impl Default for ScenarioSpec {
    fn default() -> Self {
        Self {
            start_level: 0,
            num_levels: 0,
            n_symbols: 4,
            n_days: 120,
            distribution_mode: DistributionMode::Calm,
        }
    }
}

/// SplitMix64 — the same dependency-free PRNG family [`Dataset::synthetic`] uses, so
/// jump injection stays cross-runtime deterministic (no transcendental calls).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed ^ 0x1234_5678_9ABC_DEF0)
    }

    /// Next draw in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Difficulty knobs for [`amplify`]: a `mode_salt` (so each tier draws a distinct jump
/// stream), the volatility multiplier on each bar's deviation from the symbol's mean
/// return, and the per-bar jump probability and magnitude.
struct AmplifyParams {
    mode_salt: u64,
    vol_mult: f64,
    jump_prob: f64,
    jump_size: f64,
}

/// Post-process a base panel: amplify each symbol's per-bar volatility around its mean
/// return and inject seeded jumps. Returns are recomputed simple (mul/add/div/`max`
/// only — no `ln`/`exp`), so the result is byte-identical across runtimes and prices
/// stay strictly positive (the `max(-0.95)` floor keeps `1 + r > 0`).
fn amplify(mut base: Dataset, seed: u64, p: AmplifyParams) -> Dataset {
    let mut rng = SplitMix64::new(seed ^ p.mode_salt);
    for series in base.closes.values_mut() {
        if series.len() < 2 {
            continue;
        }
        let rets: Vec<f64> = (1..series.len())
            .map(|t| series[t] / series[t - 1] - 1.0)
            .collect();
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let mut price = series[0];
        for (i, r) in rets.iter().enumerate() {
            let jump = if rng.next_unit() < p.jump_prob {
                if rng.next_unit() < 0.5 {
                    p.jump_size
                } else {
                    -p.jump_size
                }
            } else {
                0.0
            };
            let adjusted = (mean + p.vol_mult * (r - mean) + jump).max(-0.95);
            price *= 1.0 + adjusted;
            series[i + 1] = price;
        }
    }
    base
}

/// A strictly-positive price floor — the level analog of the `max(-0.95)` return
/// floor, keeping every generated close above zero with mul/add/`max` only.
const MIN_PRICE: f64 = 0.01;

/// Draw a symmetric shock in `[-amp, amp)` from one `next_unit` draw (mul/add only).
fn signed(rng: &mut SplitMix64, amp: f64) -> f64 {
    (rng.next_unit() - 0.5) * 2.0 * amp
}

/// Overwrite the panel with **cointegrated pairs**. Consecutive symbols (sorted
/// order) form a pair sharing a per-pair common-trend random walk `f_t` (an I(1)
/// integrated level): `x_t = a_x*f_t + idio_x`, `y_t = a_y*f_t + idio_y + spread_t`,
/// where `idio_*` are stationary white noise and `spread_t = phi*spread_{t-1} +
/// noise` is a stationary AR(1) (`phi = 0.85`). With `beta = a_y/a_x` the common
/// trend cancels in `y - beta*x`, leaving the stationary spread — so the pair is
/// genuinely cointegrated. Pure mul/add/div/`max`; an odd trailing symbol gets a
/// standalone positive random walk. Prices stay `>= MIN_PRICE`.
#[allow(clippy::needless_range_loop)] // indices pair two distinct series at bar `t`
fn cointegrated_pairs(mut base: Dataset, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed ^ 0x436F_5061_6972_735F);
    let mut series: Vec<&mut Vec<f64>> = base.closes.values_mut().collect();
    let n = series.len();
    let mut i = 0;
    while i + 1 < n {
        let len = series[i].len().min(series[i + 1].len());
        let a_x = 0.8 + 0.4 * rng.next_unit();
        let a_y = 0.8 + 0.4 * rng.next_unit();
        let mut f = 100.0_f64;
        let mut spread = 0.0_f64;
        for t in 0..len {
            if t > 0 {
                f = (f + signed(&mut rng, 1.5)).max(MIN_PRICE);
            }
            let idio_x = signed(&mut rng, 0.6);
            let idio_y = signed(&mut rng, 0.6);
            spread = 0.85 * spread + signed(&mut rng, 1.0);
            series[i][t] = (a_x * f + idio_x).max(MIN_PRICE);
            series[i + 1][t] = (a_y * f + idio_y + spread).max(MIN_PRICE);
        }
        i += 2;
    }
    if i < n {
        let len = series[i].len();
        let mut f = 100.0_f64;
        for t in 0..len {
            if t > 0 {
                f = (f + signed(&mut rng, 1.5)).max(MIN_PRICE);
            }
            series[i][t] = f;
        }
    }
    base
}

/// Overwrite each symbol with a **momentum→reversion regime shift**: a trending
/// segment `[0, cp)` (positive drift + persistent momentum AR(1) on returns) spliced
/// to a whipsaw segment `[cp, len)` (negative return autocorrelation, no drift,
/// higher per-bar volatility). The changepoint `cp` is drawn per symbol from the seed
/// inside `[len/3, 2*len/3]`, so the first and last thirds are always in their
/// respective regimes. Returns are compounded with the `max(-0.95)` floor (mul/add
/// only — no `ln`/`exp`), keeping prices positive and cross-runtime identical.
#[allow(clippy::needless_range_loop)] // `t` and `t-1` lookback drive the recurrence
fn regime_shift(mut base: Dataset, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed ^ 0x5265_676D_5368_6674);
    for s in base.closes.values_mut() {
        let len = s.len();
        if len < 2 {
            continue;
        }
        let lo = len / 3;
        let span = ((2 * len) / 3 - lo).max(1);
        let cp = lo + (rng.next_unit() * span as f64) as usize;
        let mut price = 100.0_f64;
        s[0] = price;
        let mut prev_r = 0.0_f64;
        for t in 1..len {
            let r = if t < cp {
                0.004 + 0.7 * prev_r + signed(&mut rng, 0.01)
            } else {
                -0.6 * prev_r + signed(&mut rng, 0.03)
            };
            let r = r.max(-0.95);
            price *= 1.0 + r;
            s[t] = price;
            prev_r = r;
        }
    }
    base
}

/// Generate the [`Dataset`] for `spec` under `seed`. Deterministic: identical
/// `(spec, seed)` ⇒ identical `Dataset`. `n_symbols` / `n_days` are honored by every
/// tier; `Hard` / `Extreme` amplify the same seeded panel (see [`amplify`]), while
/// `CointegratedPairs` / `RegimeShift` overwrite it with a bespoke structured panel.
pub fn generate_scenario(spec: &ScenarioSpec, seed: u64) -> Dataset {
    let base = Dataset::synthetic(spec.n_symbols, spec.n_days, seed);
    match spec.distribution_mode {
        DistributionMode::Calm => base,
        DistributionMode::Hard => amplify(
            base,
            seed,
            AmplifyParams {
                mode_salt: 0x4861_7264_5f5f_5f5f,
                vol_mult: 1.8,
                jump_prob: 0.02,
                jump_size: 0.06,
            },
        ),
        DistributionMode::Extreme => amplify(
            base,
            seed,
            AmplifyParams {
                mode_salt: 0x4578_7472_5f5f_5f5f,
                vol_mult: 3.0,
                jump_prob: 0.06,
                jump_size: 0.13,
            },
        ),
        DistributionMode::CointegratedPairs => cointegrated_pairs(base, seed),
        DistributionMode::RegimeShift => regime_shift(base, seed),
    }
}

/// The concrete seed for the `index`-th level of `spec`'s interval, mirroring
/// Procgen: `start_level + (index % effective_num_levels)`, where the effective span
/// is `num_levels` (bounded) or the full `[start_level, u64::MAX)` width (unbounded).
pub fn level_seed(spec: &ScenarioSpec, index: u64) -> u64 {
    let span = if spec.num_levels == 0 {
        u64::MAX - spec.start_level
    } else {
        spec.num_levels
    };
    spec.start_level + (index % span)
}

/// Carve a **provably disjoint** test family from a (necessarily bounded) `train`
/// family: the test interval starts at `train.start_level + train.num_levels + gap`,
/// so no seed is shared. Panel dimensions and difficulty are inherited from `train`.
pub fn train_test_split(
    train: ScenarioSpec,
    n_test: u64,
    gap: u64,
) -> (ScenarioSpec, ScenarioSpec) {
    debug_assert!(
        train.num_levels > 0,
        "an unbounded train interval admits no disjoint test split"
    );
    let test_start = train.start_level + train.num_levels + gap;
    let test = ScenarioSpec {
        start_level: test_start,
        num_levels: n_test,
        ..train.clone()
    };
    debug_assert!(
        test.start_level >= train.start_level + train.num_levels,
        "test interval [{}, …) overlaps train [{}, {})",
        test.start_level,
        train.start_level,
        train.start_level + train.num_levels
    );
    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dependency-free FNV-1a/64 over bytes — the canonical-JSON fingerprint used to
    /// pin cross-runtime serialization determinism without adding a hash crate.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Golden fingerprint of `generate_scenario(&Calm{4×120}, seed=7)` serialized to
    /// JSON. A published generalization number must reproduce on any runtime, so this
    /// pins the FP/serialization determinism; the wasm crate asserts the same value.
    const GOLDEN_CALM_4X120_SEED7_FNV1A: u64 = 0xb7cf_976c_7121_9c52;
    const GOLDEN_HARD_4X120_SEED7_FNV1A: u64 = 0x2ef5_aff1_a716_05e6;
    const GOLDEN_EXTREME_4X120_SEED7_FNV1A: u64 = 0xb082_0c4d_2c73_7f88;
    const GOLDEN_COINTEGRATED_4X120_SEED7_FNV1A: u64 = 0xa3d2_2742_4ef0_5868;
    const GOLDEN_REGIME_4X120_SEED7_FNV1A: u64 = 0x8b82_2cf3_c9d3_038f;

    /// Mean per-symbol stdev of simple returns — a realized-volatility proxy.
    fn realized_vol(d: &Dataset) -> f64 {
        let mut acc = 0.0;
        for series in d.closes.values() {
            let rets: Vec<f64> = (1..series.len())
                .map(|t| series[t] / series[t - 1] - 1.0)
                .collect();
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
            acc += var.sqrt();
        }
        acc / d.closes.len() as f64
    }

    fn golden_spec() -> ScenarioSpec {
        ScenarioSpec {
            distribution_mode: DistributionMode::Calm,
            n_symbols: 4,
            n_days: 120,
            ..ScenarioSpec::default()
        }
    }

    #[test]
    fn generate_is_deterministic() {
        let spec = ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            ..ScenarioSpec::default()
        };
        let a = serde_json::to_string(&generate_scenario(&spec, 42)).unwrap();
        let b = serde_json::to_string(&generate_scenario(&spec, 42)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn distribution_modes_diverge() {
        let calm = ScenarioSpec::default();
        let hard = ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            ..ScenarioSpec::default()
        };
        let extreme = ScenarioSpec {
            distribution_mode: DistributionMode::Extreme,
            ..ScenarioSpec::default()
        };
        let cj = serde_json::to_string(&generate_scenario(&calm, 1)).unwrap();
        let hj = serde_json::to_string(&generate_scenario(&hard, 1)).unwrap();
        let ej = serde_json::to_string(&generate_scenario(&extreme, 1)).unwrap();
        assert_ne!(cj, hj);
        assert_ne!(hj, ej);
    }

    #[test]
    fn distribution_mode_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&DistributionMode::Extreme).unwrap(),
            "\"extreme\""
        );
    }

    #[test]
    fn level_seed_bounded_wraps_within_interval() {
        let spec = ScenarioSpec {
            start_level: 100,
            num_levels: 8,
            ..ScenarioSpec::default()
        };
        for index in 0..32 {
            let s = level_seed(&spec, index);
            assert!((100..108).contains(&s));
        }
        assert_eq!(level_seed(&spec, 0), 100);
        assert_eq!(level_seed(&spec, 8), 100);
        assert_eq!(level_seed(&spec, 9), 101);
    }

    #[test]
    fn level_seed_unbounded_is_offset() {
        let spec = ScenarioSpec {
            start_level: 5,
            num_levels: 0,
            ..ScenarioSpec::default()
        };
        assert_eq!(level_seed(&spec, 0), 5);
        assert_eq!(level_seed(&spec, 17), 22);
    }

    #[test]
    fn train_test_split_is_disjoint() {
        let train = ScenarioSpec {
            start_level: 0,
            num_levels: 1000,
            ..ScenarioSpec::default()
        };
        let (train, test) = train_test_split(train, 200, 50);
        let train_end = train.start_level + train.num_levels;
        assert!(test.start_level >= train_end);
        // No legal train seed equals any legal test seed.
        for ti in [0u64, 1, 999] {
            let train_seed = level_seed(&train, ti);
            for xi in [0u64, 1, 199] {
                assert_ne!(train_seed, level_seed(&test, xi));
            }
        }
        assert_eq!(test.start_level, 1050);
        assert_eq!(test.num_levels, 200);
    }

    #[test]
    fn golden_hash_is_stable() {
        let json = serde_json::to_string(&generate_scenario(&golden_spec(), 7)).unwrap();
        assert_eq!(fnv1a(json.as_bytes()), GOLDEN_CALM_4X120_SEED7_FNV1A);
    }

    #[test]
    fn golden_hash_hard_extreme_stable() {
        let hard = ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            ..golden_spec()
        };
        let extreme = ScenarioSpec {
            distribution_mode: DistributionMode::Extreme,
            ..golden_spec()
        };
        let hj = serde_json::to_string(&generate_scenario(&hard, 7)).unwrap();
        let ej = serde_json::to_string(&generate_scenario(&extreme, 7)).unwrap();
        assert_eq!(fnv1a(hj.as_bytes()), GOLDEN_HARD_4X120_SEED7_FNV1A);
        assert_eq!(fnv1a(ej.as_bytes()), GOLDEN_EXTREME_4X120_SEED7_FNV1A);
    }

    fn coint_spec() -> ScenarioSpec {
        ScenarioSpec {
            distribution_mode: DistributionMode::CointegratedPairs,
            ..golden_spec()
        }
    }

    fn regime_spec() -> ScenarioSpec {
        ScenarioSpec {
            distribution_mode: DistributionMode::RegimeShift,
            ..golden_spec()
        }
    }

    #[test]
    fn structured_modes_serialize_snake_case() {
        assert_eq!(
            serde_json::to_string(&DistributionMode::CointegratedPairs).unwrap(),
            "\"cointegrated_pairs\""
        );
        assert_eq!(
            serde_json::to_string(&DistributionMode::RegimeShift).unwrap(),
            "\"regime_shift\""
        );
    }

    #[test]
    fn structured_modes_are_deterministic() {
        for spec in [coint_spec(), regime_spec()] {
            let a = serde_json::to_string(&generate_scenario(&spec, 7)).unwrap();
            let b = serde_json::to_string(&generate_scenario(&spec, 7)).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn structured_modes_diverge_from_calm() {
        let cj = serde_json::to_string(&generate_scenario(&golden_spec(), 7)).unwrap();
        let pj = serde_json::to_string(&generate_scenario(&coint_spec(), 7)).unwrap();
        let rj = serde_json::to_string(&generate_scenario(&regime_spec(), 7)).unwrap();
        assert_ne!(cj, pj);
        assert_ne!(cj, rj);
        assert_ne!(pj, rj);
    }

    #[test]
    fn golden_hash_structured_modes_stable() {
        let pj = serde_json::to_string(&generate_scenario(&coint_spec(), 7)).unwrap();
        let rj = serde_json::to_string(&generate_scenario(&regime_spec(), 7)).unwrap();
        assert_eq!(fnv1a(pj.as_bytes()), GOLDEN_COINTEGRATED_4X120_SEED7_FNV1A);
        assert_eq!(fnv1a(rj.as_bytes()), GOLDEN_REGIME_4X120_SEED7_FNV1A);
    }

    /// Ordinary-least-squares slope of `y` on `x` (with intercept), mul/add/div only.
    fn ols_beta(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var = 0.0;
        for (xi, yi) in x.iter().zip(y) {
            cov += (xi - mx) * (yi - my);
            var += (xi - mx) * (xi - mx);
        }
        cov / var
    }

    /// Lo–MacKinlay variance ratio at horizon `q`: `Var(s_t - s_{t-q}) / (q *
    /// Var(s_t - s_{t-1}))`. `< 1` ⇒ mean-reverting, `≈ 1` ⇒ random walk, `> 1` ⇒
    /// trending. Pure mul/add/div.
    fn variance_ratio(s: &[f64], q: usize) -> f64 {
        let diff_var = |k: usize| {
            let d: Vec<f64> = (k..s.len()).map(|t| s[t] - s[t - k]).collect();
            let m = d.iter().sum::<f64>() / d.len() as f64;
            d.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / d.len() as f64
        };
        diff_var(q) / (q as f64 * diff_var(1))
    }

    #[test]
    fn cointegrated_spread_is_mean_reverting() {
        let d = generate_scenario(&coint_spec(), 7);
        let cols: Vec<&Vec<f64>> = d.closes.values().collect();
        // 4 symbols → 2 pairs; every pair's residual spread is bounded (VR < 1),
        // whereas each leg alone is an integrated random walk (VR ≈ 1, not < 1).
        let mut pairs = 0;
        let mut i = 0;
        while i + 1 < cols.len() {
            let x = cols[i];
            let y = cols[i + 1];
            let beta = ols_beta(x, y);
            let spread: Vec<f64> = x.iter().zip(y).map(|(xi, yi)| yi - beta * xi).collect();
            let vr = variance_ratio(&spread, 8);
            assert!(
                vr < 1.0,
                "pair {i} spread VR {vr} should be < 1 (mean-reverting)"
            );
            let leg_vr = variance_ratio(x, 8);
            assert!(
                leg_vr > vr,
                "integrated leg VR {leg_vr} should exceed spread VR {vr}"
            );
            pairs += 1;
            i += 2;
        }
        assert_eq!(pairs, 2);
    }

    #[test]
    fn regime_shift_halves_differ() {
        let d = generate_scenario(&regime_spec(), 7);
        // First third is always trending, last third always whipsaw (cp ∈
        // [len/3, 2*len/3]). Compare realized drift + vol over those two windows.
        let seg_stats = |series: &[f64], a: usize, b: usize| {
            let rets: Vec<f64> = (a + 1..b)
                .map(|t| series[t] / series[t - 1] - 1.0)
                .collect();
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var = rets.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / rets.len() as f64;
            (mean, var.sqrt())
        };
        let mut trend_drift = 0.0;
        let mut whip_drift = 0.0;
        let mut trend_vol = 0.0;
        let mut whip_vol = 0.0;
        let n = d.closes.len() as f64;
        for series in d.closes.values() {
            let len = series.len();
            let (td, tv) = seg_stats(series, 0, len / 3);
            let (wd, wv) = seg_stats(series, (2 * len) / 3, len);
            trend_drift += td;
            whip_drift += wd;
            trend_vol += tv;
            whip_vol += wv;
        }
        trend_drift /= n;
        whip_drift /= n;
        trend_vol /= n;
        whip_vol /= n;
        assert!(
            trend_drift > whip_drift,
            "trending drift {trend_drift} should exceed whipsaw drift {whip_drift}"
        );
        assert!(
            whip_vol > trend_vol,
            "whipsaw vol {whip_vol} should exceed trending vol {trend_vol}"
        );
    }

    #[test]
    fn structured_mode_prices_are_positive_and_finite() {
        for spec in [coint_spec(), regime_spec()] {
            let d = generate_scenario(&spec, 3);
            for series in d.closes.values() {
                for &p in series {
                    assert!(p.is_finite() && p > 0.0, "price {p} must be finite and > 0");
                }
            }
        }
    }

    #[test]
    fn realized_vol_increases_with_difficulty() {
        let spec = |m| ScenarioSpec {
            distribution_mode: m,
            ..ScenarioSpec::default()
        };
        let calm = realized_vol(&generate_scenario(&spec(DistributionMode::Calm), 7));
        let hard = realized_vol(&generate_scenario(&spec(DistributionMode::Hard), 7));
        let extreme = realized_vol(&generate_scenario(&spec(DistributionMode::Extreme), 7));
        assert!(calm < hard, "calm {calm} should be < hard {hard}");
        assert!(hard < extreme, "hard {hard} should be < extreme {extreme}");
    }
}
