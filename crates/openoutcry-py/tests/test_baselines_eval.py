"""Tests for the mean-variance / Kelly baselines + per-regime eval + radar score.

Dependency-light logic (policy shapes, simplex projection, causal covariance, regime
labeling, radar anchoring) runs on numpy alone. The end-to-end sweeps and per-regime
rollouts additionally need the native binding and are skipped when it is absent.
"""

import importlib

import pytest

try:
    import numpy as np

    from openoutcry.baselines import (
        BASELINE_POLICIES,
        KellyVolTargetPolicy,
        MaxSharpePolicy,
        MinVariancePolicy,
        run_baselines,
        trailing_covariance,
        _project_simplex,
    )
    from openoutcry.regime_eval import (
        CHOP,
        REGIMES,
        TREND_DOWN,
        TREND_UP,
        evaluate_per_regime,
        label_regime,
        radar_score,
    )

    _HAVE_NUMPY = True
    _HAVE_BINDING = importlib.util.find_spec("openoutcry.openoutcry_py") is not None
except Exception:  # pragma: no cover - exercised only without numpy/binding
    _HAVE_NUMPY = False
    _HAVE_BINDING = False


requires_numpy = pytest.mark.skipif(not _HAVE_NUMPY, reason="numpy not importable")
requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native openoutcry binding not built"
)


def _feed(policy, closes_seq):
    """Drive ``policy`` over a sequence of close vectors, returning the actions."""
    return [policy({"closes": np.asarray(c, dtype=np.float64)}) for c in closes_seq]


# -- weight-vector validity -------------------------------------------------


@requires_numpy
def test_simplex_projection_is_valid():
    for v in (np.array([0.2, -0.5, 1.3]), np.array([5.0, 5.0, 5.0]), np.array([-1.0, -2.0])):
        w = _project_simplex(v)
        assert np.all(w >= -1e-9)
        assert np.isclose(w.sum(), 1.0)


@requires_numpy
@pytest.mark.parametrize("cls", [MinVariancePolicy, MaxSharpePolicy])
def test_meanvar_policies_are_long_only_simplex(cls):
    closes_seq = [[100.0, 50.0, 25.0]]
    for t in range(1, 30):
        closes_seq.append([c * (1.0 + 0.01 * np.sin(t + i)) for i, c in enumerate(closes_seq[-1])])
    actions = _feed(cls(), closes_seq)
    for a in actions:
        assert a.shape == (3,)
        assert np.all(np.isfinite(a))
        assert np.all(a >= -1e-6)
        assert np.isclose(a.sum(), 1.0, atol=1e-5)


@requires_numpy
def test_kelly_is_finite_and_gross_bounded():
    closes_seq = [[100.0, 50.0]]
    for t in range(1, 30):
        closes_seq.append([c * (1.0 + 0.02 * np.cos(t + i)) for i, c in enumerate(closes_seq[-1])])
    pol = KellyVolTargetPolicy(max_weight=1.0)
    for a in _feed(pol, closes_seq):
        assert a.shape == (2,)
        assert np.all(np.isfinite(a))
        assert np.all(np.abs(a) <= 1.0 + 1e-6)
        assert np.abs(a).sum() <= 1.0 + 1e-6


@requires_numpy
def test_meanvar_warmup_is_equal_weight():
    pol = MinVariancePolicy(min_history=3)
    first = pol({"closes": np.array([10.0, 20.0])})
    assert np.allclose(first, 0.5)


# -- causal covariance ------------------------------------------------------


@requires_numpy
def test_covariance_is_causal():
    rng = np.random.default_rng(0)
    closes_seq = [list(100.0 + rng.standard_normal(3).cumsum()) for _ in range(25)]
    pol = MinVariancePolicy(lookback=60, min_history=3)
    snapshots = []
    for t, c in enumerate(closes_seq):
        pol({"closes": np.asarray(c)})
        if pol.last_cov is not None:
            # The estimate at step t must match the covariance of exactly the closes
            # observed up to t — never any future close.
            prefix = np.asarray(closes_seq[: t + 1], dtype=np.float64)
            assert np.allclose(pol.last_cov, trailing_covariance(prefix, 60))
            snapshots.append(pol.last_cov.copy())
    # A past estimate is unaffected by closes that arrive afterward.
    assert len(snapshots) >= 2
    assert not np.allclose(snapshots[0], snapshots[-1])


# -- regime labeling --------------------------------------------------------


@requires_numpy
def test_label_regime_expected_labels():
    up = list(np.linspace(100.0, 130.0, 40))
    down = list(np.linspace(130.0, 100.0, 40))
    flat = [100.0 + (1.0 if i % 2 else -1.0) for i in range(40)]
    assert label_regime(up, window=20, trend_frac=0.5) == TREND_UP
    assert label_regime(down, window=20, trend_frac=0.5) == TREND_DOWN
    assert label_regime(flat, window=20, trend_frac=0.5) == CHOP
    # Warmup before a full window is chop.
    assert label_regime([100.0, 101.0], window=20, trend_frac=0.5) == CHOP


# -- radar score ------------------------------------------------------------


@requires_numpy
def test_radar_score_bounded_and_anchored():
    zero = {"deflated_sharpe": 0.0, "max_drawdown": 0.30}
    base = {"deflated_sharpe": 0.5, "max_drawdown": 0.10}
    flat = radar_score(zero, zero_anchor=zero, base_anchor=base, base=50.0, scale=100.0)
    based = radar_score(base, zero_anchor=zero, base_anchor=base, base=50.0, scale=100.0)
    assert flat["profitability"] == pytest.approx(0.0, abs=1e-6)
    assert flat["risk_control"] == pytest.approx(0.0, abs=1e-6)
    assert based["profitability"] == pytest.approx(50.0, abs=1e-6)
    assert based["risk_control"] == pytest.approx(50.0, abs=1e-6)
    # An all-around stronger panel stays bounded and beats the base anchor.
    strong = {"deflated_sharpe": 2.0, "max_drawdown": 0.02}
    sr = radar_score(strong, zero_anchor=zero, base_anchor=base)
    for axis in ("profitability", "risk_control", "overall"):
        assert 0.0 <= sr[axis] <= 100.0
    assert sr["profitability"] > 50.0


@requires_numpy
def test_radar_degenerate_anchor_does_not_crash():
    same = {"deflated_sharpe": 0.1, "max_drawdown": 0.1}
    out = radar_score(same, zero_anchor=same, base_anchor=same)
    for axis in ("profitability", "risk_control", "overall"):
        assert 0.0 <= out[axis] <= 100.0


# -- end-to-end (needs the native binding) ----------------------------------


@requires_binding
def test_run_baselines_includes_new_policies():
    from openoutcry.baselines import run_baselines as run

    rows = run(n_symbols=3, n_days=40, seeds=range(3))
    names = {r["policy"] for r in rows}
    assert {"min_variance", "max_sharpe", "kelly_vol_target"} <= names
    assert len(rows) == len(BASELINE_POLICIES)
    for r in rows:
        assert np.isfinite(r["deflated_sharpe"])
        assert 0.0 <= r["passed_k_rate"] <= 1.0
        assert np.isfinite(r["mean_return"])


def _make_env_for_seed(seed):
    from openoutcry.gym import OpenOutcryEnv

    return OpenOutcryEnv(n_symbols=3, n_days=40, seed=seed)


@requires_binding
def test_evaluate_per_regime_structure_and_determinism():
    pol = MinVariancePolicy()

    def fresh(_seed):
        return _make_env_for_seed(_seed)

    a = evaluate_per_regime(fresh, [0, 1], lambda obs: MinVariancePolicy()(obs), window=10)
    b = evaluate_per_regime(fresh, [0, 1], lambda obs: MinVariancePolicy()(obs), window=10)
    assert set(a["per_regime"]) == set(REGIMES)
    total = sum(a["per_regime"][r]["n_bars"] for r in REGIMES)
    assert total == a["overall"]["n_bars"] > 0
    for r in REGIMES:
        assert np.isfinite(a["per_regime"][r]["deflated_sharpe"])
    assert a == b


@requires_binding
def test_evaluate_per_regime_radar_pipeline():
    from openoutcry.baselines import EqualWeightLongPolicy, FlatPolicy
    from openoutcry.generalization import evaluate_seeds

    seeds = [0, 1]
    flat = evaluate_seeds(_make_env_for_seed, seeds, FlatPolicy())
    base = evaluate_seeds(_make_env_for_seed, seeds, EqualWeightLongPolicy())
    cand = evaluate_seeds(_make_env_for_seed, seeds, MaxSharpePolicy())
    # evaluate_seeds reports deflated_sharpe but not max_drawdown; radar tolerates the
    # missing risk key by falling back to 0 drawdown for every panel.
    sr = radar_score(cand, zero_anchor=flat, base_anchor=base)
    for axis in ("profitability", "risk_control", "overall"):
        assert 0.0 <= sr[axis] <= 100.0
