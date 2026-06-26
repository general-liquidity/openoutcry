"""Tests for the baseline policies + leaderboard runner.

When the native binding and numpy are importable these exercise a real (small)
baseline sweep against the SharpeBench kernel. When the binding is absent the file
still imports and the dependency-light logic (markdown rendering, policy shapes) is
checked directly.
"""

import importlib

import pytest

try:
    import numpy as np  # noqa: F401

    from openoutcry import baselines
    from openoutcry.baselines import (
        BASELINE_POLICIES,
        EqualWeightLongPolicy,
        FlatPolicy,
        MomentumPolicy,
        leaderboard_markdown,
        run_baselines,
    )

    _HAVE_BINDING = importlib.util.find_spec("openoutcry.openoutcry_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding/numpy
    _HAVE_BINDING = False


requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native openoutcry binding not built"
)


# -- logic that needs no native binding -------------------------------------


@requires_binding
def test_policies_produce_correct_shape():
    obs = {"closes": np.array([100.0, 101.0, 99.0])}
    for _name, factory in BASELINE_POLICIES:
        action = factory()(obs)
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))


@requires_binding
def test_flat_is_zero_and_equal_weight_sums_to_one():
    obs = {"closes": np.array([10.0, 20.0, 30.0, 40.0])}
    assert np.allclose(FlatPolicy()(obs), 0.0)
    ew = EqualWeightLongPolicy()(obs)
    assert np.isclose(ew.sum(), 1.0)
    assert np.all(ew > 0.0)


@requires_binding
def test_momentum_warms_up_then_signs():
    pol = MomentumPolicy()
    # First call has no prior close: warms up to equal weight.
    first = pol({"closes": np.array([100.0, 100.0])})
    assert np.allclose(first, 0.5)
    # Second call differences against the first: one up, one down.
    second = pol({"closes": np.array([101.0, 99.0])})
    assert second[0] > 0.0 and second[1] < 0.0


@requires_binding
def test_leaderboard_renders_sorted_by_deflated_sharpe():
    rows = [
        {"policy": "a", "deflated_sharpe": 0.1, "passed_k_rate": 0.5, "mean_return": 0.01},
        {"policy": "b", "deflated_sharpe": 0.9, "passed_k_rate": 0.8, "mean_return": 0.02},
        {"policy": "c", "deflated_sharpe": 0.4, "passed_k_rate": 0.2, "mean_return": -0.01},
    ]
    md = leaderboard_markdown(rows)
    assert "Deflated Sharpe" in md
    # Highest deflated Sharpe ranks first.
    body = md.splitlines()[2:]
    assert body[0].split("|")[2].strip() == "b"
    assert body[-1].split("|")[2].strip() == "a"


def test_leaderboard_markdown_importable_without_binding():
    # Pure rendering must not require the native kernel.
    from openoutcry.baselines import leaderboard_markdown as render

    md = render([{"policy": "x", "deflated_sharpe": 0.0, "passed_k_rate": 0.0, "mean_return": 0.0}])
    assert md.startswith("| Rank | Policy |")
    assert "| x |" in md


# -- end-to-end sweep against the real kernel -------------------------------


@requires_binding
def test_run_baselines_returns_scored_rows():
    rows = run_baselines(n_symbols=3, n_days=40, seeds=range(4))
    assert len(rows) == len(BASELINE_POLICIES) > 1
    names = {r["policy"] for r in rows}
    assert {"flat", "equal_weight_long", "momentum"} <= names
    for r in rows:
        assert set(r) == {"policy", "deflated_sharpe", "passed_k_rate", "mean_return"}
        assert np.isfinite(r["deflated_sharpe"])
        assert 0.0 <= r["passed_k_rate"] <= 1.0
        assert np.isfinite(r["mean_return"])


@requires_binding
def test_run_baselines_is_deterministic():
    a = run_baselines(n_symbols=3, n_days=40, seeds=range(4))
    b = run_baselines(n_symbols=3, n_days=40, seeds=range(4))
    assert a == b


@requires_binding
def test_run_baselines_distribution_mode_degrades_gracefully():
    # The distribution_mode kwarg is added by a sibling stream; the runner must not
    # crash on any tier whether or not the binding wires it yet.
    for mode in ("calm", "hard", "extreme"):
        rows = run_baselines(n_symbols=3, n_days=30, seeds=range(2), distribution_mode=mode)
        assert len(rows) == len(BASELINE_POLICIES)
