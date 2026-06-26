"""Tests for the frozen named held-out eval-seed regression set.

When the native binding and numpy are importable these roll the reference policy over
the committed seeds and assert determinism (the regression-gate property). The seed-set
invariants (disjointness from the train band, uniqueness) and the
:func:`assert_no_regression` gate logic are checked whenever the module imports.
"""

import importlib

import pytest

try:
    import numpy as np  # noqa: F401

    from openoutcry.dataset import EVAL_SEED_BASE
    from openoutcry.eval_seeds import (
        EVAL_SEEDS,
        EVAL_SET_VERSION,
        SCHEMA_VERSION,
        assert_no_regression,
        evaluate_eval_set,
    )

    _IMPORTED = True
    _HAVE_BINDING = importlib.util.find_spec("openoutcry.openoutcry_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding/numpy
    _IMPORTED = False
    _HAVE_BINDING = False


requires_import = pytest.mark.skipif(not _IMPORTED, reason="openoutcry not importable")
requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native openoutcry binding not built"
)


# -- committed-set invariants ----------------------------------------------


@requires_import
def test_seeds_are_in_the_held_out_band_and_unique():
    assert len(EVAL_SEEDS) >= 6
    values = list(EVAL_SEEDS.values())
    assert len(set(values)) == len(values), "eval seeds must be unique"
    for name, seed in EVAL_SEEDS.items():
        assert seed >= EVAL_SEED_BASE, f"{name}={seed} leaked into the train band"


@requires_import
def test_versions_are_pinned_strings():
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION
    assert isinstance(EVAL_SET_VERSION, str) and EVAL_SET_VERSION


# -- assert_no_regression gate ---------------------------------------------


@requires_import
def test_assert_no_regression_passes_on_identity():
    ref = {
        "held_out_00": {"deflated_sharpe": 1.5, "passed_k": True, "mean_return": 0.01},
        "held_out_01": {"deflated_sharpe": -0.2, "passed_k": False, "mean_return": -0.0},
    }
    assert assert_no_regression(ref, ref) is None


@requires_import
def test_assert_no_regression_flags_float_drift():
    ref = {"a": {"deflated_sharpe": 1.0, "passed_k": True, "mean_return": 0.01}}
    cur = {"a": {"deflated_sharpe": 1.0 + 1e-6, "passed_k": True, "mean_return": 0.01}}
    with pytest.raises(AssertionError):
        assert_no_regression(ref, cur)


@requires_import
def test_assert_no_regression_flags_passed_k_flip():
    ref = {"a": {"deflated_sharpe": 1.0, "passed_k": True, "mean_return": 0.01}}
    cur = {"a": {"deflated_sharpe": 1.0, "passed_k": False, "mean_return": 0.01}}
    with pytest.raises(AssertionError):
        assert_no_regression(ref, cur)


@requires_import
def test_assert_no_regression_flags_set_change():
    ref = {"a": {"deflated_sharpe": 1.0, "passed_k": True, "mean_return": 0.0}}
    cur = {
        "a": {"deflated_sharpe": 1.0, "passed_k": True, "mean_return": 0.0},
        "b": {"deflated_sharpe": 1.0, "passed_k": True, "mean_return": 0.0},
    }
    with pytest.raises(AssertionError):
        assert_no_regression(ref, cur)


# -- end-to-end evaluation against the real kernel --------------------------


@requires_binding
def test_evaluate_eval_set_shape_and_finite():
    result = evaluate_eval_set(n_symbols=3, n_days=40)
    assert set(result) == set(EVAL_SEEDS)
    for name, scores in result.items():
        assert set(scores) == {"deflated_sharpe", "passed_k", "mean_return"}
        assert np.isfinite(scores["deflated_sharpe"])
        assert np.isfinite(scores["mean_return"])
        assert isinstance(scores["passed_k"], bool)


@requires_binding
def test_evaluate_eval_set_is_byte_identical():
    a = evaluate_eval_set(n_symbols=3, n_days=40)
    b = evaluate_eval_set(n_symbols=3, n_days=40)
    assert a == b
    # The committed snapshot is its own zero-drift reference.
    assert assert_no_regression(a, b) is None


@requires_binding
def test_evaluate_eval_set_detects_a_perturbation():
    ref = evaluate_eval_set(n_symbols=3, n_days=40)
    perturbed = {k: dict(v) for k, v in ref.items()}
    first = next(iter(perturbed))
    perturbed[first]["deflated_sharpe"] += 1.0
    with pytest.raises(AssertionError):
        assert_no_regression(ref, perturbed)
