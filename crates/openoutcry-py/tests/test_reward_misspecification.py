"""Tests for the reward-misspecification negative-control track.

When the native binding and numpy are importable these exercise a real (small) proxy sweep
against the SharpeBench kernel and the falsifiable punishment demonstration. The module
imports ``score_run`` at top, so the whole file is binding-gated like ``test_baselines``.
"""

import importlib

import pytest

try:
    import numpy as np

    from openoutcry.reward_misspecification import (
        MISSPECIFIED_PROXY_POLICIES,
        MISSPECIFIED_REWARDS,
        demonstrate_punishment,
        indicator_shaped,
        misspecification_gap,
        raw_pnl_unpenalized,
        recency_biased,
        win_rate,
    )
    from openoutcry.gym import OpenOutcryEnv

    _HAVE_BINDING = importlib.util.find_spec("openoutcry.openoutcry_py") is not None
except Exception:  # pragma: no cover - exercised only without the binding/numpy
    _HAVE_BINDING = False


requires_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native openoutcry binding not built"
)


def _make_env(seed: int):
    return OpenOutcryEnv(n_symbols=3, n_days=60, seed=seed)


def _hand_state() -> dict:
    return {
        "returns": [0.01, -0.02, 0.03, -0.005, 0.012, -0.001, 0.004],
        "events": [
            {"event": "target_weights", "weights": [0.3, 0.3, 0.3]} for _ in range(7)
        ],
    }


# -- rewards: bounded + computable from state -------------------------------


@requires_binding
def test_misspecified_rewards_are_bounded_and_computable():
    state = _hand_state()
    for name, fn in MISSPECIFIED_REWARDS.items():
        r = fn(state=state)
        assert isinstance(r, float)
        assert np.isfinite(r)
        assert -1.0 <= r <= 1.0, name
    # win_rate / indicator_shaped live in [0, 1].
    assert 0.0 <= win_rate(state=state) <= 1.0
    assert 0.0 <= indicator_shaped(state=state) <= 1.0


@requires_binding
def test_rewards_handle_empty_state():
    for fn in MISSPECIFIED_REWARDS.values():
        assert fn(state={}) == 0.0
        assert fn(state=None) == 0.0


@requires_binding
def test_raw_pnl_rewards_gross_return_unpenalized():
    # No risk/cost term: a bigger summed return strictly scores higher.
    low = raw_pnl_unpenalized(state={"returns": [0.001, 0.001]})
    high = raw_pnl_unpenalized(state={"returns": [0.05, 0.05]})
    assert high > low


@requires_binding
def test_win_rate_ignores_magnitude():
    # Many tiny wins + one huge loss still scores high (the blow-up reward).
    blowup = win_rate(state={"returns": [0.001, 0.001, 0.001, -0.5]})
    assert blowup == pytest.approx(0.75)


# -- proxy policies produce valid actions -----------------------------------


@requires_binding
def test_proxy_policies_produce_valid_actions():
    obs = {"closes": np.array([100.0, 101.0, 99.0])}
    for factory in MISSPECIFIED_PROXY_POLICIES.values():
        policy = factory()
        first = policy(obs)
        second = policy({"closes": np.array([101.0, 100.0, 100.0])})
        for action in (first, second):
            assert action.shape == (3,)
            assert np.all(np.isfinite(action))
            assert np.all(np.abs(action) <= 1.0 + 1e-6)


# -- the falsifiable wedge --------------------------------------------------


@requires_binding
def test_demonstrate_punishment_scorer_punishes_flawed_proxies():
    table = demonstrate_punishment(_make_env, range(6), max_steps=128)
    assert set(table) == set(MISSPECIFIED_REWARDS)
    for row in table.values():
        assert set(row) == {"deflated_sharpe", "passed_k", "mean_return"}
        assert np.isfinite(row["deflated_sharpe"])
        assert 0.0 <= row["passed_k"] <= 1.0

    # No flawed proxy wins on the real metric: deflated Sharpe stays ~0.
    assert max(r["deflated_sharpe"] for r in table.values()) <= 0.5

    # The wedge: the proxy with the best raw mean return looks profitable in-sample yet
    # earns ~0 deflated Sharpe — naive reward -> high raw return, scorer does not reward it.
    best = max(table.values(), key=lambda r: r["mean_return"])
    assert best["mean_return"] > 0.0
    assert best["deflated_sharpe"] <= 0.5


@requires_binding
def test_misspecification_gap_reports_clean_vs_flawed():
    g = misspecification_gap(_make_env, range(6), flawed_reward="raw_pnl_unpenalized", max_steps=128)
    assert g["clean_reward"] == "differential_sharpe"
    assert g["flawed_reward"] == "raw_pnl_unpenalized"
    assert g["proxy_is_stand_in"] is True
    assert np.isfinite(g["gap_deflated_sharpe"])
    assert np.isfinite(g["gap_mean_return"])
    for side in ("clean", "flawed"):
        assert set(g[side]) == {"deflated_sharpe", "passed_k", "mean_return"}


@requires_binding
def test_misspecification_gap_rejects_unknown_reward():
    with pytest.raises(ValueError):
        misspecification_gap(_make_env, range(2), flawed_reward="not_a_reward")


# -- the critical invariant: never registered into production ----------------


@requires_binding
def test_negative_controls_are_not_in_production_registry():
    from openoutcry import rewards as production_rewards

    for name in (
        "raw_pnl_unpenalized",
        "win_rate",
        "indicator_shaped",
        "recency_biased",
    ):
        assert name not in production_rewards.REWARD_SCHEMES
