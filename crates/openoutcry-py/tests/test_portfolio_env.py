"""Tests for :class:`openoutcry.portfolio_env.PortfolioEnv`.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_portfolio_env.py -q
"""

import math

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from openoutcry.gym import OpenOutcryEnv
from openoutcry.portfolio_env import PortfolioEnv


def _simplex_action(n_actions: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n_actions).astype(np.float32)


def test_constructs_and_spaces():
    env = PortfolioEnv(n_symbols=4, n_days=60, seed=0)
    assert env.n_assets == 4
    assert env.action_space.shape == (5,)  # n_symbols + cash
    assert float(env.action_space.low.min()) == 0.0  # long-only
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"closes", "positions", "cash"}
    assert obs["closes"].shape == (4,)


def test_step_normalizes_to_valid_simplex():
    env = PortfolioEnv(n_symbols=4, n_days=60, seed=1)
    env.reset(seed=1)
    action = np.array([2.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    w = info["weights"]
    assert w.shape == (5,)
    assert math.isclose(float(w.sum()), 1.0, abs_tol=1e-9)
    assert np.all(w >= 0.0)
    # 2:1:1:0:0 over a sum of 4 → cash 0.5, two assets 0.25, rest 0.
    np.testing.assert_allclose(w, [0.5, 0.25, 0.25, 0.0, 0.0], atol=1e-9)


def test_all_zero_action_is_all_cash():
    env = PortfolioEnv(n_symbols=4, n_days=60, seed=2)
    env.reset(seed=2)
    _, reward, _, _, info = env.step(np.zeros(5, dtype=np.float32))
    w = info["weights"]
    np.testing.assert_allclose(w, [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-9)
    # All-cash holds no position → exactly flat, log(1+0) = 0.
    assert math.isclose(info["simple_return"], 0.0, abs_tol=1e-12)
    assert math.isclose(reward, 0.0, abs_tol=1e-12)


def test_log_return_matches_underlying_cost_model():
    """Reward must equal log(1 + costed simple return) from the SAME underlying step —
    a raw OpenOutcryEnv fed the asset slice prices the move with fees/slippage."""
    action = np.array([1.0, 3.0, 2.0, 1.0, 1.0], dtype=np.float32)
    a = np.asarray(action, dtype=np.float64)
    weights = a / a.sum()
    asset_weights = weights[1:].astype(np.float32)

    penv = PortfolioEnv(n_symbols=4, n_days=80, seed=7)
    penv.reset(seed=7)
    _, reward, _, _, info = penv.step(action)

    raw = OpenOutcryEnv(n_symbols=4, n_days=80, seed=7, allow_short=False, max_weight=1.0)
    raw.reset(seed=7)
    _, r, _, _, _ = raw.step(asset_weights)

    assert math.isfinite(reward)
    assert math.isclose(info["simple_return"], float(r), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(reward, math.log1p(float(r)), rel_tol=0, abs_tol=1e-12)


def test_log_return_guarded_on_wipeout():
    assert math.isfinite(PortfolioEnv._log_return(-1.0))
    assert PortfolioEnv._log_return(-1.0) < -1.0
    assert math.isclose(PortfolioEnv._log_return(0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(PortfolioEnv._log_return(0.1), math.log1p(0.1), abs_tol=1e-12)


def test_determinism_same_seed_same_trajectory():
    rng = np.random.default_rng(123)
    actions = [_simplex_action(5, rng) for _ in range(8)]

    def rollout():
        env = PortfolioEnv(n_symbols=4, n_days=60, seed=5)
        env.reset(seed=5)
        out = []
        for act in actions:
            obs, reward, term, trunc, _ = env.step(act)
            out.append((float(reward), float(obs["cash"][0]), bool(term or trunc)))
            if term or trunc:
                break
        return out

    assert rollout() == rollout()


def test_return_last_action_appends_prior_weights():
    env = PortfolioEnv(n_symbols=4, n_days=60, seed=3, return_last_action=True)
    obs, _ = env.reset(seed=3)
    assert "last_action" in obs
    # First obs carries the initial all-cash weights.
    np.testing.assert_allclose(obs["last_action"], [1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-9)

    action = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    obs2, _, _, _, info = env.step(action)
    assert "last_action" in obs2
    # After the step the obs reports the weights just applied.
    np.testing.assert_allclose(obs2["last_action"], info["weights"], atol=1e-9)
    np.testing.assert_allclose(obs2["last_action"], [0.0, 0.25, 0.25, 0.25, 0.25], atol=1e-9)


def test_allow_short_l1_normalizes():
    env = PortfolioEnv(n_symbols=3, n_days=60, seed=4, allow_short=True)
    env.reset(seed=4)
    assert float(env.action_space.low.min()) == -1.0
    action = np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(action)
    w = info["weights"]
    assert math.isclose(float(np.abs(w).sum()), 1.0, abs_tol=1e-9)
