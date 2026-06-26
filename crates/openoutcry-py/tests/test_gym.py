"""Conformance + determinism tests for the OpenOutcry Python distribution.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest -q
"""

import json
import math

import numpy as np
import pytest

import openoutcry
from openoutcry import OpenOutcryEnv, TradingEnv


def _equal_weight_action(env: OpenOutcryEnv) -> np.ndarray:
    n = env.action_space.shape[0]
    return np.full((n,), 1.0 / n, dtype=np.float32)


def test_reexports_present():
    assert hasattr(openoutcry, "TradingEnv")
    assert hasattr(openoutcry, "OpenOutcryEnv")


def test_native_binding_json_boundary():
    env = TradingEnv(n_symbols=3, n_days=40, seed=1)
    obs = json.loads(env.reset())
    assert "symbols" in obs and "cash" in obs and "portfolio" in obs
    assert len(obs["symbols"]) == 3

    decision = {
        "orders": [
            {"symbol": s["symbol"], "action": "buy", "target_weight": 0.33}
            for s in obs["symbols"]
        ],
        "reasoning": "test",
    }
    obs_json, reward, done, info_json = env.step(json.dumps(decision))
    assert isinstance(obs_json, str)
    assert math.isfinite(reward)
    assert isinstance(done, bool)
    info = json.loads(info_json)
    assert "nav" in info and "events" in info


def test_native_binding_rejects_bad_json():
    env = TradingEnv(n_symbols=2, n_days=20, seed=1)
    env.reset()
    with pytest.raises(Exception):
        env.step("{not valid json")


def test_reset_returns_observation():
    env = OpenOutcryEnv(n_symbols=4, n_days=60, seed=3)
    obs, info = env.reset()
    assert set(obs) == {"closes", "positions", "cash"}
    assert obs["closes"].shape == (4,)
    assert np.all(np.isfinite(obs["closes"]))
    assert env.observation_space.contains(obs)


def test_full_episode_finite_rewards_and_terminates():
    env = OpenOutcryEnv(n_symbols=4, n_days=60, seed=5)
    env.reset()
    action = _equal_weight_action(env)
    rewards = []
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        assert math.isfinite(reward)
        rewards.append(reward)
        done = terminated or truncated
        steps += 1
        assert steps <= 1000, "episode failed to terminate"
    assert steps > 0
    assert done is True


def test_determinism_same_seed_identical_rewards():
    def run(seed: int) -> list[float]:
        env = OpenOutcryEnv(n_symbols=4, n_days=80, seed=seed)
        env.reset()
        action = _equal_weight_action(env)
        out = []
        done = False
        while not done:
            _obs, reward, terminated, truncated, _info = env.step(action)
            out.append(reward)
            done = terminated or truncated
        return out

    a = run(11)
    b = run(11)
    assert a == b, "same seed must produce identical reward sequences"

    c = run(12)
    # A different seed should (almost surely) diverge somewhere.
    assert a != c


def test_reset_seed_selects_scenario():
    """``reset(seed=k)`` rebuilds a synthetic env on the new seed, so one env can be
    re-pointed at distinct, reproducible scenarios via the gymnasium seed arg."""
    env = OpenOutcryEnv(n_symbols=3, n_days=60, seed=0)
    action = _equal_weight_action(env)

    def rollout(seed: int) -> list[float]:
        env.reset(seed=seed)
        out = []
        done = False
        while not done:
            _obs, reward, terminated, truncated, _info = env.step(action)
            out.append(reward)
            done = terminated or truncated
        return out

    assert rollout(7) == rollout(7), "reset(seed) must be reproducible"
    assert rollout(7) != rollout(8), "distinct reset seeds must give distinct scenarios"
    _obs, info = env.reset(seed=99)
    assert info["scenario_seed"] == 99


def test_from_csv_classmethod():
    csv = (
        "date,symbol,close\n"
        "2025-01-01,AAA,10\n2025-01-01,BBB,20\n"
        "2025-01-02,AAA,11\n2025-01-02,BBB,19\n"
        "2025-01-03,AAA,12\n2025-01-03,BBB,21\n"
    )
    env = TradingEnv.from_csv(csv, seed=1)
    obs = json.loads(env.reset())
    syms = sorted(s["symbol"] for s in obs["symbols"])
    assert syms == ["AAA", "BBB"]


def test_verifiers_import_guarded():
    # Must import cleanly even though `verifiers` is absent.
    from openoutcry import verifiers_env

    assert hasattr(verifiers_env, "load_environment")
    # Reward functions are pure and runnable without verifiers installed.
    state = {"returns": [0.01, -0.02, 0.03, 0.01]}
    assert math.isfinite(verifiers_env.deflated_sharpe_reward(state=state))
    assert verifiers_env.pass_k_reward(state=state) in (0.0, 1.0)
    assert verifiers_env.process_check_reward(state={"events": []}) == 1.0
