"""Tests for the vectorized batched OpenOutcry surface.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest -q tests/test_vector.py

Covers the native ``VecTradingEnv`` SoA JSON boundary, B=1 parity with the scalar
``TradingEnv``, same-step auto-reset (the ``first`` flag), and the gymnasium-vector
wrapper shapes.
"""

import json
import math

import numpy as np
import pytest

from openoutcry.openoutcry_py import TradingEnv, VecTradingEnv
from openoutcry.vector import OpenOutcryVectorEnv


def _flat_decisions(observations, weight):
    decisions = []
    for obs in observations:
        decisions.append(
            {
                "orders": [
                    {
                        "symbol": s["symbol"],
                        "action": "buy" if weight > 0 else "hold",
                        "target_weight": weight,
                        "confidence": 0.5,
                    }
                    for s in obs["symbols"]
                ],
                "reasoning": "test",
            }
        )
    return decisions


def test_native_vec_soa_json_boundary():
    env = VecTradingEnv(seeds=[1, 2, 3], n_symbols=3, n_days=40)
    assert env.num_envs == 3
    assert env.scenario_seeds == [1, 2, 3]

    reset = json.loads(env.reset_batch())
    assert reset["n"] == 3
    assert len(reset["observations"]) == 3

    decisions = _flat_decisions(reset["observations"], 0.2)
    step = json.loads(env.step_batch(json.dumps(decisions)))
    for key in ("observations", "rewards", "terminated", "truncated", "first", "infos"):
        assert key in step and len(step[key]) == 3, key
    assert all(math.isfinite(r) for r in step["rewards"])
    assert all("nav" in i and "events" in i for i in step["infos"])


def test_b1_matches_scalar_engine():
    seed, n_symbols, n_days = 11, 4, 40
    scalar = TradingEnv(n_symbols=n_symbols, n_days=n_days, seed=seed)
    vec = VecTradingEnv(seeds=[seed], n_symbols=n_symbols, n_days=n_days)

    s_obs = json.loads(scalar.reset())
    v_reset = json.loads(vec.reset_batch())
    assert v_reset["observations"][0] == s_obs

    while True:
        dec = _flat_decisions([s_obs], 0.25)
        s_obs_json, s_reward, s_done, s_info_json = scalar.step(json.dumps(dec[0]))
        v_step = json.loads(vec.step_batch(json.dumps(dec)))

        s_info = json.loads(s_info_json)
        assert v_step["rewards"][0] == s_reward
        assert v_step["infos"][0]["nav"] == s_info["nav"]
        assert v_step["truncated"][0] == s_done

        if s_done:
            assert v_step["first"][0] is True
            break
        assert v_step["first"][0] is False
        s_obs = json.loads(s_obs_json)
        assert v_step["observations"][0] == s_obs


def test_auto_reset_keeps_batch_running():
    # Short windows so each lane exhausts quickly and must auto-reset in place.
    env = VecTradingEnv(seeds=[1, 2], n_symbols=3, n_days=25)
    obs = json.loads(env.reset_batch())["observations"]
    resets = [0, 0]
    for _ in range(120):
        decisions = _flat_decisions(obs, 0.1)
        step = json.loads(env.step_batch(json.dumps(decisions)))
        assert len(step["observations"]) == 2
        for lane, first in enumerate(step["first"]):
            if first:
                resets[lane] += 1
        obs = step["observations"]
    assert resets[0] > 1 and resets[1] > 1


def test_step_batch_rejects_wrong_decision_count():
    env = VecTradingEnv(seeds=[1, 2, 3], n_symbols=2, n_days=30)
    env.reset_batch()
    with pytest.raises(Exception):
        env.step_batch(json.dumps([{"orders": [], "reasoning": ""}]))


def test_vector_wrapper_reset_step_shapes():
    env = OpenOutcryVectorEnv(seeds=[1, 2, 3, 4], n_symbols=4, n_days=60)
    assert env.num_envs == 4
    obs, infos = env.reset()
    assert set(obs) == {"closes", "positions", "cash"}
    assert obs["closes"].shape == (4, 4)
    assert obs["cash"].shape == (4, 1)
    assert infos["first"].shape == (4,)
    assert infos["first"].all()

    actions = np.full((4, 4), 0.1, dtype=np.float32)
    obs, rewards, terminated, truncated, infos = env.step(actions)
    assert obs["closes"].shape == (4, 4)
    assert rewards.shape == (4,)
    assert terminated.shape == (4,) and truncated.shape == (4,)
    assert infos["nav"].shape == (4,)
    assert np.all(np.isfinite(rewards))


def test_vector_wrapper_num_envs_from_count():
    env = OpenOutcryVectorEnv(3, n_symbols=2, n_days=30)
    assert env.num_envs == 3
    assert env.scenario_seeds == [0, 1, 2]
    obs, _ = env.reset()
    assert obs["closes"].shape == (3, 2)
