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
    # same_step mirrors the scalar reset()/step() pattern (reset in place on the
    # ending step), so the B=1 lane is byte-identical to the scalar engine.
    vec = VecTradingEnv(
        seeds=[seed], n_symbols=n_symbols, n_days=n_days, autoreset_mode="same_step"
    )

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


def test_hard_distribution_diverges_from_calm():
    def rollout(mode: str) -> list[float]:
        env = OpenOutcryVectorEnv(seeds=[5], n_symbols=4, n_days=60, distribution_mode=mode)
        env.reset()
        out = []
        for _ in range(40):
            actions = np.full((1, 4), 0.5, dtype=np.float32)
            _o, rewards, _t, _tr, _i = env.step(actions)
            out.append(float(rewards[0]))
        return out

    assert rollout("calm") != rollout("hard")


def test_next_step_defers_reset_to_following_step():
    env = OpenOutcryVectorEnv(seeds=[1], n_symbols=3, n_days=25, autoreset_mode="next_step")
    assert env._autoreset_mode == "next_step"
    env.reset()
    prev_ended = False
    saw_deferred_reset = False
    for _ in range(120):
        actions = np.zeros((1, 3), dtype=np.float32)
        _o, rewards, terminated, truncated, infos = env.step(actions)
        if prev_ended:
            # The step after an ending step is the deferred reset: reward 0, flags False.
            assert infos["first"][0]
            assert rewards[0] == 0.0
            assert not terminated[0] and not truncated[0]
            saw_deferred_reset = True
            break
        ended = bool(terminated[0] or truncated[0])
        if ended:
            assert not infos["first"][0], "next_step must not reset on the ending step"
        prev_ended = ended
    assert saw_deferred_reset


def test_same_step_surfaces_final_obs():
    env = OpenOutcryVectorEnv(seeds=[1], n_symbols=3, n_days=25, autoreset_mode="same_step")
    assert env._autoreset_mode == "same_step"
    env.reset()
    saw_reset = False
    for _ in range(120):
        actions = np.zeros((1, 3), dtype=np.float32)
        _o, _r, _t, _tr, infos = env.step(actions)
        if infos["first"][0]:
            assert "final_obs" in infos and "final_info" in infos
            assert infos["final_obs"][0] is not None
            assert infos["final_info"][0] is not None
            saw_reset = True
            break
    assert saw_reset


def test_disabled_never_resets():
    env = OpenOutcryVectorEnv(seeds=[1], n_symbols=3, n_days=25, autoreset_mode="disabled")
    env.reset()
    infos = {}
    for _ in range(80):
        actions = np.zeros((1, 3), dtype=np.float32)
        _o, _r, _t, _tr, infos = env.step(actions)
        assert not infos["first"][0], "disabled never flags first"
    assert "final_obs" not in infos


def test_unknown_autoreset_mode_rejected():
    with pytest.raises(Exception):
        OpenOutcryVectorEnv(seeds=[1], n_symbols=2, n_days=20, autoreset_mode="bogus")
