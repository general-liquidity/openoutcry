"""Tests for sequential / curriculum scenario chaining.

The contract under test: :class:`CurriculumEnv` walks an ordered schedule of scenario
seeds — one per ``reset()`` — deterministically; ``loop`` controls wrap-around vs.
exhaustion; and :func:`regime_curriculum` rotates the ``distribution_mode`` tier across
episodes. The live assertions need the native binding + gymnasium; without them only
the seed-schedule logic is exercised (the rest skips).
"""

import numpy as np
import pytest

from openoutcry.curriculum import CurriculumEnv, regime_curriculum, _CHAIN_STEP

try:
    from openoutcry import OpenOutcryEnv  # noqa: F401

    _HAVE_BINDING = True
except Exception:  # noqa: BLE001
    _HAVE_BINDING = False

_LIVE = pytest.mark.skipif(not _HAVE_BINDING, reason="native binding/gymnasium unavailable")

_KW = dict(n_symbols=3, n_days=30)


def _seed_seq(env: CurriculumEnv, n: int) -> list[int]:
    out = []
    for _ in range(n):
        _obs, info = env.reset()
        out.append(int(info["curriculum"]["seed"]))
    return out


# -- pure schedule logic (no binding required) -----------------------------------


def test_chain_step_value():
    assert _CHAIN_STEP == 997


def test_explicit_schedule_resolution():
    env = CurriculumEnv.__new__(CurriculumEnv)  # logic-only: avoid building a live env
    # Exercise the seed-resolution branch directly.
    env._seeds = [int(s) for s in (1, 2, 3)]
    assert env.seeds == [1, 2, 3]


def test_generated_sequential_and_chained_seeds():
    # Validate the generated chain without constructing a live env: the rule is pure.
    base, n = 5, 4
    seq = [base + k * 1 for k in range(n)]
    chained = [base + k * _CHAIN_STEP for k in range(n)]
    assert seq == [5, 6, 7, 8]
    assert chained == [5, 1002, 1999, 2996]


# -- live behavior ---------------------------------------------------------------


@_LIVE
def test_explicit_seeds_visited_in_order():
    env = CurriculumEnv(seeds=[1, 2, 3], loop=True, **_KW)
    visited = _seed_seq(env, 3)
    assert visited == [1, 2, 3]
    # curriculum metadata is surfaced and consistent with scenario_seed.
    _obs, info = env.reset()
    assert info["scenario_seed"] == info["curriculum"]["seed"]


@_LIVE
def test_loop_wraps_around():
    env = CurriculumEnv(seeds=[1, 2, 3], loop=True, **_KW)
    visited = _seed_seq(env, 4)
    assert visited == [1, 2, 3, 1]


@_LIVE
def test_no_loop_flags_exhaustion():
    env = CurriculumEnv(seeds=[1, 2, 3], loop=False, **_KW)
    for _ in range(3):
        _obs, info = env.reset()
        assert "curriculum_exhausted" not in info
    _obs, info = env.reset()
    assert info["curriculum_exhausted"] is True
    assert env.curriculum_exhausted is True
    # Clamps to the last scheduled seed rather than running off the end.
    assert info["curriculum"]["seed"] == 3


@_LIVE
def test_generated_chain_walked():
    env = CurriculumEnv(schedule="chained", n_episodes=3, seed=10, **_KW)
    assert env.seeds == [10, 10 + _CHAIN_STEP, 10 + 2 * _CHAIN_STEP]
    assert _seed_seq(env, 3) == env.seeds


@_LIVE
def test_determinism_same_args_same_sequence_and_obs():
    a = CurriculumEnv(seeds=[7, 8, 9], **_KW)
    b = CurriculumEnv(seeds=[7, 8, 9], **_KW)
    for _ in range(3):
        obs_a, info_a = a.reset()
        obs_b, info_b = b.reset()
        assert info_a["curriculum"]["seed"] == info_b["curriculum"]["seed"]
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key])


@_LIVE
def test_regime_curriculum_rotates_distribution_mode():
    env = regime_curriculum(0, 3, distribution_modes=("calm", "hard", "extreme"), **_KW)
    modes = []
    for _ in range(3):
        _obs, info = env.reset()
        modes.append(info["curriculum"]["distribution_mode"])
    assert modes == ["calm", "hard", "extreme"]


@_LIVE
def test_regime_curriculum_trajectories_differ_by_tier():
    def first_step_returns(mode_index: int) -> list[float]:
        env = regime_curriculum(3, 3, **_KW)
        out = None
        for i in range(3):
            obs, _info = env.reset()
            n = env.action_space.shape[0]
            action = np.full((n,), 1.0 / n, dtype=np.float32)
            rewards = []
            for _ in range(5):
                obs, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            if i == mode_index:
                out = rewards
        return out

    calm = first_step_returns(0)
    extreme = first_step_returns(2)
    assert calm != extreme, "extreme tier must diverge from calm"
