"""Conformance, causal-wrapper, and generalization tests for OpenOutcry (Stream C).

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest crates/openoutcry-py/tests/test_conformance.py -q
"""

import warnings

import numpy as np
import pytest

from gymnasium.utils.env_checker import check_env as gym_check_env

from openoutcry import OpenOutcryEnv
from openoutcry.check_env import check_env, check_determinism_across_constructors
from openoutcry.wrappers import (
    TimeLimit,
    CausalNormalizeObservation,
    CausalNormalizeReward,
    FrameStack,
    RecordEpisodeStatistics,
)
from openoutcry.generalization import (
    train_test_seeds,
    evaluate_seeds,
    generalization_gap,
)


def _make(seed: int) -> OpenOutcryEnv:
    return OpenOutcryEnv(n_symbols=3, n_days=50, seed=seed)


def _equal_weight(env) -> np.ndarray:
    n = env.action_space.shape[0]
    return np.full((n,), 1.0 / n, dtype=np.float32)


# -- conformance ------------------------------------------------------------

def test_check_env_passes():
    check_env(_make(0), make_env_for_seed=_make)


def test_check_determinism_across_constructors():
    check_determinism_across_constructors(lambda: _make(7))


def test_gymnasium_env_checker_passes():
    # Gymnasium's own checker asserts more than the home-grown one (per-step passive
    # checks, normalized-action bounds, that the reset seed is actually consumed). Run it
    # alongside ours. A missing-spec warning is expected and harmless.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gym_check_env(_make(0), skip_render_check=True)


def test_same_seed_identical_reset_obs():
    a, _ = _make(3).reset()
    b, _ = _make(3).reset()
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_different_seed_reset_obs_differ():
    a, _ = _make(3).reset()
    b, _ = _make(4).reset()
    assert not all(np.array_equal(a[k], b[k]) for k in a)


# -- wrappers preserve the 5-tuple -----------------------------------------

@pytest.mark.parametrize(
    "wrap",
    [
        lambda e: TimeLimit(e, 5),
        CausalNormalizeObservation,
        CausalNormalizeReward,
        lambda e: FrameStack(e, 3),
        RecordEpisodeStatistics,
    ],
)
def test_wrapper_preserves_5_tuple(wrap):
    env = wrap(_make(1))
    obs, info = env.reset()
    assert isinstance(info, dict)
    out = env.step(_equal_weight(env))
    assert len(out) == 5
    _o, reward, terminated, truncated, _info = out
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


def test_time_limit_truncates_at_cap():
    env = TimeLimit(_make(1), 3)
    env.reset()
    a = _equal_weight(env)
    flags = [env.step(a)[3] for _ in range(3)]
    assert flags[-1] is True
    assert flags[:2] == [False, False]


def test_frame_stack_shape():
    env = FrameStack(_make(1), 4)
    obs, _ = env.reset()
    assert obs["closes"].shape == (4, 3)
    obs2 = env.step(_equal_weight(env))[0]
    assert obs2["closes"].shape == (4, 3)


def test_record_episode_statistics_injects_episode():
    env = RecordEpisodeStatistics(_make(2))
    env.reset()
    a = _equal_weight(env)
    info = {}
    done = False
    while not done:
        _o, _r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
    assert "episode" in info
    ep = info["episode"]
    assert set(ep) >= {"r", "l", "t", "sharpe", "max_drawdown"}
    assert ep["l"] > 0


# -- causality: step-t stats never depend on bars > t -----------------------

def _normalized_closes_sequence(wrap_obs, seed: int, steps: int) -> list[np.ndarray]:
    env = CausalNormalizeObservation(_make(seed))
    obs, _ = env.reset()
    seq = [obs["closes"].copy()]
    a = _equal_weight(env)
    for _ in range(steps - 1):
        obs = env.step(a)[0]
        seq.append(obs["closes"].copy())
    return seq


def test_causal_normalize_obs_is_prefix_stable():
    # A causal normalizer's output at step t must be identical whether the rollout
    # stops at t or continues — i.e. no future bar reaches back into step t.
    short = _normalized_closes_sequence(CausalNormalizeObservation, 5, 6)
    long = _normalized_closes_sequence(CausalNormalizeObservation, 5, 12)
    assert len(long) >= len(short)
    for t, (s, l) in enumerate(zip(short, long)):
        assert np.array_equal(s, l), f"causality break at step {t}"
    # First frame has no history → emitted as zeros (no leak from any bar).
    assert np.array_equal(short[0], np.zeros_like(short[0]))


def test_causal_normalize_reward_prefix_stable():
    def scaled(steps: int) -> list[float]:
        env = CausalNormalizeReward(_make(5))
        env.reset()
        a = _equal_weight(env)
        out = []
        for _ in range(steps):
            r = env.step(a)[1]
            out.append(r)
        return out

    short, long = scaled(5), scaled(10)
    assert short == long[: len(short)]


# -- generalization ---------------------------------------------------------

def test_train_test_seeds_disjoint():
    train, test = train_test_seeds(8, 4, seed_start=0, gap=10_000)
    assert len(train) == 8 and len(test) == 4
    assert set(train).isdisjoint(test)
    assert min(test) >= max(train) + 10_000


def test_evaluate_seeds_keys():
    out = evaluate_seeds(_make, [0, 1], max_steps=16)
    assert set(out) >= {"deflated_sharpe", "passed_k_rate", "mean_return", "n_seeds"}


def test_generalization_gap_end_to_end():
    out = generalization_gap(_make, n_train=2, n_test=2, gap=10_000, max_steps=16)
    assert set(out) == {"train", "test", "gap_deflated_sharpe", "gap_mean_return"}
    assert set(out["train"]) >= {"deflated_sharpe", "passed_k_rate", "mean_return"}
    assert np.isfinite(out["gap_deflated_sharpe"])
    assert np.isfinite(out["gap_mean_return"])
