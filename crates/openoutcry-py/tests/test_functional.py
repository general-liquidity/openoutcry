"""Tests for the stateless functional (replay-model) view of OpenOutcry.

The contract under test: ``OpenOutcryFuncEnv`` is a genuinely stateless FuncEnv whose
state is ``(seed, actions)``. ``observation``/``reward``/``terminal`` must agree with
driving a stateful ``OpenOutcryEnv`` through the same action sequence, transitions must
not mutate their input, and the state must be picklable so tree-search nodes can key on
it.

If the native binding or gymnasium isn't importable, the live tests skip and only the
import / pure-tuple logic is exercised.
"""

import pickle

import numpy as np
import pytest

# functional.py is import-safe even without gymnasium's FuncEnv (it shims). The binding
# is only needed for the live replay tests.
from openoutcry.functional import (
    OpenOutcryFuncEnv,
    FUNC_ENV_SOURCE,
    replay,
)

try:
    from openoutcry import OpenOutcryEnv

    _HAVE_BINDING = True
except Exception:  # noqa: BLE001
    _HAVE_BINDING = False

_PARAMS = OpenOutcryFuncEnv.default_params(n_symbols=3, n_days=40, distribution_mode="calm")


def _equal_weight(n: int) -> list[float]:
    return [1.0 / n] * n


def test_func_env_source_resolved():
    # Records which base we bound to; informative for the handoff, never empty.
    assert FUNC_ENV_SOURCE in (
        "gymnasium.functional",
        "gymnasium.experimental.functional",
        "builtin-shim",
    )


def test_initial_and_transition_are_pure_tuples():
    fe = OpenOutcryFuncEnv()
    s0 = fe.initial(rng=7, params=_PARAMS)
    assert s0 == (7, tuple())

    a = [0.2, 0.2, 0.2]
    s1 = fe.transition(s0, a, rng=None, params=_PARAMS)
    # s0 unchanged (no hidden mutation), s1 extends it by one decision.
    assert s0 == (7, tuple())
    assert s1[0] == 7
    assert len(s1[1]) == 1
    assert s1[1][0] == (0.2, 0.2, 0.2)


def test_transition_twice_from_same_state_is_identical():
    fe = OpenOutcryFuncEnv()
    s0 = fe.initial(rng=3, params=_PARAMS)
    a = [0.1, 0.3, -0.2]
    s1a = fe.transition(s0, a)
    s1b = fe.transition(s0, a)
    assert s1a == s1b
    # Two independent branches from s1a don't interfere.
    left = fe.transition(s1a, [0.5, 0.0, 0.0])
    right = fe.transition(s1a, [0.0, 0.5, 0.0])
    assert left != right
    assert left[1][:1] == right[1][:1] == s1a[1]


def test_state_is_picklable():
    fe = OpenOutcryFuncEnv()
    s0 = fe.initial(rng=11, params=_PARAMS)
    s2 = fe.transition(fe.transition(s0, [0.3, 0.3, 0.3]), [0.0, 0.0, 0.0])
    blob = pickle.dumps(s2)
    restored = pickle.loads(blob)
    assert restored == s2
    # Hashable -> usable as a tree-search node key.
    assert hash(restored) == hash(s2)


@pytest.mark.skipif(not _HAVE_BINDING, reason="native binding not built")
def test_observation_reward_match_stateful_env():
    """Replaying through the FuncEnv must equal driving OpenOutcryEnv directly."""
    fe = OpenOutcryFuncEnv()
    n = _PARAMS["n_symbols"]
    actions = [_equal_weight(n) for _ in range(5)]

    # Drive the stateful env directly, capturing per-step obs/reward/done.
    direct = OpenOutcryEnv(
        n_symbols=n, n_days=_PARAMS["n_days"], seed=21, distribution_mode="calm"
    )
    direct.reset(seed=21)
    direct_obs = []
    direct_rewards = []
    for a in actions:
        obs, reward, terminated, truncated, _info = direct.step(np.asarray(a))
        direct_obs.append(obs)
        direct_rewards.append(reward)
        if terminated or truncated:
            break

    # Build the same sequence functionally and compare each frontier.
    fe_state = fe.initial(rng=21, params=_PARAMS)
    for i, a in enumerate(actions[: len(direct_rewards)]):
        next_state = fe.transition(fe_state, a, params=_PARAMS)
        fe_obs = fe.observation(next_state, params=_PARAMS)
        fe_reward = fe.reward(fe_state, a, next_state, params=_PARAMS)

        for key in ("closes", "positions", "cash"):
            np.testing.assert_allclose(
                fe_obs[key], direct_obs[i][key], rtol=0, atol=0,
                err_msg=f"obs[{key}] diverged at step {i}",
            )
        assert fe_reward == direct_rewards[i], f"reward diverged at step {i}"
        fe_state = next_state


@pytest.mark.skipif(not _HAVE_BINDING, reason="native binding not built")
def test_observation_is_stateless_repeatable():
    """observation/reward are functions of state only — repeated calls match exactly."""
    fe = OpenOutcryFuncEnv()
    n = _PARAMS["n_symbols"]
    s = fe.initial(rng=5, params=_PARAMS)
    for _ in range(3):
        s = fe.transition(s, _equal_weight(n), params=_PARAMS)

    o1 = fe.observation(s, params=_PARAMS)
    o2 = fe.observation(s, params=_PARAMS)
    for key in o1:
        np.testing.assert_array_equal(o1[key], o2[key])
    assert replay(s, _PARAMS).reward == replay(s, _PARAMS).reward


@pytest.mark.skipif(not _HAVE_BINDING, reason="native binding not built")
def test_independent_seeds_do_not_interfere():
    fe = OpenOutcryFuncEnv()
    n = _PARAMS["n_symbols"]
    a = _equal_weight(n)
    s_a = fe.transition(fe.initial(rng=1, params=_PARAMS), a, params=_PARAMS)
    s_b = fe.transition(fe.initial(rng=2, params=_PARAMS), a, params=_PARAMS)
    # Same decision, different scenario seed -> (almost surely) different observation.
    o_a = fe.observation(s_a, params=_PARAMS)
    o_b = fe.observation(s_b, params=_PARAMS)
    assert not np.array_equal(o_a["closes"], o_b["closes"])


@pytest.mark.skipif(not _HAVE_BINDING, reason="native binding not built")
def test_terminal_true_at_end_of_window():
    """Terminal flips True once the decision sequence exhausts the engine window."""
    fe = OpenOutcryFuncEnv()
    n = _PARAMS["n_symbols"]
    a = _equal_weight(n)
    s = fe.initial(rng=9, params=_PARAMS)
    assert fe.terminal(s, params=_PARAMS) is False  # fresh reset, not done
    done = False
    for _ in range(_PARAMS["n_days"] + 5):
        s = fe.transition(s, a, params=_PARAMS)
        if fe.terminal(s, params=_PARAMS):
            done = True
            break
    assert done, "terminal must eventually become True over the window"


@pytest.mark.skipif(not _HAVE_BINDING, reason="native binding not built")
def test_state_info_carries_engine_diagnostics():
    fe = OpenOutcryFuncEnv()
    n = _PARAMS["n_symbols"]
    s = fe.transition(fe.initial(rng=4, params=_PARAMS), _equal_weight(n), params=_PARAMS)
    info = fe.state_info(s, params=_PARAMS)
    assert info["seed"] == 4
    assert info["n_steps"] == 1
    assert "engine_info" in info and "nav" in info["engine_info"]
