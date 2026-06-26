"""The PettingZoo multi-agent **competition** env — verified against the Farama suite.

The env-behavior tests skip when ``pettingzoo`` is not installed; the rest of the package
works without it. The optional-import test runs *only* when pettingzoo is absent and
asserts the import is clean (the module imports, construction raises a clear
``RuntimeError``).
"""

import importlib.util

import numpy as np
import pytest

# These always work — the module is import-safe without pettingzoo, and the binding is a
# hard dependency of the package. The env classes live in the module namespace regardless
# of whether pettingzoo is importable.
import openoutcry  # noqa: F401  (ensures the package imports without pettingzoo)
import openoutcry.pettingzoo_env as pz_env
from openoutcry.pettingzoo_env import MultiAgentOpenOutcryEnv, make_aec_env

_HAS_PETTINGZOO = importlib.util.find_spec("pettingzoo") is not None
needs_pz = pytest.mark.skipif(not _HAS_PETTINGZOO, reason="pettingzoo not installed")


def _make(n_agents=3, n_symbols=3, n_days=24, seed=0):
    return MultiAgentOpenOutcryEnv(
        n_agents=n_agents, n_symbols=n_symbols, n_days=n_days, seed=seed
    )


# ---------------------------------------------------------------------------
# Optional-import contract (runs when pettingzoo is NOT installed)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_HAS_PETTINGZOO, reason="pettingzoo is installed")
def test_import_works_and_construction_raises_without_pettingzoo():
    """`import openoutcry.pettingzoo_env` succeeds without pettingzoo; constructing the
    env (or the AEC view) raises a clear RuntimeError rather than an ImportError."""
    assert pz_env._HAS_PETTINGZOO is False
    with pytest.raises(RuntimeError, match="pettingzoo is not installed"):
        MultiAgentOpenOutcryEnv(n_agents=2)
    with pytest.raises(RuntimeError, match="pettingzoo is not installed"):
        make_aec_env(n_agents=2)


# ---------------------------------------------------------------------------
# Full env behavior (runs when pettingzoo IS installed)
# ---------------------------------------------------------------------------

@needs_pz
def test_construct_reset_and_step_shapes():
    env = _make()
    assert env.possible_agents == ["agent_0", "agent_1", "agent_2"]
    obs, infos = env.reset(seed=0)

    assert set(obs.keys()) == set(env.agents) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.agents)
    for agent in env.agents:
        assert env.observation_space(agent).contains(obs[agent])

    # a few random simultaneous steps with per-agent dict actions
    for _ in range(3):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for d in (obs, rewards, terms, truncs, infos):
            assert set(d.keys()) == set(actions.keys())
        for a in rewards:
            assert isinstance(rewards[a], float)
            assert isinstance(terms[a], bool) and isinstance(truncs[a], bool)
    env.close()


@needs_pz
def test_same_seed_agents_see_identical_obs_at_reset():
    """Every agent trades the SAME frozen scenario, so their reset observations match
    exactly (a leak-free identical price path — the batched-competition invariant)."""
    env = _make(n_agents=4, n_symbols=3, n_days=20, seed=42)
    obs, _ = env.reset(seed=42)
    ref = obs["agent_0"]
    for agent in env.agents[1:]:
        for key in ref:
            np.testing.assert_array_equal(obs[agent][key], ref[key])
    env.close()


@needs_pz
def test_episode_end_attaches_cross_agent_ranking():
    """At episode end the per-agent info carries the cross-agent deflated-Sharpe
    leaderboard, ranked and reproducible."""
    env = _make(n_agents=3, n_symbols=2, n_days=16, seed=1)
    env.reset(seed=1)
    last_infos = {}
    for _ in range(64):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        _, _, terms, truncs, last_infos = env.step(actions)
        if all(terms[a] or truncs[a] for a in terms):
            break

    assert last_infos
    any_agent = next(iter(last_infos))
    ranking = last_infos[any_agent]["ranking"]
    # the leaderboard is sorted by deflated_sharpe desc, ties broken on agent id
    expected = sorted(ranking, key=lambda r: (-r["deflated_sharpe"], r["agent"]))
    assert ranking == expected
    assert [r["rank"] for r in ranking] == list(range(len(ranking)))
    env.close()


@needs_pz
def test_parallel_api_test():
    from pettingzoo.test import parallel_api_test

    env = _make(n_agents=2, n_symbols=3, n_days=20, seed=0)
    parallel_api_test(env, num_cycles=30)


@needs_pz
def test_parallel_seed_test():
    """Same seed -> identical rollouts (Farama parallel seed test, if available)."""
    try:
        from pettingzoo.test import parallel_seed_test
    except ImportError:
        pytest.skip("parallel_seed_test unavailable in this pettingzoo version")
    parallel_seed_test(lambda: _make(n_agents=2, n_symbols=3, n_days=20, seed=7))


@needs_pz
def test_aec_view_constructs_and_steps():
    """The AEC view derived via parallel_to_aec is a usable PettingZoo AEC env."""
    aec = make_aec_env(n_agents=2, n_symbols=2, n_days=16, seed=0)
    aec.reset(seed=0)
    assert set(aec.possible_agents) == {"agent_0", "agent_1"}
    for agent in aec.agent_iter(max_iter=8):
        _, _, termination, truncation, _ = aec.last()
        action = None if (termination or truncation) else aec.action_space(agent).sample()
        aec.step(action)
    aec.close()
