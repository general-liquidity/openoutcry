"""Tests for the gymnasium registration layer (``openoutcry.registration``).

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest -q tests/test_registration.py

Covers the ID matrix shape, idempotent (double) registration, the eval-band kwarg
contract, and end-to-end ``make`` / ``make_vec`` resolution to the scalar / vector envs.

The ``make`` / ``make_vec`` round-trips depend on the parallel CORE-ENV stream having
landed the ``distribution_mode`` constructor kwarg on both envs; until then they skip
(the registry-level assertions still gate the contract).
"""

import inspect

import pytest

gymnasium = pytest.importorskip("gymnasium")

from openoutcry import registration
from openoutcry.registration import register_envs, env_ids, EVAL_SEED_BASE


def _env_accepts_distribution_mode() -> bool:
    """True once the CORE-ENV stream has added the ``distribution_mode`` kwarg the
    registered IDs pass into the constructor."""
    try:
        from openoutcry.gym import OpenOutcryEnv
    except Exception:
        return False
    return "distribution_mode" in inspect.signature(OpenOutcryEnv.__init__).parameters


_needs_dist_mode = pytest.mark.skipif(
    not _env_accepts_distribution_mode(),
    reason="OpenOutcryEnv has no distribution_mode kwarg yet (parallel CORE-ENV stream)",
)


def test_env_ids_matrix():
    ids = env_ids()
    assert ids == [
        "OpenOutcry/Calm-v1",
        "OpenOutcry/Hard-v1",
        "OpenOutcry/Extreme-v1",
        "OpenOutcry/Calm-Eval-v1",
        "OpenOutcry/Hard-Eval-v1",
        "OpenOutcry/Extreme-Eval-v1",
    ]


def test_register_envs_populates_registry():
    register_envs()
    for env_id in env_ids():
        assert env_id in gymnasium.registry, env_id
        spec = gymnasium.registry[env_id]
        assert spec.entry_point == registration.ENTRY_POINT
        assert spec.vector_entry_point == registration.VECTOR_ENTRY_POINT
        assert spec.max_episode_steps == registration.MAX_EPISODE_STEPS
        assert spec.kwargs["n_symbols"] == registration.N_SYMBOLS
        assert spec.kwargs["n_days"] == registration.N_DAYS
        assert spec.kwargs["distribution_mode"] in registration.TIERS


def test_tier_kwargs_match_id():
    register_envs()
    for tier in registration.TIERS:
        spec = gymnasium.registry[f"OpenOutcry/{tier.capitalize()}-v1"]
        assert spec.kwargs["distribution_mode"] == tier
        assert "mode" not in spec.kwargs  # train band is the default


def test_eval_variants_select_disjoint_band():
    register_envs()
    assert EVAL_SEED_BASE == 1_000_000
    for tier in registration.TIERS:
        spec = gymnasium.registry[f"OpenOutcry/{tier.capitalize()}-Eval-v1"]
        # The eval variants carry the shared seed-band selector understood by both envs.
        assert spec.kwargs["mode"] == "eval"
        assert spec.kwargs["distribution_mode"] == tier


def test_double_registration_does_not_raise():
    first = register_envs()
    # Calling again (or after gymnasium auto-registers via the plugin) must be a no-op,
    # never an "id already registered" error.
    second = register_envs()
    assert first == second == env_ids()
    # And the registry holds exactly one spec per id (no duplicates / overwrites).
    for env_id in env_ids():
        assert env_id in gymnasium.registry


@_needs_dist_mode
def test_make_resolves_each_tier():
    register_envs()
    for env_id in env_ids():
        env = gymnasium.make(env_id)
        try:
            obs, info = env.reset(seed=0)
            assert set(obs) == {"closes", "positions", "cash"}
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            import math

            assert math.isfinite(float(reward))
        finally:
            env.close()


@_needs_dist_mode
def test_make_vec_routes_to_vector_env():
    register_envs()
    from openoutcry.vector import OpenOutcryVectorEnv

    env = gymnasium.make_vec("OpenOutcry/Hard-v1", num_envs=8)
    try:
        assert isinstance(env.unwrapped, OpenOutcryVectorEnv)
        assert env.num_envs == 8
        obs, infos = env.reset()
        assert obs["closes"].shape[0] == 8
    finally:
        env.close()
