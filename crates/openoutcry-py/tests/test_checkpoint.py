"""Tests for replay-based clone/restore (:mod:`openoutcry.checkpoint`).

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_checkpoint.py -q

The live tests need the native binding (``OpenOutcryEnv`` runs a real engine); they skip
cleanly if it isn't importable. The pure-data round-trip below the skip guard exercises
``CheckpointState`` serialization without the binding.
"""

import pickle

import numpy as np
import pytest

# CheckpointState is pure data, but importing the module pulls in the engine binding (via
# .gym). Guard the whole live surface so the suite skips rather than errors when the native
# extension isn't built/current.
checkpoint = pytest.importorskip("openoutcry.checkpoint")
CheckpointableEnv = checkpoint.CheckpointableEnv
CheckpointState = checkpoint.CheckpointState

from openoutcry.gym import OpenOutcryEnv  # noqa: E402  (after importorskip)


def _equal_weight(env) -> np.ndarray:
    n = env.action_space.shape[0]
    return np.full((n,), 1.0 / n, dtype=np.float32)


def _obs_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _make(seed: int = 5) -> CheckpointableEnv:
    env = CheckpointableEnv(OpenOutcryEnv(n_symbols=3, n_days=60, seed=seed))
    env.reset()
    return env


def _roll(env, action, k: int):
    """Step ``action`` ``k`` times, returning the [(obs, reward)] trajectory."""
    traj = []
    for _ in range(k):
        obs, reward, terminated, truncated, _info = env.step(action)
        traj.append((obs, reward))
        if terminated or truncated:
            break
    return traj


# -- restore -----------------------------------------------------------------


def test_restore_returns_to_snapshot_point():
    """Roll a few steps, snapshot, roll further; restore -> the post-snapshot trajectory
    replays byte-identically."""
    env = _make()
    action = _equal_weight(env)

    _roll(env, action, 4)
    snap = env.clone_state()

    after = _roll(env, action, 6)  # the "ground truth" continuation from the snapshot
    assert after, "expected some post-snapshot steps"

    env.restore_state(snap)
    restored = _roll(env, action, 6)

    assert len(restored) == len(after)
    for (o1, r1), (o2, r2) in zip(after, restored):
        assert r1 == r2, "rewards must replay byte-identically"
        assert _obs_equal(o1, o2), "observations must replay byte-identically"


def test_restore_resets_recorded_prefix():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 3)
    snap = env.clone_state()
    assert snap.step == 3
    _roll(env, action, 5)

    env.restore_state(snap)
    again = env.clone_state()
    assert again.step == 3
    assert len(again.actions) == 3


# -- branch ------------------------------------------------------------------


def test_branch_is_independent_of_original():
    """Stepping a branch must not perturb the parent env."""
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 4)
    snap = env.clone_state()

    branch = env.branch(snap)
    # Drive the branch far forward.
    _roll(branch, action, 8)

    # The parent, restored to the same snapshot, still replays the original continuation.
    parent_after = _roll(env.branch(snap), action, 5)
    env.restore_state(snap)
    parent_restored = _roll(env, action, 5)
    for (o1, r1), (o2, r2) in zip(parent_after, parent_restored):
        assert r1 == r2 and _obs_equal(o1, o2)


def test_two_branches_same_actions_identical():
    """Determinism: two independent branches from one snapshot, fed identical actions,
    produce identical trajectories."""
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 5)
    snap = env.clone_state()

    b1 = env.branch(snap)
    b2 = env.branch(snap)
    t1 = _roll(b1, action, 7)
    t2 = _roll(b2, action, 7)

    assert len(t1) == len(t2) and t1
    for (o1, r1), (o2, r2) in zip(t1, t2):
        assert r1 == r2 and _obs_equal(o1, o2)


def test_branch_does_not_share_action_list():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 3)
    snap = env.clone_state()
    branch = env.branch(snap)
    _roll(branch, action, 4)
    # Parent's recorded prefix is untouched by branch stepping.
    assert env.clone_state().step == 3
    assert branch.clone_state().step == 3 + 4


# -- CheckpointState serialization ------------------------------------------


def test_state_roundtrips_through_dict():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 4)
    snap = env.clone_state()

    restored = CheckpointState.from_dict(snap.to_dict())
    assert restored.params == snap.params
    assert restored.step == snap.step
    assert restored.include_rng == snap.include_rng
    assert np.allclose(np.array(restored.actions), np.array(snap.actions))


def test_state_is_picklable():
    env = _make()
    action = _equal_weight(env)
    _roll(env, action, 4)
    snap = env.clone_state()

    blob = pickle.dumps(snap)
    back = pickle.loads(blob)
    assert isinstance(back, CheckpointState)
    assert back.to_dict() == snap.to_dict()

    # A pickled state restores a fresh env exactly.
    fresh = env.branch(back)
    a, b = _roll(fresh, action, 3), _roll(env.branch(snap), action, 3)
    for (o1, r1), (o2, r2) in zip(a, b):
        assert r1 == r2 and _obs_equal(o1, o2)


def test_state_carries_no_env_handle():
    """Leak-safety: the captured params must not contain a dataset / env handle."""
    env = _make()
    snap = env.clone_state()
    for value in snap.params.values():
        assert not (
            callable(getattr(value, "reset", None))
            and callable(getattr(value, "step", None))
        ), "checkpoint params must not embed a live env/dataset handle"


def test_native_o1_checkpoint_matches_replay():
    """The native O(1) snapshot (native=True) restores byte-identically and agrees with
    the replay path."""
    import numpy as np
    from openoutcry import OpenOutcryEnv, CheckpointableEnv

    def _act(env):
        n = env.action_space.shape[0]
        return np.full((n,), 0.2, dtype=np.float32)

    env = CheckpointableEnv(OpenOutcryEnv(n_symbols=3, n_days=60, seed=9))
    env.reset(seed=9)
    a = _act(env)
    for _ in range(8):
        env.step(a)
    snap_native = env.clone_state(native=True)
    snap_replay = env.clone_state(native=False)
    # advance, then restore via the native O(1) path
    after = [tuple(env.step(a)[0]["closes"]) for _ in range(4)]
    env.restore_state(snap_native)
    native_after = [tuple(env.step(a)[0]["closes"]) for _ in range(4)]
    # and via replay, on an independent branch
    branch = env.branch(snap_replay)
    replay_after = [tuple(branch.step(a)[0]["closes"]) for _ in range(4)]
    assert native_after == after
    assert native_after == replay_after
    # the native snapshot round-trips through to_dict/from_dict
    from openoutcry import CheckpointState
    rt = CheckpointState.from_dict(snap_native.to_dict())
    assert rt.native_state == snap_native.native_state
