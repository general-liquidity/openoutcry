"""A stateless, functional-RL view of OpenOutcry via the *replay model*.

Gymnasium's :class:`~gymnasium.experimental.functional.FuncEnv` describes a *pure*
environment as five free functions over an explicit ``state`` value::

    initial(rng, params)                       -> state
    transition(state, action, rng, params)     -> state
    observation(state, rng, params)            -> obs
    reward(state, action, next_state, rng, p)  -> float
    terminal(state, rng, params)               -> bool

OpenOutcry's native engine is a deterministic, point-in-time transition: given a seed
and an ordered sequence of decisions, the produced observations / rewards / done-flags
are byte-identical on replay. That property lets us model the FuncEnv ``state`` as a
**plain immutable tuple** ``(seed, actions)`` where ``actions`` is the tuple of decisions
taken so far. There is *no* hidden mutable engine state living in the FuncEnv:

* ``initial``    -> ``(seed, ())``
* ``transition`` -> ``(seed, actions + (action,))`` (append-only, returns a new tuple)
* ``observation`` / ``reward`` / ``terminal`` -> build a fresh :class:`OpenOutcryEnv`
  at ``seed``, replay ``actions`` through it, and decode the result.

Because the state is just ``seed + decisions`` (no engine handle, no numpy buffers), it
is picklable and usable as a tree-search node key, and the *only* thing that can affect a
rollout is the seed and the decision path — a strictly stronger leak guarantee than the
stateful :class:`OpenOutcryEnv` offers.

Honest JAX limitation
---------------------
``transition`` / ``observation`` / ``reward`` / ``terminal`` call into the native
(Rust/pyo3) binding through :class:`OpenOutcryEnv`. They are therefore **NOT**
``jax.jit`` / ``jax.vmap`` traceable: there is no JAX acceleration here and you cannot
wrap this in :class:`gymnasium.experimental.functional_jax_env.FunctionalJaxEnv`
(which requires JAX-array-pure functions). A JAX-native engine is a separate, future
build. What this module *does* deliver is the genuine stateless FuncEnv **contract**,
the stronger leak guarantee, and compatibility with functional-RL tooling and tree
search that key on an immutable, picklable state.

The replay cost is O(len(actions)) per ``observation`` / ``reward`` call (a full rebuild
+ replay). That is acceptable for tree search and functional rollouts of modest depth;
for long throughput-bound rollouts use the stateful :class:`OpenOutcryEnv` directly.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .gym import OpenOutcryEnv

# Locate the FuncEnv base across gymnasium layouts. 1.2.x ships it under
# ``gymnasium.experimental.functional``; earlier/later cuts exposed (or may expose) it
# at ``gymnasium.functional``. Support whichever imports and degrade gracefully to a
# minimal protocol-compatible shim so this module always imports.
try:  # pragma: no cover - import-path probing
    from gymnasium.functional import FuncEnv as _FuncEnv  # type: ignore

    _FUNC_ENV_SOURCE = "gymnasium.functional"
except Exception:  # noqa: BLE001 - any import failure falls through to the next probe
    try:  # pragma: no cover
        from gymnasium.experimental.functional import FuncEnv as _FuncEnv  # type: ignore

        _FUNC_ENV_SOURCE = "gymnasium.experimental.functional"
    except Exception:  # noqa: BLE001

        class _FuncEnv:  # type: ignore[no-redef]
            """Minimal protocol-compatible stand-in when gymnasium ships no FuncEnv.

            Implements only the constructor contract (``__init__`` stores metadata)
            so :class:`OpenOutcryFuncEnv` can subclass and run even on a gymnasium
            build that lacks the functional API.
            """

            def __init__(self, options: Optional[dict] = None) -> None:
                self.__dict__.update(options or {})

        _FUNC_ENV_SOURCE = "builtin-shim"


#: Which base class this module bound to, for diagnostics / handoff reporting.
FUNC_ENV_SOURCE = _FUNC_ENV_SOURCE

# A FuncEnv state: the scenario seed plus the ordered tuple of decisions taken so far.
# Each action is normalized to a tuple of floats (target weights) so the whole state is
# immutable, picklable, and hashable — a valid tree-search node key.
State = Tuple[int, Tuple[Tuple[float, ...], ...]]


def _default_params() -> dict[str, Any]:
    """The replay parameters that pin a scenario. Mirrors :class:`OpenOutcryEnv`'s
    synthetic constructor knobs; ``seed`` is carried in the *state*, not here, so two
    states under the same params differ only by seed + decisions."""
    return {
        "n_symbols": 4,
        "n_days": 120,
        "distribution_mode": "calm",
        "max_weight": 1.0,
        "allow_short": True,
        "mode": "train",
        "window_start": None,
        "window_end": None,
        "csv_text": None,
        "env_kwargs": None,
    }


def _normalize_action(action: Any) -> Tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(action, dtype=np.float64).reshape(-1))


def _build_env(seed: int, params: dict[str, Any]) -> OpenOutcryEnv:
    p = {**_default_params(), **(params or {})}
    return OpenOutcryEnv(
        n_symbols=p["n_symbols"],
        n_days=p["n_days"],
        seed=int(seed),
        window_start=p["window_start"],
        window_end=p["window_end"],
        csv_text=p["csv_text"],
        max_weight=p["max_weight"],
        allow_short=p["allow_short"],
        distribution_mode=p["distribution_mode"],
        mode=p["mode"],
        env_kwargs=p["env_kwargs"],
    )


class _Replay:
    """Decoded result of replaying a state's action sequence through a fresh env."""

    __slots__ = ("obs", "reward", "terminated", "truncated", "info", "n_steps")

    def __init__(
        self,
        obs: dict[str, np.ndarray],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        n_steps: int,
    ) -> None:
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.n_steps = n_steps

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def replay(state: State, params: Optional[dict] = None) -> _Replay:
    """Build a fresh :class:`OpenOutcryEnv` at ``state``'s seed and drive it through the
    state's decision sequence. Returns the final decoded observation, the *last* reward,
    and the done flags. With no actions, returns the post-``reset`` observation and a
    reward of ``0.0`` (no step has scored yet).

    This is the single point where the native binding is touched; everything else in the
    FuncEnv is a pure tuple manipulation.
    """
    seed, actions = state
    env = _build_env(seed, params or {})
    obs, _info = env.reset()
    last_reward = 0.0
    terminated = False
    truncated = False
    info: dict = dict(_info)
    for action in actions:
        if terminated or truncated:
            # Replaying past the absorbing/exhausted point is a no-op: the engine has
            # no further bars. We stop scoring but keep the last decoded frame.
            break
        arr = np.asarray(action, dtype=np.float64)
        obs, last_reward, terminated, truncated, info = env.step(arr)
    return _Replay(
        obs=obs,
        reward=float(last_reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info,
        n_steps=len(actions),
    )


class OpenOutcryFuncEnv(_FuncEnv):
    """Stateless functional view of OpenOutcry over the ``(seed, actions)`` replay state.

    See the module docstring for the design and the (honest) JAX limitation: the
    transition/observation functions call the native binding and are **not**
    JAX-traceable.

    Usage::

        fe = OpenOutcryFuncEnv()
        params = fe.default_params(n_symbols=3, n_days=40)
        s0 = fe.initial(rng=7, params=params)
        s1 = fe.transition(s0, action=[0.3, 0.3, 0.3], rng=None, params=params)
        obs = fe.observation(s1, rng=None, params=params)
        r = fe.reward(s0, action=[0.3, 0.3, 0.3], next_state=s1, rng=None, params=params)
        done = fe.terminal(s1, rng=None, params=params)
    """

    def __init__(self, options: Optional[dict] = None) -> None:
        try:
            super().__init__(options=options)
        except TypeError:  # pragma: no cover - shim base takes no/other kwargs
            super().__init__()

    # -- param helper ------------------------------------------------------

    @staticmethod
    def default_params(**overrides: Any) -> dict[str, Any]:
        """Replay params (scenario knobs). Override any of ``n_symbols``, ``n_days``,
        ``distribution_mode``, ``max_weight``, ``allow_short``, ``mode``,
        ``window_start``/``window_end``, ``csv_text``, ``env_kwargs``."""
        return {**_default_params(), **overrides}

    # gymnasium's base also exposes ``get_default_params``; mirror it.
    def get_default_params(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D102
        return self.default_params(**kwargs)

    @staticmethod
    def _seed_from_rng(rng: Any) -> int:
        """Coerce a FuncEnv ``rng`` into the integer scenario seed our replay needs.

        FuncEnv is rng-threaded for JAX PRNG keys; our engine is seeded by a single int,
        so we accept an int directly, a numpy ``Generator``/``SeedSequence`` (draw one
        int), or ``None`` (seed 0). The seed fully determines the scenario, preserving
        the functional contract."""
        if rng is None:
            return 0
        if isinstance(rng, (int, np.integer)):
            return int(rng)
        if isinstance(rng, np.random.SeedSequence):
            return int(rng.generate_state(1)[0])
        if isinstance(rng, np.random.Generator):
            return int(rng.integers(0, 2**31 - 1))
        # Last resort: hash whatever was handed in into a stable seed.
        return int(abs(hash(rng)) % (2**31 - 1))

    # -- FuncEnv core ------------------------------------------------------

    def initial(self, rng: Any, params: Optional[dict] = None) -> State:
        """The empty-decision state: ``(seed, ())``. ``rng`` selects the scenario seed."""
        return (self._seed_from_rng(rng), tuple())

    def transition(
        self, state: State, action: Any, rng: Any = None, params: Optional[dict] = None
    ) -> State:
        """Append ``action`` to the decision sequence and return a *new* state tuple.

        Pure: ``state`` is not mutated (tuples are immutable), so calling ``transition``
        twice from the same ``state`` yields identical successors. ``rng`` is unused —
        the engine's execution noise is already pinned by the scenario seed."""
        seed, actions = state
        return (seed, actions + (_normalize_action(action),))

    def observation(
        self, state: State, rng: Any = None, params: Optional[dict] = None
    ) -> dict[str, np.ndarray]:
        """Decoded observation at ``state`` (replay the decision sequence)."""
        return replay(state, params).obs

    def reward(
        self,
        state: State,
        action: Any,
        next_state: State,
        rng: Any = None,
        params: Optional[dict] = None,
    ) -> float:
        """Reward earned by stepping from ``state`` to ``next_state`` via ``action``.

        Defined as the last reward of the ``next_state`` replay — i.e. the engine reward
        produced by the final decision in ``next_state``. ``action`` is accepted to match
        the FuncEnv signature and is expected to be ``next_state``'s last decision."""
        return replay(next_state, params).reward

    def terminal(
        self, state: State, rng: Any = None, params: Optional[dict] = None
    ) -> bool:
        """Whether ``state`` is done: engine bankruptcy (terminated) **or** end-of-window
        (truncated). FuncEnv collapses both into a single done flag."""
        r = replay(state, params)
        return r.done

    # -- optional info hooks (the base calls these if present) -------------

    def state_info(self, state: State, params: Optional[dict] = None) -> dict:
        """Diagnostics for a state: seed, decisions taken, and the replayed engine info
        (nav, events, …) at the state's frontier."""
        r = replay(state, params)
        seed, actions = state
        return {
            "seed": seed,
            "n_steps": len(actions),
            "terminated": r.terminated,
            "truncated": r.truncated,
            "engine_info": r.info,
        }

    def transition_info(
        self,
        state: State,
        action: Any,
        next_state: State,
        params: Optional[dict] = None,
    ) -> dict:
        """Diagnostics for a transition: the reward produced and the resulting info."""
        r = replay(next_state, params)
        return {
            "reward": r.reward,
            "terminated": r.terminated,
            "truncated": r.truncated,
            "engine_info": r.info,
        }

    # -- rendering: not supported (no visual surface) ----------------------

    def render_image(self, *args: Any, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("OpenOutcryFuncEnv has no visual rendering")
