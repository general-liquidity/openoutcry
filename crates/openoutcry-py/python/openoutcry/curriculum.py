"""Sequential / curriculum scenario chaining over :class:`OpenOutcryEnv`.

Procgen's ``use_sequential_levels`` chains level seeds (``current_level_seed + 997``)
into a continuous curriculum instead of resampling i.i.d. The trading analog: walk an
ordered list of scenario seeds (or a deterministic seed-chaining rule) so an agent
experiences a fixed sequence of regimes — e.g. calm → shock → recovery — as successive
episodes, rather than independently sampled ones.

:class:`CurriculumEnv` is a thin :class:`gymnasium.Wrapper` that, on every ``reset()``,
points the wrapped env at the *next* scenario seed in its schedule. The schedule is
fixed up front (an explicit seed list or a pure ``base + step`` rule), so the same
constructor args always yield the same episode sequence and seeds are never drawn from
future bars (leak-free).

:func:`regime_curriculum` rotates the synthetic ``distribution_mode`` (calm/hard/extreme)
across episodes. Because :class:`OpenOutcryEnv` fixes ``distribution_mode`` at
construction, the wrapper rebuilds the underlying env per ``reset()`` via an
``env_factory`` — the documented approach for changing a construction-time parameter
between episodes.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import gymnasium as gym

from .gym import OpenOutcryEnv

# Procgen's sequential-level stride; chaining seeds by a fixed coprime-ish step keeps
# successive scenarios far apart in seed space while staying a pure function of k.
_CHAIN_STEP = 997

EnvFactory = Callable[[int], gym.Env]


class CurriculumEnv(gym.Wrapper):
    """Advance through an ordered schedule of scenario seeds, one per ``reset()``.

    Parameters
    ----------
    seeds:
        Explicit ordered scenario seeds to walk. If ``None``, a deterministic chain of
        length ``n_episodes`` is generated from the base seed (``env_kwargs['seed']``,
        default 0) per ``schedule``.
    schedule:
        ``"sequential"`` → ``base + k``; ``"chained"`` → Procgen's ``base + 997*k``.
        When ``seeds`` is given this is only a descriptive label surfaced in ``info``.
    n_episodes:
        Length of the generated chain (required when ``seeds`` is ``None``).
    loop:
        If ``True`` the schedule wraps around at the end; if ``False`` the schedule is
        consumed once, after which each further ``reset()`` clamps to the last seed and
        flags ``info['curriculum_exhausted'] = True``.
    env_factory:
        Optional ``(episode_index) -> gym.Env`` builder used to rebuild the underlying
        env per episode (e.g. to rotate a construction-time ``distribution_mode``). When
        ``None`` a single :class:`OpenOutcryEnv` is built from ``env_kwargs`` and reused
        (``reset(seed=...)`` re-points it at the next scenario).
    **env_kwargs:
        Forwarded to :class:`OpenOutcryEnv` when ``env_factory`` is ``None``.
    """

    def __init__(
        self,
        seeds: Optional[Sequence[int]] = None,
        *,
        schedule: str = "sequential",
        n_episodes: Optional[int] = None,
        loop: bool = True,
        env_factory: Optional[EnvFactory] = None,
        **env_kwargs,
    ) -> None:
        if seeds is not None:
            self._seeds: list[int] = [int(s) for s in seeds]
            if not self._seeds:
                raise ValueError("seeds must be non-empty")
        else:
            if schedule not in ("sequential", "chained"):
                raise ValueError("schedule must be 'sequential' or 'chained'")
            if n_episodes is None or n_episodes < 1:
                raise ValueError("n_episodes (>= 1) is required when seeds is None")
            base = int(env_kwargs.get("seed", 0))
            step = 1 if schedule == "sequential" else _CHAIN_STEP
            self._seeds = [base + k * step for k in range(int(n_episodes))]

        self._schedule = schedule
        self._loop = bool(loop)
        self._env_factory = env_factory
        self._episode = 0
        self.curriculum_exhausted = False

        env = env_factory(0) if env_factory is not None else OpenOutcryEnv(**env_kwargs)
        super().__init__(env)

    @property
    def seeds(self) -> list[int]:
        """The resolved scenario-seed schedule (copy)."""
        return list(self._seeds)

    @property
    def schedule(self) -> str:
        return self._schedule

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset onto the next scheduled scenario seed (the curriculum drives the seed;
        any externally supplied ``seed`` is ignored so the sequence stays deterministic).
        """
        n = len(self._seeds)
        pos = self._episode
        exhausted = False
        if pos >= n:
            if self._loop:
                idx = pos % n
            else:
                exhausted = True
                idx = n - 1
        else:
            idx = pos
        self.curriculum_exhausted = exhausted

        active_seed = self._seeds[idx]
        if self._env_factory is not None:
            self.env = self._env_factory(idx)
        obs, info = self.env.reset(seed=active_seed)

        curriculum = {"index": idx, "seed": active_seed, "schedule": self._schedule}
        mode = getattr(self.env, "_distribution_mode", None)
        if mode is not None:
            curriculum["distribution_mode"] = mode
        info["curriculum"] = curriculum
        if exhausted:
            info["curriculum_exhausted"] = True

        self._episode += 1
        return obs, info


def regime_curriculum(
    base_seed: int,
    length: int,
    *,
    distribution_modes: Sequence[str] = ("calm", "hard", "extreme"),
    schedule: str = "chained",
    **env_kwargs,
) -> CurriculumEnv:
    """A :class:`CurriculumEnv` whose successive episodes rotate the difficulty tier.

    Episode ``k`` runs ``distribution_modes[k % len(distribution_modes)]`` over a
    chained scenario seed — the literal "calm → shock → recovery" curriculum. Because
    ``distribution_mode`` is fixed at :class:`OpenOutcryEnv` construction, each episode
    gets a freshly built env (via ``env_factory``); the seed for the episode is then
    applied by the wrapper's ``reset``.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if not distribution_modes:
        raise ValueError("distribution_modes must be non-empty")
    if schedule not in ("sequential", "chained"):
        raise ValueError("schedule must be 'sequential' or 'chained'")

    step = 1 if schedule == "sequential" else _CHAIN_STEP
    seeds = [int(base_seed) + k * step for k in range(int(length))]
    modes = list(distribution_modes)

    def _factory(idx: int) -> gym.Env:
        return OpenOutcryEnv(
            distribution_mode=modes[idx % len(modes)],
            seed=seeds[idx],
            **env_kwargs,
        )

    return CurriculumEnv(
        seeds=seeds,
        schedule="regime",
        loop=True,
        env_factory=_factory,
    )


__all__ = ["CurriculumEnv", "regime_curriculum"]
