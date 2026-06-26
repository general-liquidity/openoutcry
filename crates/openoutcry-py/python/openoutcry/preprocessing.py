"""Canonical, declarative observation-preprocessing defaults for OpenOutcry.

ALE exposes preprocessing as first-class declarative kwargs (frameskip, framestack,
grayscale, reward clip, ...) and freezes a *documented* default set, because a
benchmark's value is *agreement*: if everyone feeds the policy a different feature
pipeline, the scores aren't comparable. This module is the trading analog — a
:class:`PreprocessingConfig` of canonical knobs with frozen defaults, a one-call
:func:`make_preprocessed_env` constructor that applies the implied wrapper stack in a
fixed, documented order, and :func:`describe_preprocessing` which renders the
"these settings MUST be disclosed alongside your score" summary.

The defaults are deliberately near-raw: the only default-on transform is causal
(point-in-time-safe) observation normalization, which conditions features without
introducing a future leak. Reward shaping, frame-stacking, flattening, episode caps,
and execution noise are all default-off, so the canonical pipeline does not silently
bake difficulty or shaping into the benchmark. Raw-obs access stays available: build
with ``PreprocessingConfig(causal_normalize_obs=False, lookback=1)`` and the base env
passes through unwrapped except for episode bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym

from .gym import OpenOutcryEnv
from .wrappers import (
    TimeLimit,
    CausalNormalizeObservation,
    CausalNormalizeReward,
    FrameStack,
    RecordEpisodeStatistics,
)
from .execution_noise import ExecutionNoiseWrapper
from .spaces import FlattenObservation
from .indicators import CausalIndicatorObservation, INDICATORS


@dataclass(frozen=True)
class ExecutionNoiseConfig:
    """Reportable benchmark-integrity knob (the trading analog of ALE sticky actions).

    Both fields default to ``0.0`` — pure pass-through, no random draws. A non-zero
    value changes task difficulty and therefore MUST be disclosed with any score.
    """

    # Probability the previous bar's realized action lands instead (order one bar late).
    delay_prob: float = 0.0
    # Bounded multiplicative target-weight jitter, in basis points.
    slippage_bps: float = 0.0
    # Stream seed for the perturbation; None => nondeterministic-but-stable per instance.
    seed: Optional[int] = None

    @property
    def enabled(self) -> bool:
        return self.delay_prob != 0.0 or self.slippage_bps != 0.0


@dataclass(frozen=True)
class PreprocessingConfig:
    """The canonical, declarative observation-preprocessing knob set.

    Each field documents its frozen default and *why* that default was chosen. Pass an
    instance to :func:`make_preprocessed_env`; the wrapper stack is derived from the
    config, never hand-assembled at the call site.
    """

    # Frame-stack window. Default 1 = no stacking (the env obs is already a trailing
    # close_history; a Markov policy needs no extra temporal context by default). N>1
    # stacks the last N observations along a new leading axis.
    lookback: int = 1
    # Causal (point-in-time-safe) z-scoring of observation features. Default ON: it
    # conditions features without a future leak (stats use bars 0..t-1 only). This is
    # the one default-on transform — better-conditioned inputs with no integrity cost.
    causal_normalize_obs: bool = True
    # Causal reward scaling by running std of PAST rewards. Default OFF: reward scale is
    # part of the benchmark contract; rescaling it changes what "good" means, so it is
    # opt-in and must be disclosed.
    causal_normalize_reward: bool = False
    # Symmetric reward clip to +/- this magnitude. Default None = no clip; clipping is a
    # shaping choice that alters the objective and must be disclosed when used.
    reward_clip: Optional[float] = None
    # Episode-length cap (truncation). Default None = use the env's own window length.
    max_episode_steps: Optional[int] = None
    # Named causal technical-indicator block (the freqtrade declarative-list pattern).
    # Default empty = no indicators (near-raw obs). Each name adds one scalar per symbol
    # under obs["indicators"], computed from a leak-free rolling close buffer. Adding
    # features changes what the policy sees, so it must be disclosed with any score.
    indicators: tuple[str, ...] = ()
    # Flatten the Dict obs into a single Box. Default False: the Dict obs is the package
    # default (named fields), flattening is an MLP-convenience opt-in.
    flatten: bool = False
    # Execution-noise sub-config (default off; a disclosed integrity knob).
    execution_noise: ExecutionNoiseConfig = field(default_factory=ExecutionNoiseConfig)

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if self.reward_clip is not None and self.reward_clip <= 0.0:
            raise ValueError("reward_clip must be > 0 when set")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1 when set")
        unknown = [name for name in self.indicators if name not in INDICATORS]
        if unknown:
            raise ValueError(f"unknown indicators: {unknown}")


# The OpenOutcry standard input pipeline. Freeze this; report deviations from it.
CANONICAL_PREPROCESSING = PreprocessingConfig()


class _RewardClip(gym.Wrapper):
    """Clip reward to ``[-limit, +limit]`` (symmetric)."""

    def __init__(self, env: gym.Env, limit: float) -> None:
        super().__init__(env)
        self._limit = float(limit)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = float(reward)
        if r > self._limit:
            r = self._limit
        elif r < -self._limit:
            r = -self._limit
        return obs, r, terminated, truncated, info


def make_preprocessed_env(
    config: Optional[PreprocessingConfig] = None, **env_kwargs
) -> gym.Env:
    """Build an :class:`OpenOutcryEnv` and apply the config's wrapper stack in order.

    Wrapper order (outermost last) — each layer wraps the one before it:

    1. ``OpenOutcryEnv(**env_kwargs)`` — the base point-in-time env.
    2. ``TimeLimit`` — truncate first, so every later layer sees the capped horizon.
    3. ``ExecutionNoiseWrapper`` — perturb the *realized* action before it reaches the
       env, but after the episode boundary is fixed.
    4. ``CausalIndicatorObservation`` — append the named indicator block from a leak-free
       rolling close buffer, before frame-stacking so each frame carries indicators.
    5. ``FrameStack`` — assemble temporal context from raw obs (before normalization, so
       each stacked frame is normalized consistently downstream).
    6. ``CausalNormalizeObservation`` — z-score features using past bars only.
    7. ``CausalNormalizeReward`` / reward clip — reward transforms last among the
       reward-touching layers; normalize-then-clip if both are set.
    8. ``RecordEpisodeStatistics`` — observe the *final* reward (post clip/normalize) so
       ``info["episode"]`` reflects what the agent actually optimized.
    9. ``FlattenObservation`` — optional outermost view; flattens whatever obs shape the
       inner stack produced (stacked + normalized Dict -> flat Box).
    """
    cfg = config if config is not None else CANONICAL_PREPROCESSING
    env: gym.Env = OpenOutcryEnv(**env_kwargs)

    if cfg.max_episode_steps is not None:
        env = TimeLimit(env, cfg.max_episode_steps)

    if cfg.execution_noise.enabled:
        env = ExecutionNoiseWrapper(
            env,
            delay_prob=cfg.execution_noise.delay_prob,
            slippage_bps=cfg.execution_noise.slippage_bps,
            seed=cfg.execution_noise.seed,
        )

    if cfg.indicators:
        env = CausalIndicatorObservation(env, indicators=cfg.indicators)

    if cfg.lookback > 1:
        env = FrameStack(env, cfg.lookback)

    if cfg.causal_normalize_obs:
        env = CausalNormalizeObservation(env)

    if cfg.causal_normalize_reward:
        env = CausalNormalizeReward(env)

    if cfg.reward_clip is not None:
        env = _RewardClip(env, cfg.reward_clip)

    env = RecordEpisodeStatistics(env)

    if cfg.flatten:
        env = FlattenObservation(env)

    return env


def describe_preprocessing(config: Optional[PreprocessingConfig] = None) -> str:
    """Render the must-disclose preprocessing summary (the ALE "report these" contract).

    Lines flagged ``[DISCLOSE]`` are settings that change task difficulty or the
    objective and MUST be reported alongside any score for it to be comparable.
    """
    cfg = config if config is not None else CANONICAL_PREPROCESSING
    en = cfg.execution_noise
    lines = [
        "OpenOutcry preprocessing config",
        f"  lookback (frame-stack)      = {cfg.lookback}",
        f"  indicators                  = {list(cfg.indicators)}  [DISCLOSE]",
        f"  causal_normalize_obs        = {cfg.causal_normalize_obs}",
        f"  causal_normalize_reward     = {cfg.causal_normalize_reward}  [DISCLOSE]",
        f"  reward_clip                 = {cfg.reward_clip}  [DISCLOSE]",
        f"  max_episode_steps           = {cfg.max_episode_steps}  [DISCLOSE]",
        f"  flatten                     = {cfg.flatten}",
        f"  execution_noise.delay_prob  = {en.delay_prob}  [DISCLOSE]",
        f"  execution_noise.slippage_bps= {en.slippage_bps}  [DISCLOSE]",
    ]
    canonical = cfg == CANONICAL_PREPROCESSING
    lines.append(
        "  -> CANONICAL OpenOutcry pipeline (no disclosure needed)."
        if canonical
        else "  -> NON-CANONICAL: report the [DISCLOSE] settings with your score."
    )
    return "\n".join(lines)


__all__ = [
    "ExecutionNoiseConfig",
    "PreprocessingConfig",
    "CANONICAL_PREPROCESSING",
    "make_preprocessed_env",
    "describe_preprocessing",
]
