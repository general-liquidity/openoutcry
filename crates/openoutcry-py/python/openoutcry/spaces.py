"""Flatten/unflatten helpers for OpenOutcry's Dict observation.

Thin wrappers over :mod:`gymnasium.spaces` (``flatten`` / ``unflatten`` / ``flatten_space``
/ ``flatdim``) plus a :class:`FlattenObservation` wrapper that turns the env's ``Dict`` obs
into a flat ``Box`` for SB3-style MLP feature extractors. The Dict observation stays the
package default; flattening is opt-in and exactly invertible (``unflatten`` reconstructs the
original Dict from the flat vector).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import flatten, unflatten, flatten_space, flatdim


def flatten_obs(observation_space: spaces.Space, obs: Any) -> np.ndarray:
    """Flatten ``obs`` into a 1-D vector consistent with ``observation_space``."""
    return flatten(observation_space, obs)


def unflatten_obs(observation_space: spaces.Space, x: np.ndarray) -> Any:
    """Reconstruct a structured observation from its flat vector (inverse of :func:`flatten_obs`)."""
    return unflatten(observation_space, x)


def flat_dim(observation_space: spaces.Space) -> int:
    """Length of the flat vector that :func:`flatten_obs` produces for ``observation_space``."""
    return flatdim(observation_space)


class FlattenObservation(gym.ObservationWrapper):
    """Expose the env's ``Dict`` observation as a flat ``Box`` (MLP-friendly).

    The flattened ``observation_space`` is :func:`gymnasium.spaces.flatten_space` of the
    wrapped env's space; :meth:`unflatten` recovers the original Dict observation exactly.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._dict_space = env.observation_space
        self.observation_space = flatten_space(env.observation_space)

    def observation(self, observation: Any) -> np.ndarray:
        return flatten(self._dict_space, observation)

    def unflatten(self, observation: np.ndarray) -> Any:
        """Invert :meth:`observation`, recovering the original Dict observation."""
        return unflatten(self._dict_space, observation)


__all__ = [
    "flatten_obs",
    "unflatten_obs",
    "flat_dim",
    "FlattenObservation",
]
