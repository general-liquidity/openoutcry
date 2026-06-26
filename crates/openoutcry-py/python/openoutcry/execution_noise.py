"""Execution-noise wrapper — the trading analog of ALE sticky actions.

Open-loop trajectory memorization is the vector-env analog of an Atari agent learning a
fixed button sequence: a policy can overfit a deterministic point-in-time market by
replaying a memorized action path. :class:`ExecutionNoiseWrapper` perturbs the *realized*
action relative to the *requested* one — repeating the previous action one bar late (the
delay/"sticky" knob) and/or jittering the target weights (the slippage knob) — which breaks
that open-loop replay while leaving a closed-loop policy that reacts to observations intact.

All randomness comes from a single seeded :class:`numpy.random.Generator`, so a given
``(seed, delay_prob, slippage_bps)`` produces a byte-reproducible perturbation stream — the
reproducible-trace guarantee is preserved. Both knobs default to zero (pass-through).

This is a **reportable benchmark-integrity knob**: any eval run that enables a non-zero
``delay_prob`` or ``slippage_bps`` MUST disclose those values alongside its scores, since
they change the difficulty of the task.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym


class ExecutionNoiseWrapper(gym.Wrapper):
    """Perturb the realized action vs. the requested one (seeded, default-off).

    With probability ``delay_prob`` the previous step's realized action is applied instead
    of the requested one (the order lands one bar late). Independently, multiplicative
    Gaussian jitter with scale ``slippage_bps`` basis points is added to the (post-delay)
    target weights. The result is clipped back into the env's action space. With both knobs
    at their ``0.0`` default the requested action passes through unchanged and no random
    draws are taken.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        delay_prob: float = 0.0,
        slippage_bps: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self._delay_prob = float(delay_prob)
        self._slippage_bps = float(slippage_bps)
        self._rng = np.random.default_rng(seed)
        self._prev_action: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        realized = np.asarray(action, dtype=np.float64).copy()

        if self._delay_prob > 0.0 and self._prev_action is not None:
            if self._rng.random() < self._delay_prob:
                realized = self._prev_action.copy()

        if self._slippage_bps > 0.0:
            jitter = self._rng.normal(
                0.0, self._slippage_bps * 1e-4, size=realized.shape
            )
            realized = realized * (1.0 + jitter)

        space = self.env.action_space
        realized = np.clip(realized, space.low, space.high).astype(space.dtype)
        self._prev_action = realized.copy()
        return self.env.step(realized)


__all__ = ["ExecutionNoiseWrapper"]
