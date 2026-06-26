"""Portfolio-allocation env over OpenOutcry's leak-free price panel.

:class:`PortfolioEnv` reframes the single-instrument, position-style
:class:`~openoutcry.gym.OpenOutcryEnv` as the canonical RL-finance allocation task: a
**simplex weight vector** over ``n_symbols`` assets plus cash, with a **log
portfolio-return** reward. The underlying env is reused verbatim — its leak-free,
point-in-time engine drives the price panel and its cost model (fees / slippage /
impact) prices every reallocation — so this layer never touches market dynamics.

Action → underlying mapping. The agent emits a length ``n_symbols + 1`` vector (index 0
is cash). A deterministic simplex normalization (long-only: clip ≥ 0, divide by the sum;
``allow_short``: divide by the L1 norm) yields weights that sum to 1; an all-zero action
maps to all-cash. The asset slice ``w[1:]`` is the per-symbol target-weight vector the
underlying ``step`` consumes, leaving ``w[0]`` in cash by construction.

Reward. The underlying ``step`` returns the costed simple bar return ``r`` (NAV change net
of execution costs) for the very target weights we hand it, so the costed log portfolio
return is ``log(1 + r)``, guarded for the bankruptcy edge ``r <= -1``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .gym import OpenOutcryEnv

# Floor for the log-return guard: a bar that wipes out NAV (r <= -1) maps to this instead
# of -inf, keeping the reward finite for bootstrapping.
_LOG_FLOOR = -1e3
_SUM_EPS = 1e-12


class PortfolioEnv(gym.Env):
    """Simplex allocation over ``n_symbols`` assets + cash with a log-return reward.

    Wraps an :class:`~openoutcry.gym.OpenOutcryEnv`; all constructor parameters not named
    here (``n_days``, ``distribution_mode``, ``mode``, ``window_start`` …) are forwarded to
    it. Set ``return_last_action=True`` to append the previous simplex weights to the
    observation (the FinRL ``env_portfolio_optimization`` pattern — makes turnover/cost
    observable to the policy).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_symbols: int = 4,
        *,
        return_last_action: bool = False,
        allow_short: bool = False,
        **env_kwargs: Any,
    ) -> None:
        super().__init__()
        self._allow_short = bool(allow_short)
        self._return_last_action = bool(return_last_action)
        # The underlying action space is the signed per-symbol target weight; long-only
        # clips its lower bound to 0. max_weight defaults to 1 (a simplex weight is ≤ 1).
        env_kwargs.setdefault("max_weight", 1.0)
        self._env = OpenOutcryEnv(
            n_symbols=n_symbols, allow_short=self._allow_short, **env_kwargs
        )
        n = len(self._env.symbols)
        self._n_assets = n
        self._n_actions = n + 1  # index 0 is cash

        low = -1.0 if self._allow_short else 0.0
        self.action_space = spaces.Box(
            low=low, high=1.0, shape=(self._n_actions,), dtype=np.float32
        )

        base = dict(self._env.observation_space.spaces)
        if self._return_last_action:
            base["last_action"] = spaces.Box(
                low=low, high=1.0, shape=(self._n_actions,), dtype=np.float64
            )
        self.observation_space = spaces.Dict(base)

        self._last_weights = self._all_cash()

    # -- internal helpers --------------------------------------------------

    def _all_cash(self) -> np.ndarray:
        w = np.zeros(self._n_actions, dtype=np.float64)
        w[0] = 1.0
        return w

    def _simplex(self, action: np.ndarray) -> np.ndarray:
        """Deterministically project ``action`` onto the (signed) simplex summing to 1.

        Long-only: clip negatives, divide by the sum. ``allow_short``: divide by the L1
        norm (so net signed weight is 1 and shorts are allowed). A degenerate (≈0) vector
        maps to all-cash."""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._allow_short:
            s = float(np.abs(a).sum())
            if s < _SUM_EPS:
                return self._all_cash()
            return a / s
        a = np.clip(a, 0.0, None)
        s = float(a.sum())
        if s < _SUM_EPS:
            return self._all_cash()
        return a / s

    def _augment(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._return_last_action:
            obs = dict(obs)
            obs["last_action"] = self._last_weights.astype(np.float64, copy=True)
        return obs

    @staticmethod
    def _log_return(r: float) -> float:
        if r <= -1.0:
            return _LOG_FLOOR
        return float(np.log1p(r))

    # -- gymnasium API -----------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return self._env.symbols

    @property
    def n_assets(self) -> int:
        return self._n_assets

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self._env.reset(seed=seed)
        self._last_weights = self._all_cash()
        return self._augment(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        weights = self._simplex(action)
        # Hand the asset slice to the underlying env as its target-weight vector; cash
        # (index 0) is the residual it holds, and its costed return prices the move.
        obs, r, terminated, truncated, info = self._env.step(weights[1:])
        reward = self._log_return(float(r))
        self._last_weights = weights
        info = dict(info)
        info["weights"] = weights
        info["simple_return"] = float(r)
        return self._augment(obs), reward, terminated, truncated, info

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        return self._env.close()


__all__ = ["PortfolioEnv"]
