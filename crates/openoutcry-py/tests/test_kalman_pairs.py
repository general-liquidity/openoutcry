"""Tests for :class:`openoutcry.pairs.KalmanSpreadObservation`.

The convergence / signal tests run against a fake Dict-obs env replaying a
*constructed* cointegrated pair (``y = alpha + beta*x + OU-spread``) so the true
hedge ratio and the spread excursions are known; the integration test over a real
:class:`OpenOutcryEnv` is skipped when the compiled binding is unavailable.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_kalman_pairs.py -q
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from openoutcry.pairs import KalmanSpreadObservation


class _PairEnv(gym.Env):
    """Minimal Dict-obs env replaying a fixed two-symbol close series."""

    def __init__(self, closes: np.ndarray) -> None:
        self._closes = np.asarray(closes, dtype=np.float64)  # (T, 2)
        self._t = 0
        self.observation_space = spaces.Dict(
            {
                "closes": spaces.Box(0.0, np.inf, shape=(2,), dtype=np.float64),
                "cash": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _obs(self) -> dict:
        return {
            "closes": self._closes[self._t].copy(),
            "cash": np.array([0.0], dtype=np.float64),
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        done = self._t >= self._closes.shape[0] - 1
        return self._obs(), 0.0, False, bool(done), {}


def _cointegrated_pair(
    n: int,
    beta: float,
    alpha: float,
    phi: float,
    seed: int,
    x_step: float = 2.0,
    spread_noise: float = 1.0,
) -> np.ndarray:
    """``x`` is a random walk; ``y = alpha + beta*x + s`` with ``s`` a mean-reverting
    OU spread. Returns an ``(n, 2)`` close matrix (column 0 == x, column 1 == y). The
    ``x_step``/``spread_noise`` ratio sets how well ``beta`` is identified."""
    rng = np.random.default_rng(seed)
    x = 100.0 + np.cumsum(rng.normal(0.0, x_step, size=n))
    s = np.empty(n, dtype=np.float64)
    s[0] = 0.0
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0.0, spread_noise)
    y = alpha + beta * x + s
    return np.column_stack([x, y])


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _run(env, n_steps: int):
    """Drive the env, collecting per-bar closes / spread / z."""
    obs, _ = env.reset()
    closes = [obs["closes"].copy()]
    spreads = [float(obs["kalman_spread"][0])]
    zs = [float(obs["kalman_spread_z"][0])]
    for _ in range(n_steps):
        obs, _r, term, trunc, _i = env.step(_zero_action(env))
        closes.append(obs["closes"].copy())
        spreads.append(float(obs["kalman_spread"][0]))
        zs.append(float(obs["kalman_spread_z"][0]))
        if term or trunc:
            break
    return closes, spreads, zs


def test_adds_keys_with_shape_and_space():
    env = KalmanSpreadObservation(_PairEnv(_cointegrated_pair(50, 1.5, 5.0, 0.8, 1)))
    obs, _ = env.reset()
    assert obs["kalman_spread"].shape == (1,)
    assert obs["kalman_spread_z"].shape == (1,)
    assert {"closes", "cash"}.issubset(obs)
    assert env.observation_space.contains(obs)


def test_first_obs_is_history_free_zero():
    env = KalmanSpreadObservation(_PairEnv(_cointegrated_pair(50, 1.5, 5.0, 0.8, 2)))
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["kalman_spread"], np.zeros(1))
    np.testing.assert_allclose(obs["kalman_spread_z"], np.zeros(1))


def test_hedge_ratio_converges_to_true_beta():
    true_beta = 1.5
    env = KalmanSpreadObservation(
        _PairEnv(_cointegrated_pair(600, true_beta, 5.0, 0.8, 1))
    )
    _run(env, 599)
    assert env.hedge_ratio == pytest.approx(true_beta, abs=0.2)


def test_innovation_z_crosses_thresholds_and_is_finite():
    env = KalmanSpreadObservation(
        _PairEnv(_cointegrated_pair(600, 1.5, 5.0, 0.8, 4))
    )
    _closes, _spreads, zs = _run(env, 599)
    z = np.asarray(zs, dtype=np.float64)
    assert np.all(np.isfinite(z))
    # A mean-reverting spread produces standardized-innovation excursions past +/-2
    # once the filter has locked on to the hedge ratio (skip the warmup transient).
    settled = z[100:]
    assert settled.max() > 2.0
    assert settled.min() < -2.0


def test_is_leak_free_prefix_reproduces_emitted_value():
    """A future bar can never change a past output: feeding only the prefix
    ``closes[0..t]`` to a fresh filter reproduces the value emitted at bar ``t``."""
    pair = _cointegrated_pair(200, 1.5, 5.0, 0.85, 5)
    full = KalmanSpreadObservation(_PairEnv(pair))
    closes, spreads, zs = _run(full, 199)
    for t in (5, 40, 120, len(spreads) - 1):
        prefix = KalmanSpreadObservation(_PairEnv(np.asarray(closes[: t + 1])))
        _c, ps, pzs = _run(prefix, t)
        assert ps[-1] == pytest.approx(spreads[t], abs=1e-9)
        assert pzs[-1] == pytest.approx(zs[t], abs=1e-9)


def test_buffer_resets_on_episode_boundary():
    env = KalmanSpreadObservation(_PairEnv(_cointegrated_pair(60, 1.5, 5.0, 0.8, 3)))
    _run(env, 30)
    # A fresh reset drops all accumulated state; the first bar is history-free zero
    # again (the bar is folded into a from-scratch [0, 0] state, so it emits 0).
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["kalman_spread"], np.zeros(1))
    np.testing.assert_allclose(obs["kalman_spread_z"], np.zeros(1))


def test_requires_two_symbols():
    single = spaces.Dict({"closes": spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float64)})

    class _Bad(gym.Env):
        observation_space = single
        action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    with pytest.raises(AssertionError):
        KalmanSpreadObservation(_Bad())


def test_rejects_bad_hyperparameters():
    pair = _cointegrated_pair(20, 1.0, 0.0, 0.5, 1)
    with pytest.raises(AssertionError):
        KalmanSpreadObservation(_PairEnv(pair), delta=0.0)
    with pytest.raises(AssertionError):
        KalmanSpreadObservation(_PairEnv(pair), delta=1.0)
    with pytest.raises(AssertionError):
        KalmanSpreadObservation(_PairEnv(pair), obs_var=0.0)


# -- integration over the real env -------------------------------------------

try:
    import openoutcry.openoutcry_py  # noqa: F401

    _HAVE_BINDING = True
except Exception:  # pragma: no cover - environment-dependent
    _HAVE_BINDING = False

_needs_binding = pytest.mark.skipif(
    not _HAVE_BINDING, reason="native binding not built"
)


@_needs_binding
def test_wraps_real_cointegrated_env():
    from openoutcry.gym import OpenOutcryEnv

    base = OpenOutcryEnv(
        n_symbols=4, n_days=60, seed=5, distribution_mode="cointegrated_pairs"
    )
    env = KalmanSpreadObservation(base)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    for _ in range(40):
        obs, _r, term, trunc, _i = env.step(
            np.zeros(env.action_space.shape, np.float32)
        )
        assert np.all(np.isfinite(obs["kalman_spread"]))
        assert np.all(np.isfinite(obs["kalman_spread_z"]))
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
