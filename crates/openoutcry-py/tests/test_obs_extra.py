"""Tests for the extra causal observation-feature wrappers.

The wrappers are tested against a fake Dict-obs env (no native binding needed); the
integration test over a real :class:`OpenOutcryEnv` is skipped when the compiled
binding is unavailable.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_obs_extra.py -q
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from openoutcry.obs_extra import (
    MultiTimescaleMomentum,
    RollingCovarianceObservation,
    TimeToHorizonObservation,
    CounterfactualInfo,
)

_ENV_KWARGS = dict(n_symbols=3, n_days=80, seed=1)


class _FakeDictEnv(gym.Env):
    """Minimal Dict-obs env that replays a fixed per-symbol close series."""

    def __init__(self, closes: np.ndarray, nav: float = 1.0) -> None:
        self._closes = np.asarray(closes, dtype=np.float64)  # (T, n_symbols)
        self._nav = float(nav)
        self._t = 0
        n = self._closes.shape[1]
        self.observation_space = spaces.Dict(
            {
                "closes": spaces.Box(0.0, np.inf, shape=(n,), dtype=np.float64),
                "cash": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)

    def _obs(self) -> dict:
        return {
            "closes": self._closes[self._t].copy(),
            "cash": np.array([0.0], dtype=np.float64),
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {"nav": self._nav}

    def step(self, action):
        self._t += 1
        done = self._t >= self._closes.shape[0] - 1
        return self._obs(), 0.0, False, bool(done), {"nav": self._nav}


def _series(n_steps: int, n_symbols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, size=(n_steps, n_symbols))
    return 100.0 * np.cumprod(1.0 + rets, axis=0)


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


# -- MultiTimescaleMomentum --------------------------------------------------


def test_momentum_shape_and_space():
    horizons = (1, 5, 20)
    env = MultiTimescaleMomentum(_FakeDictEnv(_series(70, 3, 4)), horizons=horizons)
    obs, _ = env.reset()
    assert obs["momentum"].shape == (3, len(horizons))
    assert env.observation_space.contains(obs)
    last = obs["momentum"]
    for _ in range(60):
        obs, _r, _t, trunc, _i = env.step(_zero_action(env))
        last = obs["momentum"]
        if trunc:
            break
    assert np.all(np.isfinite(last))


def test_momentum_warmup_zeroed():
    env = MultiTimescaleMomentum(_FakeDictEnv(_series(40, 2, 5)), horizons=(1, 5))
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["momentum"], np.zeros((2, 2)))


def test_momentum_causality_prefix_equality():
    shared = _series(50, 2, seed=6)
    diverged = shared.copy()
    diverged[30:] = _series(50, 2, seed=99)[30:]
    ea = MultiTimescaleMomentum(_FakeDictEnv(shared), horizons=(1, 5, 20))
    eb = MultiTimescaleMomentum(_FakeDictEnv(diverged), horizons=(1, 5, 20))
    oa, _ = ea.reset()
    ob, _ = eb.reset()
    np.testing.assert_allclose(oa["momentum"], ob["momentum"])
    for t in range(29):
        oa, *_ = ea.step(_zero_action(ea))
        ob, *_ = eb.step(_zero_action(eb))
        np.testing.assert_allclose(
            oa["momentum"], ob["momentum"], err_msg=f"diverged at step {t + 1}"
        )


def test_momentum_rejects_bad_horizons():
    with pytest.raises(ValueError):
        MultiTimescaleMomentum(_FakeDictEnv(_series(10, 1, 1)), horizons=(0,))


# -- RollingCovarianceObservation --------------------------------------------


def test_covariance_shape_multi_asset():
    env = RollingCovarianceObservation(_FakeDictEnv(_series(60, 3, 7)), window=20)
    obs, _ = env.reset()
    assert obs["covariance"].shape == (6,)  # 3*(3+1)/2
    assert env.observation_space.contains(obs)
    last = obs["covariance"]
    for _ in range(50):
        obs, _r, _t, trunc, _i = env.step(_zero_action(env))
        last = obs["covariance"]
        if trunc:
            break
    assert np.all(np.isfinite(last))


def test_covariance_single_asset_is_variance():
    closes = _series(40, 1, 8)
    env = RollingCovarianceObservation(_FakeDictEnv(closes), window=10)
    env.reset()
    obs = None
    for _ in range(15):
        obs, *_ = env.step(_zero_action(env))
    assert obs["covariance"].shape == (1,)
    tail = closes[5:16]
    rets = np.diff(tail, axis=0)[:, 0] / tail[:-1, 0]
    assert obs["covariance"][0] == pytest.approx(float(np.var(rets, ddof=1)), rel=1e-6)


def test_covariance_warmup_zeroed():
    env = RollingCovarianceObservation(_FakeDictEnv(_series(30, 2, 9)), window=20)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["covariance"], np.zeros(3))


def test_covariance_causality_prefix_equality():
    shared = _series(50, 3, seed=11)
    diverged = shared.copy()
    diverged[30:] = _series(50, 3, seed=42)[30:]
    ea = RollingCovarianceObservation(_FakeDictEnv(shared), window=15)
    eb = RollingCovarianceObservation(_FakeDictEnv(diverged), window=15)
    oa, _ = ea.reset()
    ob, _ = eb.reset()
    np.testing.assert_allclose(oa["covariance"], ob["covariance"])
    for t in range(29):
        oa, *_ = ea.step(_zero_action(ea))
        ob, *_ = eb.step(_zero_action(eb))
        np.testing.assert_allclose(
            oa["covariance"], ob["covariance"], err_msg=f"diverged at step {t + 1}"
        )


# -- TimeToHorizonObservation ------------------------------------------------


def test_time_to_horizon_decreases_linearly():
    max_steps = 10
    env = TimeToHorizonObservation(_FakeDictEnv(_series(20, 2, 12)), max_steps=max_steps)
    obs, _ = env.reset()
    assert obs["time_to_horizon"].shape == (1,)
    assert env.observation_space.contains(obs)
    assert obs["time_to_horizon"][0] == pytest.approx(1.0)
    for step in range(1, max_steps + 1):
        obs, *_ = env.step(_zero_action(env))
        expected = (max_steps - step) / max_steps
        assert obs["time_to_horizon"][0] == pytest.approx(expected)


def test_time_to_horizon_clamped_nonnegative():
    env = TimeToHorizonObservation(_FakeDictEnv(_series(40, 1, 13)), max_steps=3)
    env.reset()
    obs = None
    for _ in range(10):
        obs, *_ = env.step(_zero_action(env))
    assert obs["time_to_horizon"][0] == pytest.approx(0.0)


# -- CounterfactualInfo ------------------------------------------------------


def test_counterfactual_in_info_not_obs():
    env = CounterfactualInfo(_FakeDictEnv(_series(40, 3, 14), nav=1000.0))
    obs, _ = env.reset()
    assert "counterfactual" not in obs
    for _ in range(20):
        obs, _r, _t, _tr, info = env.step(_zero_action(env))
        assert "counterfactual" not in obs
        cf = info["counterfactual"]
        assert cf["estimate"] is True
        assert cf["all_flat"]["reward"] == 0.0
        assert cf["all_flat"]["nav"] == pytest.approx(1000.0)
        assert np.isfinite(cf["all_long"]["reward"])
        assert np.isfinite(cf["all_long"]["nav"])


def test_counterfactual_long_matches_price_move():
    closes = _series(20, 2, 15)
    env = CounterfactualInfo(_FakeDictEnv(closes, nav=1.0))
    env.reset()
    # high == 1.0 per symbol => all-long return == sum of per-symbol bar returns.
    _o, _r, _t, _tr, info = env.step(_zero_action(env))
    r = closes[1] / closes[0] - 1.0
    assert info["counterfactual"]["all_long"]["reward"] == pytest.approx(float(r.sum()))


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
def test_wraps_real_env():
    from openoutcry.gym import OpenOutcryEnv

    base = OpenOutcryEnv(**_ENV_KWARGS)
    env = CounterfactualInfo(
        TimeToHorizonObservation(
            RollingCovarianceObservation(
                MultiTimescaleMomentum(base, horizons=(1, 5, 20)),
                window=20,
            ),
            max_steps=30,
        )
    )
    obs, _ = env.reset()
    n = len(base.symbols)
    assert obs["momentum"].shape == (n, 3)
    assert obs["covariance"].shape == (n * (n + 1) // 2,)
    assert obs["time_to_horizon"].shape == (1,)
    for _ in range(25):
        obs, _r, _t, trunc, info = env.step(
            np.zeros(env.action_space.shape, np.float32)
        )
        assert np.all(np.isfinite(obs["momentum"]))
        assert np.all(np.isfinite(obs["covariance"]))
        assert np.isfinite(info["counterfactual"]["all_long"]["nav"])
        if trunc:
            break
