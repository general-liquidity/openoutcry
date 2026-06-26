"""Tests for the causal technical-indicator observation wrapper.

The pure-numpy indicator table and the wrapper's causal buffer are tested against a
fake Dict-obs env (no native binding needed). Tests that wrap a real
:class:`OpenOutcryEnv` are skipped when the compiled binding is unavailable.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_indicators.py -q
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from openoutcry.indicators import (
    CausalIndicatorObservation,
    INDICATORS,
    DEFAULT_INDICATORS,
)

_ENV_KWARGS = dict(n_symbols=3, n_days=60, seed=1)


class _FakeDictEnv(gym.Env):
    """Minimal Dict-obs env that replays a fixed per-symbol close series."""

    def __init__(self, closes: np.ndarray) -> None:
        # closes: (T, n_symbols), oldest-first.
        self._closes = np.asarray(closes, dtype=np.float64)
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
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        done = self._t >= self._closes.shape[0] - 1
        return self._obs(), 0.0, False, bool(done), {}


def _series(n_steps: int, n_symbols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.02, size=(n_steps, n_symbols))
    return 100.0 * np.cumprod(1.0 + rets, axis=0)


# -- pure indicator table ---------------------------------------------------


def test_registry_covers_default_set():
    assert set(DEFAULT_INDICATORS) <= set(INDICATORS)
    assert set(DEFAULT_INDICATORS) == {
        "rsi", "sma", "ema", "macd", "bollinger", "realized_vol", "return",
    }


def test_warmup_returns_zero_sentinel():
    short = np.array([100.0, 101.0])  # below every window
    for name, fn in INDICATORS.items():
        if name == "return":  # return only needs 2 bars
            continue
        assert fn(short, 20) == 0.0, name


def test_indicators_finite_and_bounded_after_warmup():
    s = _series(80, 1, seed=3)[:, 0]
    assert 0.0 <= INDICATORS["rsi"](s, 20) <= 100.0
    assert INDICATORS["sma"](s, 20) > 0.0
    assert INDICATORS["ema"](s, 20) > 0.0
    for name in DEFAULT_INDICATORS:
        assert np.isfinite(INDICATORS[name](s, 20)), name


def test_return_matches_definition():
    s = np.array([100.0, 110.0])
    assert INDICATORS["return"](s, 20) == pytest.approx(0.1)


# -- wrapper over the fake env ----------------------------------------------


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def test_wrapper_shape_and_space():
    closes = _series(50, 3, seed=4)
    env = CausalIndicatorObservation(_FakeDictEnv(closes), DEFAULT_INDICATORS)
    obs, _ = env.reset()
    block = obs["indicators"]
    assert block.shape == (3, len(DEFAULT_INDICATORS))
    assert env.observation_space.contains(obs)
    for _ in range(40):
        obs, _r, _t, trunc, _i = env.step(_zero_action(env))
        if trunc:
            break
    assert np.all(np.isfinite(obs["indicators"]))


def test_first_obs_is_warmup_zeros_except_when_unavailable():
    closes = _series(50, 2, seed=5)
    env = CausalIndicatorObservation(_FakeDictEnv(closes), ("rsi", "sma", "macd"))
    obs, _ = env.reset()
    # One-bar history => every windowed indicator is the zero sentinel.
    np.testing.assert_allclose(obs["indicators"], np.zeros((2, 3)))


def test_causality_prefix_equality():
    # Two envs sharing a close prefix must emit identical indicator blocks over that
    # prefix: the block at step t depends only on closes 0..t, never future bars.
    shared = _series(40, 2, seed=6)
    diverged = shared.copy()
    diverged[25:] = _series(40, 2, seed=99)[25:]  # differ only after step 24

    inds = DEFAULT_INDICATORS
    ea = CausalIndicatorObservation(_FakeDictEnv(shared), inds)
    eb = CausalIndicatorObservation(_FakeDictEnv(diverged), inds)
    oa, _ = ea.reset()
    ob, _ = eb.reset()
    np.testing.assert_allclose(oa["indicators"], ob["indicators"])
    for t in range(24):  # steps 1..24 still inside the shared prefix
        oa, *_ = ea.step(_zero_action(ea))
        ob, *_ = eb.step(_zero_action(eb))
        np.testing.assert_allclose(
            oa["indicators"], ob["indicators"], err_msg=f"diverged at step {t + 1}"
        )


def test_buffer_resets_each_episode():
    closes = _series(30, 2, seed=7)
    env = CausalIndicatorObservation(_FakeDictEnv(closes), ("return",))
    env.reset()
    for _ in range(10):
        env.step(_zero_action(env))
    obs, _ = env.reset()
    # After reset the buffer holds one bar => return sentinel 0.0.
    np.testing.assert_allclose(obs["indicators"], np.zeros((2, 1)))


def test_unknown_indicator_rejected():
    with pytest.raises(ValueError):
        CausalIndicatorObservation(_FakeDictEnv(_series(10, 1, 1)), ("nope",))


# -- integration through the real env + preprocessing -----------------------

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
    env = CausalIndicatorObservation(base, DEFAULT_INDICATORS)
    obs, _ = env.reset()
    assert obs["indicators"].shape == (len(base.symbols), len(DEFAULT_INDICATORS))
    last = obs["indicators"]
    for _ in range(30):
        obs, _r, _t, trunc, _i = env.step(np.zeros(env.action_space.shape, np.float32))
        last = obs["indicators"]
        if trunc:
            break
    assert np.all(np.isfinite(last))


@_needs_binding
def test_make_preprocessed_env_with_indicators():
    from openoutcry.preprocessing import PreprocessingConfig, make_preprocessed_env

    cfg = PreprocessingConfig(indicators=("rsi", "sma"))
    env = make_preprocessed_env(cfg, **_ENV_KWARGS)
    obs, _ = env.reset()
    assert "indicators" in obs
    assert env.observation_space.contains(obs)
    obs, reward, _t, _tr, _i = env.step(np.zeros(env.action_space.shape, np.float32))
    assert np.isfinite(reward)
    assert np.all(np.isfinite(obs["indicators"]))
