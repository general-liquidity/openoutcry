"""Tests for the synthetic news/sentiment channel (Stream S8).

The pure helper and the wrapper are exercised without the native binding via a tiny
Dict-obs stub env; an optional live test runs against ``OpenOutcryEnv`` when the binding
imports.

    python -m pytest crates/openoutcry-py/tests/test_news.py -q
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from openoutcry.news import SyntheticNewsObservation, news_series


class _StubEnv(gym.Env):
    """Minimal Dict-obs env mirroring OpenOutcryEnv's price-only observation."""

    def __init__(self, n_symbols: int = 3, seed: int = 0) -> None:
        super().__init__()
        self._n = n_symbols
        self._seed = int(seed)
        self.symbols = [f"S{i}" for i in range(n_symbols)]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_symbols,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "closes": spaces.Box(0.0, np.inf, shape=(n_symbols,), dtype=np.float64),
                "positions": spaces.Box(-np.inf, np.inf, shape=(n_symbols,), dtype=np.float64),
                "cash": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )

    def _obs(self) -> dict:
        return {
            "closes": np.ones((self._n,), dtype=np.float64) * 100.0,
            "positions": np.zeros((self._n,), dtype=np.float64),
            "cash": np.array([1.0], dtype=np.float64),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
        return self._obs(), {"scenario_seed": self._seed}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


# -- pure helper ------------------------------------------------------------

def test_determinism_same_seed():
    a = news_series(42, 20, 3, intensity=1.0, lag=1)
    b = news_series(42, 20, 3, intensity=1.0, lag=1)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    a = news_series(1, 20, 3)
    b = news_series(2, 20, 3)
    assert not np.array_equal(a, b)


def test_bounds():
    s = news_series(7, 50, 4, intensity=1.0)
    assert s.shape == (50, 4)
    assert np.all(s >= -1.0) and np.all(s <= 1.0)


def test_intensity_zero_is_all_zero():
    s = news_series(7, 30, 4, intensity=0.0)
    assert np.count_nonzero(s) == 0


def test_intensity_scales_amplitude():
    full = news_series(9, 30, 3, intensity=1.0, lag=0)
    half = news_series(9, 30, 3, intensity=0.5, lag=0)
    np.testing.assert_allclose(half, np.clip(full * 0.5, -1.0, 1.0), atol=1e-6)


def test_lag_shifts_signal():
    base = news_series(13, 25, 3, intensity=1.0, lag=0)
    lagged = news_series(13, 25, 3, intensity=1.0, lag=1)
    assert np.all(lagged[0] == 0.0)
    np.testing.assert_array_equal(lagged[1:], base[:-1])


def test_lag_zero_reveals_current_latent():
    s0 = news_series(13, 10, 3, lag=0)
    s2 = news_series(13, 12, 3, lag=2)
    # latent index 0 is revealed at bar 0 (lag=0) and at bar 2 (lag=2).
    np.testing.assert_array_equal(s2[2], s0[0])


# -- leak-free property -----------------------------------------------------

def test_sentiment_computable_before_step():
    """The whole series is a pure function of (seed, t) — computable up front, before
    any bar is stepped — so it cannot encode the bar-t return."""
    seed = 99
    precomputed = news_series(seed, 5, 3, intensity=1.0, lag=1)
    env = SyntheticNewsObservation(_StubEnv(3, seed), intensity=1.0, lag=1)
    obs, _ = env.reset(seed=seed)
    rows = [obs["sentiment"]]
    for _ in range(4):
        obs, *_ = env.step(env.action_space.sample())
        rows.append(obs["sentiment"])
    np.testing.assert_array_equal(np.stack(rows, axis=0), precomputed)


def test_revealed_value_predates_bar():
    """Sentiment at bar t equals the latent generated for index t-lag, i.e. data already
    fixed lag bars earlier — never bar t's own (future) draw."""
    seed = 5
    lag = 2
    env = SyntheticNewsObservation(_StubEnv(3, seed), intensity=1.0, lag=lag)
    obs, _ = env.reset(seed=seed)
    latent = news_series(seed, 6, 3, intensity=1.0, lag=0)
    rows = [obs["sentiment"]]
    for _ in range(5):
        obs, *_ = env.step(env.action_space.sample())
        rows.append(obs["sentiment"])
    # bar i reveals the latent value fixed `lag` bars earlier; warmup bars (i < lag)
    # are zero and carry no latent value to compare.
    for i in range(lag, 6):
        np.testing.assert_array_equal(rows[i], latent[i - lag])


# -- wrapper integration ----------------------------------------------------

def test_wrapper_adds_sentiment_key_and_shape():
    env = SyntheticNewsObservation(_StubEnv(3, 0))
    assert "sentiment" in env.observation_space.spaces
    obs, _ = env.reset(seed=0)
    assert obs["sentiment"].shape == (3,)
    assert obs["sentiment"].dtype == np.float32
    assert set(obs) == {"closes", "positions", "cash", "sentiment"}


def test_baseline_obs_unaffected_without_wrapper():
    env = _StubEnv(3, 0)
    obs, _ = env.reset(seed=0)
    assert "sentiment" not in obs


def test_intensity_zero_channel_all_zero_in_wrapper():
    env = SyntheticNewsObservation(_StubEnv(4, 3), intensity=0.0)
    obs, _ = env.reset(seed=3)
    assert np.count_nonzero(obs["sentiment"]) == 0
    for _ in range(5):
        obs, *_ = env.step(env.action_space.sample())
        assert np.count_nonzero(obs["sentiment"]) == 0


def test_headlines_in_info_not_obs():
    env = SyntheticNewsObservation(_StubEnv(4, 11), intensity=1.0, headlines=True)
    obs, info = env.reset(seed=11)
    assert "headlines" in info
    assert isinstance(info["headlines"], list)
    assert "headlines" not in obs
    saw_headline = bool(info["headlines"])
    for _ in range(20):
        obs, _, _, _, info = env.step(env.action_space.sample())
        for h in info["headlines"]:
            assert ":" in h
        saw_headline = saw_headline or bool(info["headlines"])
    assert saw_headline


def test_headlines_off_by_default():
    env = SyntheticNewsObservation(_StubEnv(3, 1), intensity=1.0)
    _, info = env.reset(seed=1)
    assert "headlines" not in info


# -- optional live binding --------------------------------------------------

def test_live_wrapped_env():
    pytest.importorskip("openoutcry.openoutcry_py")
    from openoutcry import OpenOutcryEnv

    base = OpenOutcryEnv(n_symbols=3, n_days=40, seed=4)
    env = SyntheticNewsObservation(base, intensity=1.0, lag=1, headlines=True)
    n = base.action_space.shape[0]
    obs, info = env.reset(seed=4)
    assert obs["sentiment"].shape == (n,)
    expected = news_series(4, 10, n, intensity=1.0, lag=1)
    rows = [obs["sentiment"]]
    for _ in range(9):
        obs, _, terminated, truncated, info = env.step(np.zeros((n,), dtype=np.float32))
        rows.append(obs["sentiment"])
        if terminated or truncated:
            break
    np.testing.assert_array_equal(np.stack(rows[: len(rows)], axis=0), expected[: len(rows)])
