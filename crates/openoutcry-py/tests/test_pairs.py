"""Tests for :class:`openoutcry.pairs.SpreadObservation`.

Run from the crate dir after ``python -m maturin develop``::

    python -m pytest tests/test_pairs.py -q
"""

import numpy as np
import pytest

from openoutcry.gym import OpenOutcryEnv
from openoutcry.pairs import SpreadObservation

_ENV_KWARGS = dict(n_symbols=4, n_days=60, seed=5, distribution_mode="cointegrated_pairs")
_WINDOW = 12


def _zero_action(env) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def _spread_z_ref(buf: list[tuple[float, float]]) -> float:
    """Independent reference implementation of the documented z-score."""
    if len(buf) < 2:
        return 0.0
    arr = np.asarray(buf, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    mx, my = x.mean(), y.mean()
    var_x = float(np.sum((x - mx) ** 2))
    if var_x <= 1e-8:
        return 0.0
    beta = float(np.sum((x - mx) * (y - my)) / var_x)
    spread = y - ((my - beta * mx) + beta * x)
    sd = float(spread.std())
    if sd <= 1e-8:
        return 0.0
    return float((spread[-1] - spread.mean()) / (sd + 1e-8))


def _make() -> SpreadObservation:
    return SpreadObservation(OpenOutcryEnv(**_ENV_KWARGS), window=_WINDOW)


def test_adds_spread_zscore_key_with_shape():
    env = _make()
    obs, _info = env.reset()
    assert "spread_zscore" in obs
    assert obs["spread_zscore"].shape == (1,)
    # The original Dict keys are preserved alongside the new feature.
    assert {"closes", "positions", "cash"}.issubset(obs)
    assert env.observation_space.contains(obs)


def test_first_obs_is_history_free_zero():
    env = _make()
    obs, _ = env.reset()
    # A single buffered bar has no spread distribution yet → exactly 0.
    np.testing.assert_allclose(obs["spread_zscore"], np.zeros(1))


def test_zscore_is_finite_through_episode():
    env = _make()
    env.reset()
    for _ in range(40):
        obs, _r, terminated, truncated, _info = env.step(_zero_action(env))
        assert np.all(np.isfinite(obs["spread_zscore"]))
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break


def test_zscore_is_causal_trailing_window_only():
    """The z-score at bar t must be a pure function of the trailing `window`
    closes up to and including t — i.e. recomputing it from only those bars (no
    future bar) reproduces the emitted value exactly."""
    env = _make()
    obs, _ = env.reset()
    recorded_closes: list[tuple[float, float]] = [
        (float(obs["closes"][0]), float(obs["closes"][1]))
    ]
    emitted: list[float] = [float(obs["spread_zscore"][0])]
    for _ in range(30):
        obs, _r, terminated, truncated, _info = env.step(_zero_action(env))
        recorded_closes.append((float(obs["closes"][0]), float(obs["closes"][1])))
        emitted.append(float(obs["spread_zscore"][0]))
        if terminated or truncated:
            break
    # Recompute every step's score from ONLY the trailing window of past closes.
    for t in range(len(emitted)):
        lo = max(0, t - _WINDOW + 1)
        ref = _spread_z_ref(recorded_closes[lo : t + 1])
        assert ref == pytest.approx(emitted[t], abs=1e-9), f"leak/mismatch at step {t}"


def test_buffer_resets_on_episode_boundary():
    env = _make()
    env.reset()
    for _ in range(10):
        env.step(_zero_action(env))
    # A fresh reset must drop all accumulated history → first score is 0 again.
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["spread_zscore"], np.zeros(1))


def test_requires_two_symbols():
    with pytest.raises(AssertionError):
        SpreadObservation(OpenOutcryEnv(n_symbols=1, n_days=30, seed=1))


def test_window_must_be_at_least_two():
    with pytest.raises(AssertionError):
        SpreadObservation(OpenOutcryEnv(**_ENV_KWARGS), window=1)
