"""Tests for the risk-termination / circuit-breaker wrappers (Stream S6).

Run from the crate dir::

    python -m pytest crates/openoutcry-py/tests/test_risk.py -q

The threshold/logic tests drive a deterministic NAV-scripted stub env so the stop-out and
halt steps are exact functions of a known NAV path (no binding needed). The live-binding
tests are skipped when the native ``openoutcry`` module is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from openoutcry.risk import DrawdownStopper, TurbulenceHalt, CrossSectionalDeleverage


class _NavEnv(gym.Env):
    """Stub env that replays a scripted NAV path, surfacing ``info["nav"]`` per step.

    ``terminated`` mirrors the base-env bankruptcy contract (``nav <= 0``); ``truncated``
    mirrors running out of bars. The last action actually executed is recorded so a wrapper
    that overrides the action can be observed.
    """

    def __init__(self, navs, n: int = 2) -> None:
        super().__init__()
        self._navs = [float(x) for x in navs]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        self._i = 0
        self.executed: np.ndarray | None = None

    def _obs(self):
        return np.zeros((1,), dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self.executed = None
        return self._obs(), {}

    def step(self, action):
        self.executed = np.asarray(action).copy()
        nav = self._navs[self._i]
        self._i += 1
        out_of_bars = self._i >= len(self._navs)
        terminated = nav <= 0.0
        truncated = out_of_bars and not terminated
        return self._obs(), 0.0, terminated, truncated, {"nav": nav}


class _SeededNavEnv(gym.Env):
    """Random-walk NAV env whose path is a deterministic function of the reset seed."""

    def __init__(self, steps: int = 80, n: int = 2) -> None:
        super().__init__()
        self._steps = steps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
        self._navs: list[float] = []
        self._i = 0

    def _gen(self, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0, 0.03, self._steps)
        rets[self._steps // 2] = 0.6  # deterministic vol spike
        nav = 1.0
        out = []
        for r in rets:
            nav *= 1.0 + r
            out.append(nav)
        return out

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._navs = self._gen(0 if seed is None else int(seed))
        self._i = 0
        return np.zeros((1,), dtype=np.float64), {}

    def step(self, action):
        nav = self._navs[self._i]
        self._i += 1
        out_of_bars = self._i >= len(self._navs)
        terminated = nav <= 0.0
        return np.zeros((1,), dtype=np.float64), 0.0, terminated, out_of_bars and not terminated, {"nav": nav}


def _act(env) -> np.ndarray:
    n = env.action_space.shape[0]
    return np.full((n,), 0.5, dtype=np.float32)


# -- DrawdownStopper --------------------------------------------------------

def test_drawdown_stopper_peak_fires_at_threshold():
    # peak=1.2 after step 2; max_drawdown=0.5 -> stop when nav <= 0.6.
    env = DrawdownStopper(_NavEnv([1.0, 1.2, 1.1, 0.5, 0.4]), max_drawdown=0.5)
    env.reset()
    a = _act(env)
    rows = [env.step(a) for _ in range(4)]
    truncs = [r[3] for r in rows]
    stopped = [bool(r[4].get("stopped_out", False)) for r in rows]
    assert truncs == [False, False, False, True]
    assert stopped == [False, False, False, True]
    # never terminated: NAV stayed positive, so terminated is base-driven and False.
    assert all(r[2] is False for r in rows)


def test_drawdown_stopper_not_before_threshold():
    # nav dips to 0.7 of a peak of 1.0; max_drawdown=0.5 must NOT fire (0.7 > 0.5).
    env = DrawdownStopper(_NavEnv([1.0, 0.7, 0.65]), max_drawdown=0.5)
    env.reset()
    a = _act(env)
    rows = [env.step(a) for _ in range(2)]
    assert all(not r[4].get("stopped_out", False) for r in rows)
    assert all(r[3] is False for r in rows)


def test_drawdown_stopper_peak_vs_initial_mode_differ():
    navs = [1.0, 2.0, 0.9, 0.85]  # peak=2.0; initial=1.0; trailing bar keeps step 3 non-terminal
    peak = DrawdownStopper(_NavEnv(navs), max_drawdown=0.5, mode="peak")
    peak.reset()
    pa = _act(peak)
    peak.step(pa); peak.step(pa)
    _o, _r, _t, p_trunc, p_info = peak.step(pa)
    assert p_trunc is True and p_info.get("stopped_out") is True  # 0.9 <= 0.5*2.0=1.0

    init = DrawdownStopper(_NavEnv(navs), max_drawdown=0.5, mode="initial")
    init.reset()
    ia = _act(init)
    init.step(ia); init.step(ia)
    _o, _r, _t, i_trunc, i_info = init.step(ia)
    assert i_trunc is False and not i_info.get("stopped_out")  # 0.9 > 0.5*1.0=0.5


def test_drawdown_stopper_terminated_stays_base_driven():
    # NAV goes negative -> base env terminates; stop-out may also flag, but terminated
    # must come from the base env, not be re-labelled by the wrapper.
    env = DrawdownStopper(_NavEnv([1.0, -0.1]), max_drawdown=0.5)
    env.reset()
    a = _act(env)
    env.step(a)
    _o, _r, terminated, truncated, _info = env.step(a)
    assert terminated is True


def test_drawdown_stopper_deterministic():
    def run():
        env = DrawdownStopper(_NavEnv([1.0, 1.3, 1.0, 0.6, 0.5, 0.4]), max_drawdown=0.4)
        env.reset()
        a = _act(env)
        return [bool(env.step(a)[4].get("stopped_out", False)) for _ in range(5)]

    assert run() == run()


# -- TurbulenceHalt ---------------------------------------------------------

def _turbulence_rollout(env, base, steps):
    a = _act(env)
    rows = []
    for _ in range(steps):
        out = env.step(a)
        rows.append((bool(out[4].get("turbulence_halt", False)), base.executed.copy()))
    return rows


def test_turbulence_halt_fires_on_spike_and_flattens_action():
    # Five tiny ~0.5% returns fill the window, then a huge jump. The halt fires on the step
    # AFTER the jump return enters the trailing window (point-in-time), with a flat action.
    navs = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 1000.0, 1001.0]
    base = _NavEnv(navs)
    env = TurbulenceHalt(base, window=5, threshold=3.0)
    env.reset()
    rows = _turbulence_rollout(env, base, 8)
    halts = [i for i, (h, _a) in enumerate(rows) if h]
    assert halts == [7]  # only the bar after the spike return is in-window
    # executed action is flat (zeros) on the halt step, pass-through everywhere else.
    assert np.all(rows[7][1] == 0.0)
    for i, (_h, executed) in enumerate(rows):
        if i != 7:
            assert np.allclose(executed, _act(env))


def test_turbulence_halt_passthrough_below_threshold():
    # A calm, low-vol path never trips the breaker; every action passes through unchanged.
    navs = [100.0 + 0.1 * i for i in range(12)]
    base = _NavEnv(navs)
    env = TurbulenceHalt(base, window=5, threshold=3.0)
    env.reset()
    rows = _turbulence_rollout(env, base, 11)
    assert all(not h for h, _a in rows)
    assert all(np.allclose(executed, _act(env)) for _h, executed in rows)


def test_turbulence_halt_deterministic():
    def run(seed):
        base = _SeededNavEnv(steps=60)
        env = TurbulenceHalt(base, window=10, threshold=1.5)
        env.reset(seed=seed)
        a = _act(env)
        return [bool(env.step(a)[4].get("turbulence_halt", False)) for _ in range(59)]

    assert run(7) == run(7)
    # a vol spike is injected mid-path, so at least one halt is expected.
    assert any(run(7))


def test_turbulence_halt_preserves_5_tuple():
    base = _NavEnv([100.0, 101.0, 102.0, 103.0])
    env = TurbulenceHalt(base, window=3, threshold=3.0)
    env.reset()
    out = env.step(_act(env))
    assert len(out) == 5
    _o, reward, terminated, truncated, info = out
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


# -- CrossSectionalDeleverage -----------------------------------------------

class _CrossSectionEnv(gym.Env):
    """Stub env replaying a scripted per-symbol close path as a Dict obs.

    ``path[0]`` is the reset observation; ``path[k]`` is the obs returned by the k-th
    ``step``. The action actually executed is recorded so an override can be observed.
    """

    def __init__(self, path) -> None:
        super().__init__()
        self._path = [np.asarray(c, dtype=np.float64).reshape(-1) for c in path]
        n = int(self._path[0].shape[0])
        self._n = n
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"closes": spaces.Box(low=0.0, high=np.inf, shape=(n,), dtype=np.float64)}
        )
        self._i = 0
        self.executed: np.ndarray | None = None

    def _obs(self):
        return {"closes": self._path[self._i].copy()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i = 0
        self.executed = None
        return self._obs(), {}

    def step(self, action):
        self.executed = np.asarray(action).copy()
        self._i += 1
        truncated = self._i >= len(self._path) - 1
        return self._obs(), 0.0, False, truncated, {"nav": 1.0}


def _falling_rising_path(n_falling: int, n_rising: int, bars: int):
    """A path where the first ``n_falling`` symbols decline monotonically (RSI -> 0) and the
    rest rise monotonically (RSI -> 100). Bar ``k`` is ``[100-k]*n_falling + [100+k]*n_rising``.
    """
    path = []
    for k in range(bars):
        row = [100.0 - k] * n_falling + [100.0 + k] * n_rising
        path.append(row)
    return path


def _five_full(env) -> np.ndarray:
    n = env.action_space.shape[0]
    return np.full((n,), 0.5, dtype=np.float32)


def test_deleverage_fires_at_threshold_and_flattens_subset():
    # 5 symbols, 3 falling / 2 rising: 3 oversold == min_oversold and 3/5 == 0.60 == fraction.
    # window=3 -> RSI needs 4 closes, so the first 3 steps are warmup (cannot fire).
    base = _CrossSectionEnv(_falling_rising_path(3, 2, bars=6))
    env = CrossSectionalDeleverage(
        base, rsi_window=3, oversold=30.0, fraction=0.60, min_oversold=3, scope="subset"
    )
    env.reset()
    a = _five_full(env)
    rows = []
    for _ in range(5):
        out = env.step(a)
        rows.append((bool(out[4].get("deleverage_veto", False)), base.executed.copy(), out[4]))
    vetoes = [i for i, (v, _e, _info) in enumerate(rows) if v]
    # warmup: steps 0..2 never fire; the veto first fires on step index 3 (the 4th step).
    assert vetoes and vetoes[0] == 3
    veto_info = rows[3][2]
    assert veto_info["deleverage_oversold"] == [0, 1, 2]
    assert veto_info["deleverage_scope"] == "subset"
    executed = rows[3][1]
    assert np.all(executed[:3] == 0.0)  # oversold subset forced flat
    assert np.allclose(executed[3:], 0.5)  # non-oversold names pass through
    # warmup steps pass the action through untouched.
    for i in (0, 1, 2):
        assert np.allclose(rows[i][1], 0.5)
        assert rows[i][0] is False


def test_deleverage_scope_all_flattens_every_symbol():
    # Same broadly-oversold scenario, but scope="all" forces the WHOLE book flat (not just
    # the oversold subset) on the firing bar.
    base = _CrossSectionEnv(_falling_rising_path(3, 2, bars=6))
    env = CrossSectionalDeleverage(
        base, rsi_window=3, oversold=30.0, fraction=0.60, min_oversold=3, scope="all"
    )
    env.reset()
    a = _five_full(env)
    fired = False
    for _ in range(5):
        out = env.step(a)
        if out[4].get("deleverage_veto"):
            fired = True
            assert np.all(base.executed == 0.0)  # every symbol flat, not just the subset
            assert out[4]["deleverage_scope"] == "all"
    assert fired


def test_deleverage_below_threshold_does_not_fire():
    # Only 2 of 5 symbols oversold: 2 < min_oversold=3 and 2/5=0.40 < 0.60. Never fires.
    base = _CrossSectionEnv(_falling_rising_path(2, 3, bars=6))
    env = CrossSectionalDeleverage(
        base, rsi_window=3, oversold=30.0, fraction=0.60, min_oversold=3, scope="subset"
    )
    env.reset()
    a = _five_full(env)
    for _ in range(5):
        out = env.step(a)
        assert not out[4].get("deleverage_veto", False)
        assert np.allclose(base.executed, 0.5)  # untouched agent action every step


def test_deleverage_fraction_guards_small_breadth():
    # 3 of 5 oversold clears min_oversold but a stricter fraction=0.8 (needs >=4) blocks it.
    base = _CrossSectionEnv(_falling_rising_path(3, 2, bars=6))
    env = CrossSectionalDeleverage(
        base, rsi_window=3, oversold=30.0, fraction=0.80, min_oversold=3
    )
    env.reset()
    a = _five_full(env)
    rows = [env.step(a) for _ in range(5)]
    assert all(not r[4].get("deleverage_veto", False) for r in rows)


def test_deleverage_leak_free_current_bar_does_not_affect_veto():
    # Two envs share bars 0..3 (the closes the step-4 decision reads) but the bar returned BY
    # step 4 differs wildly. The step-4 veto must be identical: the decision uses bars 0..3
    # only, never the bar it is about to step into.
    shared = _falling_rising_path(3, 2, bars=6)  # bars 0..5
    peek_bait = [list(r) for r in shared]
    peek_bait[4] = [200.0, 200.0, 200.0, 1.0, 1.0]  # if peeked, oversold set would flip to {3,4}
    a = None
    results = []
    for path in (shared, peek_bait):
        base = _CrossSectionEnv(path)
        env = CrossSectionalDeleverage(
            base, rsi_window=3, oversold=30.0, fraction=0.60, min_oversold=3
        )
        env.reset()
        a = _five_full(env)
        step4 = None
        for _ in range(4):
            out = env.step(a)
            step4 = (bool(out[4].get("deleverage_veto", False)),
                     out[4].get("deleverage_oversold"),
                     base.executed.copy())
        results.append(step4)
    (v0, over0, ex0), (v1, over1, ex1) = results
    assert v0 is True and v1 is True
    assert over0 == over1 == [0, 1, 2]
    assert np.allclose(ex0, ex1)


def test_deleverage_deterministic():
    def run():
        base = _CrossSectionEnv(_falling_rising_path(3, 2, bars=6))
        env = CrossSectionalDeleverage(
            base, rsi_window=3, oversold=30.0, fraction=0.60, min_oversold=3
        )
        env.reset()
        a = _five_full(env)
        return [bool(env.step(a)[4].get("deleverage_veto", False)) for _ in range(5)]

    assert run() == run()


def test_deleverage_preserves_5_tuple():
    base = _CrossSectionEnv(_falling_rising_path(3, 2, bars=4))
    env = CrossSectionalDeleverage(base, rsi_window=3, min_oversold=3)
    env.reset()
    out = env.step(_five_full(env))
    assert len(out) == 5
    _o, reward, terminated, truncated, info = out
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_deleverage_rejects_bad_params():
    base = _CrossSectionEnv(_falling_rising_path(3, 2, bars=4))
    with pytest.raises(ValueError):
        CrossSectionalDeleverage(base, rsi_window=1)
    with pytest.raises(ValueError):
        CrossSectionalDeleverage(base, oversold=0.0)
    with pytest.raises(ValueError):
        CrossSectionalDeleverage(base, fraction=0.0)
    with pytest.raises(ValueError):
        CrossSectionalDeleverage(base, min_oversold=0)
    with pytest.raises(ValueError):
        CrossSectionalDeleverage(base, scope="half")


# -- live binding (skipped when the native module is absent) -----------------

openoutcry = pytest.importorskip("openoutcry")


def _live_env(seed=0):
    return openoutcry.OpenOutcryEnv(n_symbols=3, n_days=50, seed=seed)


def test_live_drawdown_stopper_preserves_5_tuple():
    env = DrawdownStopper(_live_env(0), max_drawdown=0.2)
    env.reset()
    n = env.action_space.shape[0]
    out = env.step(np.full((n,), 1.0 / n, dtype=np.float32))
    assert len(out) == 5


def test_live_turbulence_halt_preserves_5_tuple():
    env = TurbulenceHalt(_live_env(0), window=10, threshold=3.0)
    env.reset()
    n = env.action_space.shape[0]
    out = env.step(np.full((n,), 1.0 / n, dtype=np.float32))
    assert len(out) == 5


def test_live_cross_sectional_deleverage_preserves_5_tuple_and_info():
    env = CrossSectionalDeleverage(_live_env(0), rsi_window=5, min_oversold=2)
    env.reset()
    n = env.action_space.shape[0]
    a = np.full((n,), 1.0 / n, dtype=np.float32)
    for _ in range(20):
        out = env.step(a)
        assert len(out) == 5
        info = out[4]
        if info.get("deleverage_veto"):
            assert isinstance(info["deleverage_oversold"], list)
            assert info["deleverage_scope"] in ("subset", "all")
        if out[2] or out[3]:
            break
