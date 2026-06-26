"""Tests for the Avellaneda-Stoikov market-making env and its analytical optimum.

Covers the gymnasium contract (construct / reset / step / terminal liquidation),
determinism-given-seed, the inventory hard cap and squared running penalty, and the
math-validation fixture: the closed-form A-S optimal policy beats a naive fixed-spread
maker in mean reward over seeded episodes, with ``mm_regret(optimal) == 0`` and
``mm_regret(naive) > 0``.
"""

import math

import numpy as np
import pytest

from openoutcry.market_making import (
    MMParams,
    MarketMakingEnv,
    analytically_optimal_policy,
    fixed_spread_policy,
    mm_regret,
)


def _rollout(env, policy, seed):
    obs, info = env.reset(seed=seed)
    rewards, invs, infos = [], [], []
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        rewards.append(reward)
        invs.append(int(info["inventory"]))
        infos.append(info)
        if terminated or truncated:
            return rewards, invs, infos


# -- gymnasium contract -----------------------------------------------------


def test_construct_reset_step_shapes():
    env = MarketMakingEnv(MMParams(n_steps=20))
    obs, info = env.reset(seed=0)
    assert set(obs.keys()) == {"inventory", "mid", "time_remaining", "cash"}
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_episode_terminates_with_forced_liquidation():
    p = MMParams(n_steps=30)
    env = MarketMakingEnv(p)
    rewards, invs, infos = _rollout(env, fixed_spread_policy(0.05), seed=1)
    assert len(rewards) == p.n_steps  # terminates exactly at the horizon
    # terminal step force-liquidates: inventory ends flat and a liquidation is recorded.
    assert infos[-1]["inventory"] == 0
    assert any(info["liquidated"] != 0.0 for info in infos)


def test_time_remaining_counts_down_to_zero():
    p = MMParams(n_steps=10)
    env = MarketMakingEnv(p)
    obs, _ = env.reset(seed=0)
    assert obs["time_remaining"][0] == pytest.approx(p.horizon)
    last = obs
    while True:
        obs, _, term, trunc, _ = env.step(np.array([0.5, 0.5], dtype=np.float32))
        last = obs
        if term or trunc:
            break
    assert last["time_remaining"][0] == pytest.approx(0.0)


# -- determinism ------------------------------------------------------------


def test_same_seed_identical_trajectory():
    p = MMParams(n_steps=50)
    pol = analytically_optimal_policy(p)
    a = _rollout(MarketMakingEnv(p), pol, seed=7)
    b = _rollout(MarketMakingEnv(p), pol, seed=7)
    assert a[0] == b[0]  # identical reward sequence
    assert a[1] == b[1]  # identical inventory sequence


def test_distinct_seeds_distinct_trajectories():
    p = MMParams(n_steps=50)
    pol = analytically_optimal_policy(p)
    a = _rollout(MarketMakingEnv(p), pol, seed=1)
    b = _rollout(MarketMakingEnv(p), pol, seed=2)
    assert a[0] != b[0]


# -- inventory cap + squared penalty ----------------------------------------


def test_inventory_cap_respected():
    p = MMParams(n_steps=80, inventory_cap=5)
    env = MarketMakingEnv(p)
    # An extremely tight spread fills aggressively and would blow past the cap if unbounded.
    _, invs, _ = _rollout(env, fixed_spread_policy(0.0), seed=3)
    assert max(abs(q) for q in invs) <= p.inventory_cap


def test_squared_inventory_penalty_applied():
    # With no fills (a prohibitively wide spread) and a forced inventory, the per-step
    # reward is exactly the -phi*q**2 running penalty plus the mark-to-mid drift.
    p = MMParams(n_steps=4, phi=0.5, sigma=0.0, max_depth=1e9)
    env = MarketMakingEnv(p)
    env.reset(seed=0)
    env._q = 3  # forced inventory; sigma=0 so mid is frozen -> value change is 0
    _, reward, _, _, _ = env.step(np.array([1e9, 1e9], dtype=np.float32))
    assert reward == pytest.approx(-p.phi * 3**2)


# -- analytical optimum (math validation) -----------------------------------


def test_optimal_beats_naive_in_mean_reward():
    p = MMParams()
    optimal = analytically_optimal_policy(p)
    naive = fixed_spread_policy(0.3)
    env = MarketMakingEnv(p)
    opt_mean = np.mean([sum(_rollout(env, optimal, s)[0]) for s in range(24)])
    naive_mean = np.mean([sum(_rollout(env, naive, s)[0]) for s in range(24)])
    assert opt_mean > naive_mean


def test_regret_optimal_is_zero_and_naive_positive():
    p = MMParams()
    optimal = analytically_optimal_policy(p)
    assert mm_regret(optimal, params=p, n_episodes=24) == pytest.approx(0.0, abs=1e-9)
    assert mm_regret(fixed_spread_policy(0.3), params=p, n_episodes=24) > 0.0


def test_optimal_depths_skew_with_inventory():
    p = MMParams()
    pol = analytically_optimal_policy(p)
    tau = p.horizon
    base = {"mid": np.array([100.0]), "time_remaining": np.array([tau]), "cash": np.array([0.0])}
    flat = pol({**base, "inventory": np.array([0.0])})
    long = pol({**base, "inventory": np.array([10.0])})
    # Long inventory: lean to sell -> tighter ask (closer to mid), wider bid.
    assert flat[0] == pytest.approx(flat[1])  # symmetric when flat
    assert long[1] < flat[1]  # ask pulled in
    assert long[0] > flat[0]  # bid pushed out


def test_half_spread_widens_with_time_to_go():
    p = MMParams()
    pol = analytically_optimal_policy(p)
    base = {"mid": np.array([100.0]), "inventory": np.array([0.0]), "cash": np.array([0.0])}
    early = pol({**base, "time_remaining": np.array([p.horizon])})
    late = pol({**base, "time_remaining": np.array([p.dt])})
    assert early[0] > late[0]  # more time-to-go -> wider quotes
    # late half-spread floors at the order-book term (1/gamma)*ln(1+gamma/kappa).
    floor = (1.0 / p.gamma) * math.log1p(p.gamma / p.kappa)
    assert late[0] == pytest.approx(0.5 * p.gamma * p.sigma**2 * p.dt + floor)
