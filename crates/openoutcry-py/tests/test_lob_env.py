"""Tests for the M3 limit-order-book env + the native PyOrderBook engine boundary."""

import json

import numpy as np
import pytest


def _book():
    from openoutcry.openoutcry_py import PyOrderBook

    return PyOrderBook(tick_size=0.01, levels=5)


def test_orderbook_resting_and_ladder():
    b = _book()
    b.reset_book()
    r = json.loads(
        b.step_book(
            json.dumps(
                [
                    {"agent": 0, "kind": "limit", "side": "buy", "price_tick": 99, "qty": 10},
                    {"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 10},
                ]
            )
        )
    )
    assert r["ladder"]["bids"] == [[99, 10]]
    assert r["ladder"]["asks"] == [[101, 10]]
    assert r["ladder"]["mid"] == 100.0
    assert -1.0 <= r["ladder"]["queue_imbalance"] <= 1.0


def test_orderbook_market_order_crosses():
    b = _book()
    b.reset_book()
    b.step_book(json.dumps([{"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 10}]))
    r = json.loads(b.step_book(json.dumps([{"agent": 2, "kind": "market", "side": "buy", "qty": 4}])))
    assert len(r["fills"]) == 1
    f = r["fills"][0]
    assert f["price_tick"] == 101 and f["qty"] == 4 and f["taker_side"] == "buy" and f["maker_agent"] == 1
    # 4 of the 10 resting were consumed.
    assert r["ladder"]["asks"] == [[101, 6]]


def test_orderbook_canonical_order_deterministic():
    def run(orders):
        b = _book()
        b.reset_book()
        return b.step_book(json.dumps(orders))

    o = [
        {"agent": 0, "kind": "limit", "side": "buy", "price_tick": 99, "qty": 5},
        {"agent": 1, "kind": "limit", "side": "sell", "price_tick": 101, "qty": 5},
    ]
    assert run(o) == run(list(reversed(o)))  # reorder input -> identical (canonical sort)


def test_orderbook_rejects_bad_order():
    b = _book()
    with pytest.raises(Exception):
        b.step_book(json.dumps([{"agent": 0, "kind": "limit", "side": "buy"}]))  # missing price/qty


pettingzoo = pytest.importorskip("pettingzoo")


def test_lob_env_constructs_and_steps():
    from openoutcry.lob_env import LOBMarketEnv

    env = LOBMarketEnv(n_agents=2, n_steps=10, seed=1)
    obs, infos = env.reset(seed=1)
    assert set(obs) == {"agent_0", "agent_1"}
    assert obs["agent_0"].shape == env.observation_space("agent_0").shape
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert set(rewards) == {"agent_0", "agent_1"}
    assert all(np.isfinite(v) for v in rewards.values())


def test_lob_env_deterministic():
    from openoutcry.lob_env import LOBMarketEnv, symmetric_quote_policy

    def rollout(seed):
        env = LOBMarketEnv(n_agents=2, n_steps=12, seed=seed)
        env.reset(seed=seed)
        out = []
        done = False
        while not done:
            acts = {a: symmetric_quote_policy(offset=3) for a in env.agents}
            _o, r, _t, tr, _i = env.step(acts)
            out.append(tuple(round(x, 6) for x in r.values()))
            done = any(tr.values())
        return out

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


def test_lob_env_parallel_api():
    from pettingzoo.test import parallel_api_test
    from openoutcry.lob_env import LOBMarketEnv

    parallel_api_test(LOBMarketEnv(n_agents=2, n_steps=20, seed=2), num_cycles=10)
