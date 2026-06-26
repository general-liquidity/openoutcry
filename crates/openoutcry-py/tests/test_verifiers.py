"""The PrimeIntellect ``verifiers`` environment — verified end-to-end.

Skipped when ``verifiers`` is not installed; the rest of the package works without it.
"""

import asyncio
import json

import numpy as np
import pytest

pytest.importorskip("verifiers")
import verifiers as vf

import openoutcry
from openoutcry.dataset import (
    EVAL_SEED_BASE,
    build_scenario_dataset,
    seed_ranges_disjoint,
)
from openoutcry.decision_parser import parse_decision
from openoutcry.mandate import (
    Mandate,
    mandate_breach,
    mandate_from_dict,
    mandate_text,
    sample_mandate,
    validate_mandate,
)
from openoutcry.verifiers_env import (
    OpenOutcryVerifiersEnv,
    build_rubric,
    deflated_sharpe_reward,
    load_environment,
    mandate_reward,
    pass_k_reward,
    process_check_reward,
    realized_return_reward,
)

RETS = [0.01, -0.005, 0.012, 0.003, -0.001, 0.008, -0.002, 0.006]


def test_score_run_is_the_real_kernel():
    """``score_run`` returns the real SharpeBench ``CompositeScore``, not a stub."""
    s = json.loads(openoutcry.score_run(RETS, 0))
    for gate in ("deflated_sharpe", "psr", "passed_k", "process_ok", "composite"):
        assert gate in s
    assert isinstance(s["deflated_sharpe"], float)


def test_rewards_are_calibrated_to_the_kernel():
    """The deflated-Sharpe reward equals the kernel's deflated_sharpe exactly."""
    expected = json.loads(openoutcry.score_run(RETS, 0))["deflated_sharpe"]
    assert deflated_sharpe_reward(state={"returns": RETS, "events": []}) == pytest.approx(expected)
    assert pass_k_reward(state={"returns": RETS}) in (0.0, 1.0)
    assert process_check_reward(state={"events": [{"event": "manipulative_order"}]}) < 1.0
    assert process_check_reward(state={"events": []}) == 1.0
    assert deflated_sharpe_reward(state={"returns": RETS}, n_trials=1000) <= deflated_sharpe_reward(
        state={"returns": RETS}, n_trials=0
    )


def test_rubric_and_environment_construct():
    assert build_rubric() is not None
    env = load_environment()
    assert env is not None
    assert isinstance(env, vf.MultiTurnEnv)
    assert not isinstance(env, vf.SingleTurnEnv)


# -- the dense GRPO reward ---------------------------------------------------

def test_dense_reward_is_bounded_and_varies():
    """tanh-squashed realized return stays in [-1, 1] and differs across series."""
    empty = realized_return_reward(state={"returns": []})
    assert empty == 0.0
    up = realized_return_reward(state={"returns": [0.05, 0.05, 0.05]})
    down = realized_return_reward(state={"returns": [-0.05, -0.05, -0.05]})
    assert -1.0 <= down < up <= 1.0
    big = realized_return_reward(state={"returns": [10.0] * 20})
    assert -1.0 <= big <= 1.0


# -- the decision parser -----------------------------------------------------

def test_parse_decision_clamps_and_handles_malformed():
    symbols = ["AAA", "BBB", "CCC"]
    w = parse_decision('<action>{"weights": {"AAA": 0.5, "BBB": -0.3}}</action>', symbols)
    assert w.tolist() == [0.5, -0.3, 0.0]
    # out-of-range weights clamp to [-1, 1]; unknown symbols ignored.
    w2 = parse_decision('<action>{"weights": {"AAA": 5.0, "ZZZ": 9.0}}</action>', symbols)
    assert w2.tolist() == [1.0, 0.0, 0.0]
    # explicit flat and malformed both yield all-zero (hold).
    assert parse_decision('<action>{"flat": true}</action>', symbols).tolist() == [0, 0, 0]
    assert parse_decision("not xml at all", symbols).tolist() == [0, 0, 0]
    assert parse_decision('<action>{not json}</action>', symbols).tolist() == [0, 0, 0]
    # bare JSON (no XML wrapper) is accepted too.
    assert parse_decision('{"weights": {"CCC": 0.4}}', symbols).tolist() == [0, 0, 0.4]


# -- the multi-row scenario dataset -----------------------------------------

def test_build_scenario_dataset_multirow_and_disjoint():
    train = build_scenario_dataset(n_windows=8, n_symbols=3, n_days=40, mode="train")
    eval_ds = build_scenario_dataset(n_windows=4, n_symbols=3, n_days=40, mode="eval")
    assert len(train) == 8 and len(eval_ds) == 4
    # answer round-trips the seed through a string.
    assert int(train[0]["answer"]) == train[0]["info"]["seed"]
    # train/eval seed ranges are disjoint (leak-free at experiment level).
    assert seed_ranges_disjoint(train, eval_ds)
    assert all(r["seed"] < EVAL_SEED_BASE for r in train["info"])
    assert all(r["seed"] >= EVAL_SEED_BASE for r in eval_ds["info"])


# -- per-scenario mandates (the MiniGrid Fetch pattern) ----------------------

def test_sample_mandate_is_deterministic_and_varies():
    """Same seed -> identical mandate; the style varies across the seed space."""
    a = sample_mandate(42, n_symbols=4)
    b = sample_mandate(42, n_symbols=4)
    assert a == b
    assert validate_mandate(a) and a.style in (
        "long_only",
        "market_neutral",
        "momentum",
        "unconstrained",
    )
    styles = {sample_mandate(s, n_symbols=4).style for s in range(64)}
    assert len(styles) > 1
    # round-trips through plain JSON (trace/replay).
    assert mandate_from_dict(a.to_dict()) == a
    assert mandate_text(a) == a.text and isinstance(a.text, str) and a.text
    # a long-only market never draws a short-requiring (market_neutral) mandate.
    assert all(
        sample_mandate(s, n_symbols=4, allow_short=False).style != "market_neutral"
        for s in range(64)
    )


def test_mandate_breach_clean_vs_breached():
    """0 for a clean long-only series; >0 for a short / a drawdown-cap breach."""
    long_only = Mandate(style="long_only")
    clean_events = [{"event": "target_weights", "weights": [0.5, 0.3]}] * 4
    assert mandate_breach(long_only, [0.01, 0.0, 0.01, 0.0], clean_events) == 0.0
    # a short under long_only breaches.
    short_events = [{"event": "target_weights", "weights": [-0.5, 0.2]}] * 4
    assert mandate_breach(long_only, [0.01, 0.0], short_events) > 0.0
    # a drawdown-cap breach (a -30% bar against a 10% cap) penalizes even when long-only.
    capped = Mandate(style="long_only", max_drawdown=0.10)
    assert mandate_breach(capped, [-0.30, 0.0], clean_events) > 0.0
    # market-neutral: a balanced long/short book is clean, a one-sided book breaches.
    neutral = Mandate(style="market_neutral")
    assert mandate_breach(neutral, [], [{"event": "target_weights", "weights": [0.5, -0.5]}]) == 0.0
    assert mandate_breach(neutral, [], [{"event": "target_weights", "weights": [0.5, 0.5]}]) > 0.0
    # breach is bounded in [0, 1].
    assert 0.0 <= mandate_breach(long_only, [-0.9, -0.9], short_events) <= 1.0


def test_mandate_reward_is_bounded_and_objective_conditioned():
    """1 - breach, bounded; a no-mandate state is vacuously satisfied."""
    assert mandate_reward(state={"returns": [], "events": []}) == 1.0  # no mandate -> 1
    long_only = Mandate(style="long_only").to_dict()
    clean = mandate_reward(
        state={"mandate": long_only, "returns": [0.01], "events": [{"event": "target_weights", "weights": [0.4]}]}
    )
    shorted = mandate_reward(
        state={"mandate": long_only, "returns": [0.01], "events": [{"event": "target_weights", "weights": [-0.4]}]}
    )
    assert clean == 1.0
    assert 0.0 <= shorted < clean <= 1.0


def test_dataset_rows_carry_mandate_and_question_includes_it():
    ds = build_scenario_dataset(n_windows=4, n_symbols=3, n_days=30, mode="train")
    for row in ds:
        m = row["info"]["mandate"]
        assert validate_mandate(m)
        # the per-scenario objective is woven into the prompt.
        assert mandate_text(mandate_from_dict(m)) in row["question"]
        assert "Mandate:" in row["question"]


def test_rubric_includes_the_mandate_reward():
    rubric = build_rubric()
    # the mandate is a weighted reward (in `funcs`), not a zero-weight metric.
    names = {getattr(f, "__name__", str(f)) for f in rubric.funcs}
    assert "mandate_reward" in names


def test_rollout_threads_mandate_into_state():
    env = OpenOutcryVerifiersEnv(
        dataset=build_scenario_dataset(n_windows=1, n_symbols=2, n_days=20, mode="train"),
        rubric=build_rubric(),
        max_turns=6,
        max_episode_bars=3,
    )
    mandate = Mandate(style="long_only", max_drawdown=0.10).to_dict()
    state = {"info": {"seed": 5, "n_symbols": 2, "n_days": 20, "mandate": mandate}, "answer": "5"}
    _run(env.setup_state(state))
    assert validate_mandate(state["mandate"])
    assert state["mandate"]["style"] == "long_only"
    # a state without a row mandate still gets the seed-derived one.
    state2 = {"info": {"seed": 9, "n_symbols": 2, "n_days": 20}, "answer": "9"}
    _run(env.setup_state(state2))
    assert validate_mandate(state2["mandate"])


# -- the actual bug fix: a rollout steps a market and populates state --------

def _run(coro):
    return asyncio.run(coro)


def _assistant(text):
    return vf.AssistantMessage(role="assistant", content=text)


def test_rollout_steps_market_and_populates_state():
    """Driving env_response directly fills state['returns']/['events'] from a REAL market."""
    env = OpenOutcryVerifiersEnv(
        dataset=build_scenario_dataset(n_windows=2, n_symbols=3, n_days=30, mode="train"),
        rubric=build_rubric(),
        max_turns=8,
        max_episode_bars=4,
    )
    state = {"info": {"seed": 7, "n_symbols": 3, "n_days": 30}, "answer": "7"}
    _run(env.setup_state(state))
    assert state["returns"] == [] and state["events"] == []
    symbols = state["_oo_symbols"]
    assert len(symbols) == 3

    decision = f'<reasoning>go</reasoning><action>{{"weights": {{"{symbols[0]}": 0.5}}}}</action>'
    for _ in range(4):
        _run(env.env_response([_assistant(decision)], state))

    assert len(state["returns"]) == 4
    assert all(isinstance(r, float) for r in state["returns"])
    assert len(state["events"]) >= 1
    # episode-bar cap fired; the terminate stop-condition now reports done.
    assert state["_oo_done"] is True
    assert _run(env.episode_terminated(state)) is True

    # the real kernel now scores non-empty data (the bug: it used to be all-zeros).
    composite = json.loads(openoutcry.score_run(state["returns"], 0))
    assert "deflated_sharpe" in composite
    assert realized_return_reward(state=state) != 0.0


def test_malformed_decision_does_not_crash_episode():
    """A bad completion is treated as flat/hold — the rollout keeps stepping."""
    env = OpenOutcryVerifiersEnv(
        dataset=build_scenario_dataset(n_windows=1, n_symbols=2, n_days=20, mode="train"),
        rubric=build_rubric(),
        max_turns=6,
        max_episode_bars=3,
    )
    state = {"info": {"seed": 3, "n_symbols": 2, "n_days": 20}, "answer": "3"}
    _run(env.setup_state(state))
    for _ in range(3):
        _run(env.env_response([_assistant("garbage, no action tag")], state))
    assert len(state["returns"]) == 3
