"""Multi-row, point-in-time scenario datasets for the ``verifiers`` env.

Each row is one leak-free scenario: a distinct ``seed`` drives a distinct synthetic
market window. ``num_tasks > 1`` is a hard requirement for GRPO — a single-row dataset
gives the within-group reward variance nothing to vary over and training aborts.

Per-scenario config lives in the reserved ``info`` field (a dict), **not** in ad-hoc
top-level columns: top-level columns collide with reward-func argument names and would
be silently fed to the rubric. ``answer`` round-trips the seed through a string
(``str(seed)`` → ``int(answer)``) so the rollout can reconstruct the exact scenario.

Train and eval draw from **disjoint** seed ranges (leak-free at the experiment level):
train seeds start at 0, eval seeds at :data:`EVAL_SEED_BASE`. :func:`build_scenario_dataset`
asserts a train run never crosses into the eval range.
"""

from __future__ import annotations

from typing import Any, Optional

from .mandate import mandate_text, sample_mandate

EVAL_SEED_BASE = 1_000_000


def _seed_for(mode: str, seed_start: int, i: int) -> int:
    base = EVAL_SEED_BASE if mode == "eval" else 0
    return base + int(seed_start) + i


def _initial_question(
    n_symbols: int, n_days: int, mode: str, allow_short: bool, mandate_text_str: str
) -> str:
    side = "long or short" if allow_short else "long-only"
    return (
        f"You are trading a leak-free, point-in-time OpenOutcry market: {n_symbols} "
        f"symbols over a {n_days}-bar window ({side}, target weights in [-1, 1]).\n"
        f"Mandate: {mandate_text_str}\n"
        "Each turn you receive the latest bar (closes, positions, cash) and choose new "
        "target weights to maximize the run's deflated Sharpe while satisfying your "
        "mandate.\n"
        "Reply with two XML fields: <reasoning>...</reasoning> and an <action> carrying "
        'decision JSON, e.g. <action>{"weights": {"SYM00": 0.5, "SYM01": -0.3}}</action> '
        'or <action>{"flat": true}</action> to hold flat. Unlisted symbols default to 0.'
    )


def build_scenario_dataset(
    n_windows: int,
    n_symbols: int = 4,
    n_days: int = 120,
    seed_start: int = 0,
    mode: str = "train",
    *,
    regime: Optional[str] = None,
    allow_short: bool = True,
):
    """A ``datasets.Dataset`` of ``n_windows`` point-in-time scenarios.

    One row per scenario: ``question`` (initial instruction, including the scenario's
    sampled mandate), ``answer`` (``str(seed)``), ``info`` (``{"seed", "n_symbols",
    "n_days", "mode", "mandate", "regime"?}``). The mandate is sampled deterministically
    from the row seed (leak-free) and carried as a plain-JSON dict. ``mode`` selects the
    train/eval seed range; the two ranges are disjoint by construction.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    if mode not in ("train", "eval"):
        raise ValueError("mode must be 'train' or 'eval'")
    from datasets import Dataset

    questions: list[str] = []
    answers: list[str] = []
    infos: list[dict[str, Any]] = []
    for i in range(n_windows):
        seed = _seed_for(mode, seed_start, i)
        if mode == "train":
            assert seed < EVAL_SEED_BASE, (
                f"train seed {seed} crosses into the eval range >= {EVAL_SEED_BASE}; "
                "shrink n_windows/seed_start to keep train and eval disjoint"
            )
        mandate = sample_mandate(seed, n_symbols=int(n_symbols), allow_short=allow_short)
        questions.append(
            _initial_question(n_symbols, n_days, mode, allow_short, mandate_text(mandate))
        )
        answers.append(str(seed))
        info: dict[str, Any] = {
            "seed": seed,
            "n_symbols": int(n_symbols),
            "n_days": int(n_days),
            "mode": mode,
            "mandate": mandate.to_dict(),
        }
        if regime is not None:
            info["regime"] = regime
        infos.append(info)

    return Dataset.from_dict(
        {"question": questions, "answer": answers, "info": infos}
    )


def seed_ranges_disjoint(train_dataset, eval_dataset) -> bool:
    """True iff the two datasets share no scenario seed (experiment-level leak check)."""
    train_seeds = {int(r["seed"]) for r in train_dataset["info"]}
    eval_seeds = {int(r["seed"]) for r in eval_dataset["info"]}
    return train_seeds.isdisjoint(eval_seeds)


__all__ = ["build_scenario_dataset", "seed_ranges_disjoint", "EVAL_SEED_BASE"]
