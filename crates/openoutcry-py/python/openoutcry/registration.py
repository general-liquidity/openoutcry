"""Gymnasium registration for the OpenOutcry env matrix (Farama plugin convention).

:func:`register_envs` adds a versioned, namespaced ID matrix to the global gymnasium
registry so ``gymnasium.make("OpenOutcry/Hard-v1")`` and
``gymnasium.make_vec("OpenOutcry/Hard-v1", num_envs=8)`` resolve to
:class:`~openoutcry.gym.OpenOutcryEnv` / :class:`~openoutcry.vector.OpenOutcryVectorEnv`.

The matrix is the cross-product of the three scenario tiers
(``calm | hard | extreme`` — the ``distribution_mode`` contract) with the train and
eval seed bands. The eval variants pin the **disjoint** eval seed band
(:data:`~openoutcry.dataset.EVAL_SEED_BASE`) so an agent trained on a tier can be
graded out-of-sample on the same tier without leakage.

**``-v1`` is a frozen contract.** The semantics behind a versioned ID — the fill model,
the fee/cost schedule, the point-in-time leakage guard, and the tier's distribution
(volatility, gaps, shock frequency) — MUST NOT silently change. Any change to those
semantics is a new env version (``-v2``), never an in-place edit of ``-v1``. Reported
scores are only comparable across runs because the ID pins the rules.

Following the Farama plugin convention, the integrator calls :func:`register_envs` once
at package import time (from ``__init__.py``); registration is idempotent so a second
call — or gymnasium having already auto-registered the IDs — is a no-op, never an error.
"""

from __future__ import annotations

import gymnasium
from gymnasium.envs.registration import register, registry

from .dataset import EVAL_SEED_BASE

#: Import path of the scalar env (gymnasium ``entry_point``).
ENTRY_POINT = "openoutcry.gym:OpenOutcryEnv"
#: Import path of the batched env (gymnasium ``vector_entry_point``).
VECTOR_ENTRY_POINT = "openoutcry.vector:OpenOutcryVectorEnv"

#: ``distribution_mode`` value per tier (the constructor contract both envs share).
TIERS = ("calm", "hard", "extreme")

#: Panel shape baked into every registered ID (part of the ``-v1`` frozen contract).
N_SYMBOLS = 4
N_DAYS = 120

#: Hard episode cap = the panel window. The engine truncates on its own when it runs
#: out of bars (which happens at or before ``N_DAYS``), so gymnasium's ``TimeLimit`` is
#: a backstop, not the primary terminator — set to the window so it never cuts an
#: episode short.
MAX_EPISODE_STEPS = N_DAYS

#: Namespace for the ID matrix (``OpenOutcry/<Tier>[-Eval]-v1``).
NAMESPACE = "OpenOutcry"
VERSION = 1


def _env_id(tier: str, *, eval_band: bool) -> str:
    name = tier.capitalize()
    if eval_band:
        name = f"{name}-Eval"
    return f"{NAMESPACE}/{name}-v{VERSION}"


def _kwargs(tier: str, *, eval_band: bool) -> dict:
    kwargs = {
        "distribution_mode": tier,
        "n_symbols": N_SYMBOLS,
        "n_days": N_DAYS,
    }
    if eval_band:
        # ``mode="eval"`` is the shared seed-band selector understood by BOTH the scalar
        # and the vector env: it offsets the scenario seed(s) into the disjoint eval band
        # at ``EVAL_SEED_BASE`` (mirrors ``dataset.build_scenario_dataset(mode=...)``).
        # A single shared kwarg is required because ``gymnasium.register`` applies one
        # ``kwargs`` dict to both ``entry_point`` and ``vector_entry_point``.
        kwargs["mode"] = "eval"
    return kwargs


def env_ids() -> list[str]:
    """Every ID :func:`register_envs` registers, in a stable order (train tiers then
    eval tiers)."""
    return [_env_id(t, eval_band=False) for t in TIERS] + [
        _env_id(t, eval_band=True) for t in TIERS
    ]


def register_envs() -> list[str]:
    """Register the full ``OpenOutcry/*-v1`` ID matrix; return the IDs.

    Idempotent: an ID already present in the gymnasium registry is skipped, and the
    duplicate-registration error gymnasium raises (the race where it auto-registered via
    the plugin entry point) is swallowed — so calling this twice never raises, exactly
    like a Farama-bundled env package.
    """
    for tier in TIERS:
        for eval_band in (False, True):
            env_id = _env_id(tier, eval_band=eval_band)
            if env_id in registry:
                continue
            try:
                register(
                    id=env_id,
                    entry_point=ENTRY_POINT,
                    vector_entry_point=VECTOR_ENTRY_POINT,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    kwargs=_kwargs(tier, eval_band=eval_band),
                )
            except gymnasium.error.Error:
                # Already registered (e.g. gymnasium auto-loaded the plugin first).
                pass
    return env_ids()


__all__ = [
    "register_envs",
    "env_ids",
    "ENTRY_POINT",
    "VECTOR_ENTRY_POINT",
    "TIERS",
    "N_SYMBOLS",
    "N_DAYS",
    "MAX_EPISODE_STEPS",
    "NAMESPACE",
    "EVAL_SEED_BASE",
]
