"""PrimeIntellect ``verifiers`` environment for OpenOutcry — a real, multi-turn,
multi-scenario trading env.

:class:`OpenOutcryVerifiersEnv` is a :class:`vf.MultiTurnEnv` that drives
:class:`~openoutcry.gym.OpenOutcryEnv` one bar per turn. On the first turn for a rollout
it instantiates the env from the scenario seed encoded in the dataset row, ``reset()``s,
and seeds ``state['returns'] = []`` / ``state['events'] = []``. Each turn parses the
model's ``<action>`` decision, maps it to a target-weight action, ``step()``s the market,
and **appends the realized bar return to ``state['returns']`` and any ``info['events']``
to ``state['events']``** — so the SharpeBench-calibrated rewards score REAL data instead
of the empty arrays a ``SingleTurnEnv`` (which never steps a market) leaves behind.

Reward shaping is GRPO-safe: a dense, bounded ``tanh``-squashed realized-return reward
gives gradient on short/sparse episodes and varies across decision paths, with the real
deflated Sharpe as a secondary objective and ``pass^k`` / process discipline kept as
zero-weight diagnostic metrics.

Verified against ``verifiers`` 0.1.14: ``MultiTurnEnv.env_response(messages, state) ->
Messages`` (mutating ``state`` in place), ``is_completed`` is ``@final`` (terminate via a
``@vf.stop`` handler), and ``vf.Rubric(funcs=, weights=)`` filters reward-func args by
signature.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

import numpy as np

from .openoutcry_py import score_run  # the real SharpeBench scorer (pyo3)
from .gym import OpenOutcryEnv
from .decision_parser import build_parser, format_reward, parse_decision

try:  # pragma: no cover - exercised only when verifiers is installed
    import verifiers as vf

    _HAS_VERIFIERS = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    vf = None  # type: ignore[assignment]
    _HAS_VERIFIERS = False


# ---------------------------------------------------------------------------
# Reward functions — singular `state` param so the rubric routes them per-rollout
# (a plural `states`/`infos` param would be treated as a group func), `**kwargs` so
# the signature-filtering rubric can pass whatever it has.
# ---------------------------------------------------------------------------

def _returns_from_state(state: Optional[dict]) -> list[float]:
    """Per-bar realized returns recorded by the rollout, if any."""
    return [float(r) for r in (state or {}).get("returns", []) or []]


def _composite(returns: list[float], n_trials: int) -> dict:
    """The real SharpeBench ``CompositeScore`` for a return series (Rust kernel)."""
    if len(returns) < 2:
        return {}
    return json.loads(score_run(returns, n_trials))


def realized_return_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """Dense, bounded episodic reward: ``tanh`` of the summed realized bar returns.

    This is the GRPO workhorse — it is non-zero on short/sparse episodes and differs
    across decision paths (the deflated Sharpe collapses to 0 for <2 bars and is flat
    for many distinct-but-similar paths), so the within-group variance never vanishes.
    """
    rets = _returns_from_state(state)
    if not rets:
        return 0.0
    return float(np.tanh(np.sum(rets)))


def deflated_sharpe_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    n_trials: int = 0,
    **kwargs: Any,
) -> float:
    """The **real** deflated Sharpe (SharpeBench kernel), deflated for ``n_trials`` of
    declared in-sample search — the metric the benchmark ranks on."""
    return float(_composite(_returns_from_state(state), n_trials).get("deflated_sharpe", 0.0))


def pass_k_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    n_trials: int = 0,
    **kwargs: Any,
) -> float:
    """1.0 iff the run clears the per-run PSR bar (the kernel's ``passed_k`` gate)."""
    return 1.0 if _composite(_returns_from_state(state), n_trials).get("passed_k", False) else 0.0


def process_check_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """Penalize block-severity events surfaced in the env's per-bar ``info`` (the
    sim-exploitation guard, e.g. a manipulative order). 1.0 = clean."""
    events: Sequence[dict] = (state or {}).get("events", []) if state else []
    bad = sum(1 for e in events if "manipulative" in str(e.get("event", "")).lower())
    return 1.0 if bad == 0 else max(0.0, 1.0 - 0.25 * bad)


def build_rubric(*, parser: Any = None):
    """The reward bundle. Dense realized-return is the primary GRPO objective; the real
    deflated Sharpe is a secondary objective; ``pass^k`` / process discipline / format
    are zero-weight diagnostic metrics (gates, not gradient). Raises if ``verifiers`` is
    unavailable."""
    if not _HAS_VERIFIERS:
        raise RuntimeError("verifiers is not installed; cannot build a Rubric")
    rubric = vf.Rubric(
        funcs=[realized_return_reward, deflated_sharpe_reward],
        weights=[1.0, 0.5],
        parser=parser,
    )
    rubric.add_metric(pass_k_reward)
    rubric.add_metric(process_check_reward)
    rubric.add_metric(format_reward)
    return rubric


# ---------------------------------------------------------------------------
# The multi-turn rollout
# ---------------------------------------------------------------------------

def _last_assistant_text(messages: Any) -> str:
    """The most recent assistant message's text content (decision), or ``""``."""
    for m in reversed(list(messages or [])):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "assistant":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                (p.get("text") if isinstance(p, dict) else getattr(p, "text", ""))
                for p in content
            ]
            return "".join(t for t in parts if isinstance(t, str))
        return ""
    return ""


def render_observation(obs: dict, symbols: Sequence[str], *, final: bool = False) -> str:
    """A compact textual bar observation for the model's next decision."""
    closes = obs.get("closes")
    positions = obs.get("positions")
    cash = obs.get("cash")
    rows = []
    for i, s in enumerate(symbols):
        c = float(closes[i]) if closes is not None else 0.0
        p = float(positions[i]) if positions is not None else 0.0
        rows.append(f"{s}: close={c:.4f} pos={p:.4f}")
    cash_v = float(cash[0]) if cash is not None and len(cash) else 0.0
    head = "Final bar — episode complete." if final else "Market update."
    tail = "" if final else " Respond with <reasoning> and an <action> of target weights."
    return f"{head} cash={cash_v:.2f}. " + "; ".join(rows) + "." + tail


if _HAS_VERIFIERS:

    class OpenOutcryVerifiersEnv(vf.MultiTurnEnv):
        """Per-bar multi-turn rollout over :class:`~openoutcry.gym.OpenOutcryEnv`."""

        def __init__(
            self,
            *,
            n_symbols: int = 4,
            n_days: int = 120,
            max_episode_bars: int = 64,
            max_weight: float = 1.0,
            allow_short: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self._n_symbols = int(n_symbols)
            self._n_days = int(n_days)
            self._max_episode_bars = int(max_episode_bars)
            self._max_weight = float(max_weight)
            self._allow_short = bool(allow_short)

        # -- scenario plumbing --------------------------------------------

        def _scenario_seed(self, state: dict) -> int:
            info = state.get("info")
            if isinstance(info, dict) and "seed" in info:
                return int(info["seed"])
            try:
                return int(str(state.get("answer")))
            except (TypeError, ValueError):
                return 0

        def _ensure_env(self, state: dict) -> None:
            """Instantiate + reset the market on first use; seed the recorded arrays."""
            if state.get("_oo_env") is not None:
                return
            info = state.get("info") if isinstance(state.get("info"), dict) else {}
            seed = self._scenario_seed(state)
            env = OpenOutcryEnv(
                n_symbols=int(info.get("n_symbols", self._n_symbols)),
                n_days=int(info.get("n_days", self._n_days)),
                seed=seed,
                max_weight=self._max_weight,
                allow_short=self._allow_short,
            )
            obs, _ = env.reset(seed=seed)
            state["_oo_env"] = env
            state["_oo_symbols"] = env.symbols
            state["_oo_done"] = False
            state["returns"] = []
            state["events"] = []
            state["_oo_last_obs"] = obs

        def _close_env(self, state: dict) -> None:
            env = state.pop("_oo_env", None)
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001 - close is best-effort
                    pass

        # -- MultiTurnEnv contract ----------------------------------------

        async def setup_state(self, state) -> None:
            self._ensure_env(state)

        @vf.stop
        async def episode_terminated(self, state, **kwargs) -> bool:
            return bool(state.get("_oo_done", False))

        async def env_response(self, messages, state, **kwargs):
            self._ensure_env(state)
            env = state["_oo_env"]
            symbols = state["_oo_symbols"]
            action = parse_decision(_last_assistant_text(messages), symbols)
            obs, reward, terminated, truncated, info = env.step(action)
            state["returns"].append(float(reward))
            for e in info.get("events", []) or []:
                state["events"].append(e)
            state["_oo_last_obs"] = obs
            done = (
                bool(terminated or truncated)
                or len(state["returns"]) >= self._max_episode_bars
            )
            if done:
                state["_oo_done"] = True
                self._close_env(state)
            return [
                vf.UserMessage(
                    role="user", content=render_observation(obs, symbols, final=done)
                )
            ]

else:  # pragma: no cover - placeholder so the symbol exists without verifiers

    class OpenOutcryVerifiersEnv:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "verifiers is not installed; OpenOutcryVerifiersEnv is unavailable"
            )


def load_environment(dataset: Any = None, **kwargs: Any):
    """``verifiers`` entry point: the multi-turn, multi-scenario OpenOutcry env.

    Builds a default multi-row train dataset when none is supplied (never the 1-row stub —
    GRPO needs ``num_tasks > 1``). Accepts ``n_symbols``, ``n_days``, ``n_windows``,
    ``max_episode_bars``, ``max_turns``, ``max_weight``, ``allow_short``, and ``mode`` /
    ``seed_start`` — ``mode="eval"`` draws from the disjoint ``EVAL_SEED_BASE`` band so an
    in-config eval set is genuinely held out from training.
    """
    if not _HAS_VERIFIERS:
        raise RuntimeError(
            "verifiers is not installed. Install PrimeIntellect 'verifiers' to load "
            "this environment; the rest of the openoutcry package works without it."
        )
    n_symbols = int(kwargs.pop("n_symbols", 4))
    n_days = int(kwargs.pop("n_days", 120))
    n_windows = int(kwargs.pop("n_windows", 16))
    max_episode_bars = int(kwargs.pop("max_episode_bars", 64))
    max_weight = float(kwargs.pop("max_weight", 1.0))
    allow_short = bool(kwargs.pop("allow_short", True))
    mode = str(kwargs.pop("mode", "train"))
    seed_start = int(kwargs.pop("seed_start", 0))
    max_turns = kwargs.pop("max_turns", None)
    if max_turns is None:
        # +2: one turn for the initial decision, one for the final-bar message.
        max_turns = max_episode_bars + 2

    if dataset is None:
        from .dataset import build_scenario_dataset

        dataset = build_scenario_dataset(
            n_windows=n_windows,
            n_symbols=n_symbols,
            n_days=n_days,
            seed_start=seed_start,
            mode=mode,
            allow_short=allow_short,
        )

    parser = build_parser()
    return OpenOutcryVerifiersEnv(
        dataset=dataset,
        rubric=build_rubric(parser=parser),
        parser=parser,
        max_turns=max_turns,
        n_symbols=n_symbols,
        n_days=n_days,
        max_episode_bars=max_episode_bars,
        max_weight=max_weight,
        allow_short=allow_short,
        **kwargs,
    )


__all__ = [
    "realized_return_reward",
    "deflated_sharpe_reward",
    "pass_k_reward",
    "process_check_reward",
    "build_rubric",
    "load_environment",
    "render_observation",
    "OpenOutcryVerifiersEnv",
]
