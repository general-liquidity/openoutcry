"""A PettingZoo multi-agent **competition** environment over OpenOutcry.

:class:`MultiAgentOpenOutcryEnv` is a :class:`pettingzoo.ParallelEnv` that runs a
*batched tournament*: ``n_agents`` agents each trade their **own** copy of the
**same frozen scenario** (every per-agent :class:`~openoutcry.gym.OpenOutcryEnv` is
seeded to one shared scenario seed, so they face an identical leak-free price path),
and are ranked by their realized SharpeBench ``deflated_sharpe`` at episode end.

**This is competition, not a shared market.** The agents do **not** interact: there is
no cross-agent price impact, no endogenous market — one agent's orders never move the
prices another agent sees. The contest is "who trades the same tape best," scored by the
real SharpeBench kernel. An endogenous-impact market (where agents' aggregate flow moves
the book) is a separate, future flagship build; this module deliberately changes nothing
in the simulator.

**Parallel, not AEC.** Order submission in a market is *simultaneous* — every agent acts
on the same bar without seeing the others' action first — so the native API is
:class:`pettingzoo.ParallelEnv`. The Farama-recommended pattern is to author the Parallel
env and derive the turn-based AEC view from it; :func:`make_aec_env` does exactly that via
:func:`pettingzoo.utils.conversions.parallel_to_aec`.

``pettingzoo`` is an **optional** dependency (guarded like ``verifiers`` / ``mcp``):
``import openoutcry`` and ``import openoutcry.pettingzoo_env`` both work without it; only
*constructing* :class:`MultiAgentOpenOutcryEnv` (or calling :func:`make_aec_env`) raises a
clear ``RuntimeError`` when it is absent.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from .openoutcry_py import score_run  # the real SharpeBench scorer (pyo3)
from .gym import OpenOutcryEnv

try:  # pragma: no cover - exercised only when pettingzoo is installed
    from pettingzoo import ParallelEnv
    from pettingzoo.utils.conversions import parallel_to_aec

    _HAS_PETTINGZOO = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    ParallelEnv = object  # type: ignore[assignment,misc]
    parallel_to_aec = None  # type: ignore[assignment]
    _HAS_PETTINGZOO = False


def _agent_ids(n_agents: int) -> list[str]:
    """The canonical, sorted-stable tournament roster (``agent_0`` … ``agent_{n-1}``)."""
    return [f"agent_{i}" for i in range(int(n_agents))]


def _deflated_sharpe(returns: list[float], n_trials: int) -> float:
    """The real SharpeBench deflated Sharpe for a return series, or 0.0 for <2 bars."""
    if len(returns) < 2:
        return 0.0
    return float(json.loads(score_run(returns, n_trials)).get("deflated_sharpe", 0.0))


class MultiAgentOpenOutcryEnv(ParallelEnv):
    """A batched-competition :class:`pettingzoo.ParallelEnv` over OpenOutcry.

    ``n_agents`` agents each own a private :class:`~openoutcry.gym.OpenOutcryEnv` seeded to
    the **same** scenario seed, so they trade an identical, leak-free, point-in-time path.
    Actions are submitted simultaneously (Parallel API), fanned to each agent's env, and
    the per-agent results are returned as ``{agent: value}`` dicts. There is **no**
    cross-agent price impact — see the module docstring.

    Determinism: agents are always iterated in sorted roster order and the result dicts are
    built in that canonical order, so the end-of-episode ``deflated_sharpe`` ranking (and
    its tie-breaks) is reproducible.
    """

    metadata = {"render_modes": [], "name": "openoutcry_competition_v0", "is_parallelizable": True}

    def __init__(
        self,
        n_agents: int = 2,
        *,
        n_symbols: int = 4,
        n_days: int = 120,
        seed: int = 0,
        distribution_mode: str = "calm",
        max_weight: float = 1.0,
        allow_short: bool = True,
        n_trials: int = 0,
        env_kwargs: Optional[dict] = None,
    ) -> None:
        if not _HAS_PETTINGZOO:
            raise RuntimeError(
                "pettingzoo is not installed. Install 'pettingzoo' to use "
                "MultiAgentOpenOutcryEnv; the rest of the openoutcry package works "
                "without it."
            )
        if int(n_agents) < 1:
            raise ValueError("n_agents must be >= 1")

        self._n_agents = int(n_agents)
        self._n_symbols = int(n_symbols)
        self._n_days = int(n_days)
        self._scenario_seed = int(seed)
        self._distribution_mode = str(distribution_mode)
        self._max_weight = float(max_weight)
        self._allow_short = bool(allow_short)
        self._n_trials = int(n_trials)
        self._kwargs: dict[str, Any] = dict(env_kwargs or {})

        # The full tournament roster (every agent that *could* ever act). `self.agents`
        # is the live roster — agents that blow up are dropped from it but stay in
        # `possible_agents`, per the PettingZoo contract.
        self.possible_agents: list[str] = _agent_ids(self._n_agents)
        self.agents: list[str] = list(self.possible_agents)

        self._envs: dict[str, OpenOutcryEnv] = {}
        self._returns: dict[str, list[float]] = {}
        self._symbols: list[str] = []

        # Build one env per agent at the shared scenario seed to discover the symbol axis
        # and fix the per-agent / Box spaces. All agents share identical spaces.
        self._build_envs()
        probe = self._envs[self.possible_agents[0]]
        self._symbols = probe.symbols
        self._single_obs_space = probe.observation_space
        self._single_act_space = probe.action_space

    # -- internal helpers --------------------------------------------------

    def _build_envs(self) -> None:
        """(Re)construct every agent's env, all seeded to the same frozen scenario."""
        self._envs = {
            agent: OpenOutcryEnv(
                n_symbols=self._n_symbols,
                n_days=self._n_days,
                seed=self._scenario_seed,
                max_weight=self._max_weight,
                allow_short=self._allow_short,
                env_kwargs=self._kwargs,
            )
            for agent in self.possible_agents
        }

    def _ranking(self) -> list[dict[str, Any]]:
        """Cross-agent leaderboard by deflated Sharpe, descending. Ties break on the
        canonical (sorted) agent id so the order is reproducible."""
        scored = [
            {"agent": agent, "deflated_sharpe": _deflated_sharpe(self._returns[agent], self._n_trials)}
            for agent in self.possible_agents
        ]
        scored.sort(key=lambda r: (-r["deflated_sharpe"], r["agent"]))
        for rank, row in enumerate(scored):
            row["rank"] = rank
        return scored

    # -- PettingZoo (Parallel) API -----------------------------------------

    def observation_space(self, agent: str):
        """The per-agent observation space (mirrors the single env's ``Dict`` space)."""
        return self._single_obs_space

    def action_space(self, agent: str):
        """The per-agent action space (the single env's target-weight ``Box``)."""
        return self._single_act_space

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def scenario_seed(self) -> int:
        return self._scenario_seed

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict[str, dict]]:
        """Reset every agent onto the **same** frozen scenario.

        Passing an int ``seed`` reselects the shared scenario (all agents rebuild on it),
        so every agent still trades an identical leak-free path. Returns the standard
        Parallel ``(observations, infos)`` pair, both keyed by live agent in canonical
        order.
        """
        if seed is not None:
            self._scenario_seed = int(seed)
            self._build_envs()

        self.agents = list(self.possible_agents)
        self._returns = {agent: [] for agent in self.possible_agents}

        observations: dict[str, Any] = {}
        infos: dict[str, dict] = {}
        for agent in self.agents:  # sorted roster order
            obs, info = self._envs[agent].reset(seed=self._scenario_seed)
            observations[agent] = obs
            infos[agent] = dict(info)
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict]]:
        """Fan each agent's action to its own env, advance one bar, and return per-agent
        dicts. An agent that terminates (blows up) is removed from ``self.agents``; at the
        episode end the cross-agent ``deflated_sharpe`` ranking is placed in ``infos``.
        """
        observations: dict[str, Any] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}

        for agent in self.agents:  # canonical order
            obs, reward, terminated, truncated, info = self._envs[agent].step(actions[agent])
            self._returns[agent].append(float(reward))
            observations[agent] = obs
            rewards[agent] = float(reward)
            terminations[agent] = bool(terminated)
            truncations[agent] = bool(truncated)
            infos[agent] = dict(info)

        # Episode is over for an agent that blew up or ran out of bars. When every live
        # agent is done, attach the cross-agent leaderboard to each agent's info.
        episode_over = all(
            terminations[agent] or truncations[agent] for agent in self.agents
        ) if self.agents else True
        if episode_over:
            ranking = self._ranking()
            by_agent = {row["agent"]: row for row in ranking}
            for agent in self.agents:
                infos[agent]["ranking"] = ranking
                infos[agent]["rank"] = by_agent[agent]["rank"]
                infos[agent]["deflated_sharpe"] = by_agent[agent]["deflated_sharpe"]

        # Drop terminated agents from the live roster (kept in possible_agents). Iterate
        # the canonical order so the surviving roster stays sorted.
        self.agents = [
            agent
            for agent in self.agents
            if not (terminations[agent] or truncations[agent])
        ]
        return observations, rewards, terminations, truncations, infos

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self):  # pragma: no cover
        for env in self._envs.values():
            try:
                env.close()
            except Exception:  # noqa: BLE001 - close is best-effort
                pass


def make_aec_env(*args: Any, **kwargs: Any):
    """The turn-based **AEC** view of the competition, derived from the Parallel env via
    the Farama-recommended :func:`pettingzoo.utils.conversions.parallel_to_aec` wrapper.

    Accepts the same arguments as :class:`MultiAgentOpenOutcryEnv`. Raises a clear
    ``RuntimeError`` when ``pettingzoo`` is not installed.
    """
    if not _HAS_PETTINGZOO:
        raise RuntimeError(
            "pettingzoo is not installed. Install 'pettingzoo' to build the AEC view; "
            "the rest of the openoutcry package works without it."
        )
    return parallel_to_aec(MultiAgentOpenOutcryEnv(*args, **kwargs))


__all__ = [
    "MultiAgentOpenOutcryEnv",
    "make_aec_env",
]
