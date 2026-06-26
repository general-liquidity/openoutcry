"""OpenOutcry — a leak-free, point-in-time Gym for trading agents.

This package bundles the native pyo3 binding (``openoutcry.openoutcry_py``) with a
gymnasium-compatible wrapper (:class:`OpenOutcryEnv`) and a PrimeIntellect ``verifiers``
environment (:mod:`openoutcry.verifiers_env`) whose rubric is scored by the real
SharpeBench kernel via :func:`score_run`.

The native binding exchanges the language-agnostic wire JSON at its boundary:
``TradingEnv.reset()`` returns an observation JSON string and ``TradingEnv.step()``
takes a decision JSON string. The pure-Python layers parse/build that JSON.
"""

from .openoutcry_py import TradingEnv, VecTradingEnv, score_run, validate_decision_json
from .gym import OpenOutcryEnv
from .vector import OpenOutcryVectorEnv
from .check_env import check_env, check_determinism_across_constructors
from .wrappers import (
    TimeLimit,
    CausalNormalizeObservation,
    CausalNormalizeReward,
    FrameStack,
    RecordEpisodeStatistics,
)
from .generalization import train_test_seeds, evaluate_seeds, generalization_gap
from .verifiers_env import OpenOutcryVerifiersEnv, load_environment, build_rubric
from .dataset import build_scenario_dataset, seed_ranges_disjoint
from .decision_parser import parse_decision
from .lookahead_guard import LookaheadGuard, LookaheadViolation, guarded, wrap_policy
from .trace import SCHEMA_VERSION, RolloutTraceWriter, load_trace, trace_to_returns
from .metrics import RunMetrics, cost_adjusted_score
from .wrappers_vector import (
    VectorCausalNormalizeObservation,
    VectorRecordEpisodeStatistics,
)
from .spaces import flatten_obs, unflatten_obs, flat_dim, FlattenObservation
from .execution_noise import ExecutionNoiseWrapper
from .mandate import (
    Mandate,
    sample_mandate,
    mandate_text,
    mandate_breach,
    mandate_from_dict,
    validate_mandate,
)
from .verifiers_env import mandate_reward
from .baselines import run_baselines, leaderboard_markdown
from .minari_export import to_minari
from .pettingzoo_env import MultiAgentOpenOutcryEnv, make_aec_env
from .market_env import EndogenousMarketEnv
from .checkpoint import CheckpointableEnv, CheckpointState
from .functional import OpenOutcryFuncEnv
from .registration import register_envs

# Farama plugin convention: register the versioned env IDs at import time (idempotent).
register_envs()

__all__ = [
    "TradingEnv",
    "VecTradingEnv",
    "score_run",
    "validate_decision_json",
    "OpenOutcryEnv",
    "OpenOutcryVectorEnv",
    "LookaheadGuard",
    "LookaheadViolation",
    "guarded",
    "wrap_policy",
    "SCHEMA_VERSION",
    "RolloutTraceWriter",
    "load_trace",
    "trace_to_returns",
    "RunMetrics",
    "cost_adjusted_score",
    "check_env",
    "check_determinism_across_constructors",
    "TimeLimit",
    "CausalNormalizeObservation",
    "CausalNormalizeReward",
    "FrameStack",
    "RecordEpisodeStatistics",
    "train_test_seeds",
    "evaluate_seeds",
    "generalization_gap",
    "OpenOutcryVerifiersEnv",
    "load_environment",
    "build_rubric",
    "build_scenario_dataset",
    "seed_ranges_disjoint",
    "parse_decision",
    "VectorCausalNormalizeObservation",
    "VectorRecordEpisodeStatistics",
    "flatten_obs",
    "unflatten_obs",
    "flat_dim",
    "FlattenObservation",
    "ExecutionNoiseWrapper",
    "Mandate",
    "sample_mandate",
    "mandate_text",
    "mandate_breach",
    "mandate_from_dict",
    "validate_mandate",
    "mandate_reward",
    "run_baselines",
    "leaderboard_markdown",
    "to_minari",
    "MultiAgentOpenOutcryEnv",
    "make_aec_env",
    "EndogenousMarketEnv",
    "CheckpointableEnv",
    "CheckpointState",
    "OpenOutcryFuncEnv",
    "register_envs",
]
__version__ = "0.2.0"
