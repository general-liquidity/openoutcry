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
]
__version__ = "0.1.0"
