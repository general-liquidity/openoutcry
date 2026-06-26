"""OpenOutcry — a leak-free, point-in-time Gym for trading agents.

This package bundles the native pyo3 binding (``openoutcry.openoutcry_py``) with a
gymnasium-compatible wrapper (:class:`OpenOutcryEnv`) and an AUTHORED-UNVERIFIED
PrimeIntellect ``verifiers`` environment (:mod:`openoutcry.verifiers_env`).

The native binding exchanges the language-agnostic wire JSON at its boundary:
``TradingEnv.reset()`` returns an observation JSON string and ``TradingEnv.step()``
takes a decision JSON string. The pure-Python layers parse/build that JSON.
"""

from .openoutcry_py import TradingEnv
from .gym import OpenOutcryEnv

__all__ = ["TradingEnv", "OpenOutcryEnv"]
__version__ = "0.0.6"
