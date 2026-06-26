# openoutcry (Python)

Python distribution of **OpenOutcry** — a leak-free, point-in-time *Gym for trading
agents*. A pyo3 binding over the Rust environment plus a `gymnasium`-compatible
wrapper and a PrimeIntellect `verifiers` environment.

```python
from openoutcry import OpenOutcryEnv

env = OpenOutcryEnv(n_symbols=4, n_days=120, seed=7)
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()      # target-weight vector
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

The native binding (`openoutcry.openoutcry_py.TradingEnv`) exchanges the
language-agnostic wire JSON at its boundary: `reset() -> str` and
`step(decision_json) -> (obs_json, reward, done, info_json)`.

Build from source with [maturin](https://www.maturin.rs): `python -m maturin develop`.
