<!-- prettier-ignore -->
<div align="center">

# Training on OpenOutcry

### From `vf-eval` baseline to a GRPO run, on a leak-free trading floor

</div>

---

OpenOutcry ships a real, trainable [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers)
environment: `openoutcry.load_environment()` returns an `OpenOutcryVerifiersEnv`
(`vf.MultiTurnEnv`) that drives the point-in-time market one bar per turn and scores the
realized return series with the SharpeBench kernel. This guide covers running it under
`vf-eval` and training on it with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

For a copy-pasteable end-to-end config, see [`examples/prime-rl/`](../examples/prime-rl/).

## The env-package layout

The env is a standard installed Python package; the agent contract and Rust engine live
behind a pyo3 wheel built by [maturin](https://www.maturin.rs).

```
crates/openoutcry-py/
  pyproject.toml                     # package metadata + [tool.verifiers.eval] + tags
  python/openoutcry/
    __init__.py
    verifiers_env.py                 # load_environment(), the rubric, the multi-turn rollout
    dataset.py                       # build_scenario_dataset() - the multi-row taskset
    gym.py                           # the underlying gymnasium.Env stepped per turn
    openoutcry_py...pyd              # the compiled Rust kernel (SharpeBench scorer)
```

`load_environment(**args)` is the entry point `verifiers` and prime-rl call. It accepts
`n_symbols`, `n_days`, `n_windows`, `max_episode_bars`, `max_turns`, `max_weight`, and
`allow_short`. `args.n_windows` sets the dataset length, which is the number of GRPO
tasks (`num_tasks`) - it must be `> 1` so within-group reward variance has something to
vary over.

## Install and discover

```bash
pip install -e "crates/openoutcry-py[verifiers]"   # editable, with the verifiers extra
```

Once published to the [PrimeIntellect](https://app.primeintellect.ai) Environments-Hub,
the env is installable by org-qualified id:

```bash
prime env install general-liquidity/openoutcry
```

The `[project].tags` list in `pyproject.toml` feeds Hub discoverability (the `prime` CLI
reads it at push time); the `[tool.verifiers.eval]` table sets the default
`num_examples` / `rollouts_per_example` for a bare `vf-eval openoutcry`.

## Baseline with `vf-eval`

```bash
vf-eval openoutcry \
  -m Qwen/Qwen3-1.7B -n 20 \
  -a '{"n_windows": 20, "n_symbols": 4, "n_days": 120, "max_episode_bars": 16}'
```

The rubric's dense `realized_return_reward` is the GRPO objective; the real
`deflated_sharpe_reward` is a secondary objective; `pass_k_reward`,
`process_check_reward`, and `format_reward` are zero-weight diagnostics.

## v1 taskset and the subprocess runtime

Under the verifiers v1 contract the env is a **taskset** (`taskset = { id = "openoutcry" }`)
composed with a harness and a runtime. With no container image declared, it runs on the
**subprocess runtime**: prime-rl spawns a local env-server subprocess that imports the
installed package and serves rollouts over the worker pool. That is the right default
for a pure-Python + pyo3 env with no external services. Drive it from prime-rl with
either the v1 `taskset = { id = "openoutcry" }` form or the legacy `id = "openoutcry"`
form shown in [`examples/prime-rl/rl.toml`](../examples/prime-rl/rl.toml).

## Leak-freedom: point-in-time is ours, the split is yours

Two distinct guarantees:

1. **Within a scenario - structural, ours.** The data layer has no API to read a future
   bar; the environment owns the time cursor. An agent cannot peek, by construction.
2. **Across train and eval - experimental, yours.** A trustworthy training run must
   evaluate on a **strictly held-out seed band**. `build_scenario_dataset` supports this
   directly: `mode="train"` draws seeds from base `0`, `mode="eval"` from
   `EVAL_SEED_BASE = 1_000_000`, and the two ranges are asserted disjoint
   (`seed_ranges_disjoint`).

> **Caveat (openoutcry 0.1.0).** `load_environment` currently hardcodes `mode="train"`
> and does not forward `mode`/`seed_start` to `build_scenario_dataset`. Passing
> `mode = "eval"` through a prime-rl eval `args` block is a **silent no-op** (the key is
> absorbed by the verifiers `Environment(**kwargs)` catch-all), so the eval set reuses
> the train seed band. Until `load_environment` forwards `mode`, treat any in-config
> eval as **not** held-out, or build the eval `Dataset` yourself with
> `build_scenario_dataset(..., mode="eval")` and pass it into `load_environment(dataset=...)`
> from Python. The eval block in the example config encodes the intended split and is
> flagged accordingly.
