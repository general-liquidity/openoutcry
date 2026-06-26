<!-- prettier-ignore -->
<div align="center">

# OpenOutcry x prime-rl

### Train a small model on the leak-free trading floor in one command

</div>

---

[`rl.toml`](rl.toml) is a committed, illustrative GRPO config that runs the OpenOutcry
[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment under
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). It mirrors prime-rl's own
`examples/reverse_text` and `configs/gsm8k`: a renderer-mapped small model, a single
installed env taskset driven by an integer task index, and a held-out eval block for
`pass@k`.

> It is **illustrative**. The config was authored against the prime-rl schema and is
> valid TOML, but no trainer has been run against it here. Treat the structure as the
> contract and the numbers (steps, lr, batch size, GPUs) as starting points to tune.

## 1. Install the env

The OpenOutcry env is an installed Python package that exposes `load_environment`. The
`verifiers` extra pulls in the RLVR runtime; the wheel is built by maturin.

```bash
# From the repo root - editable install of the Python package with the verifiers extra.
pip install -e "crates/openoutcry-py[verifiers]"
# (or, once published to the PrimeIntellect Hub:)
# prime env install general-liquidity/openoutcry
```

`vf-eval` and prime-rl resolve the env by the package name, **`openoutcry`** - the same
id used in [`rl.toml`](rl.toml)'s `[[orchestrator.train.env]]`.

## 2. Baseline the env before you train

Sanity-check the env and get a pre-training score with `vf-eval`. `-n` is the number of
scenarios; `-a` is the JSON forwarded verbatim to `load_environment`:

```bash
vf-eval openoutcry \
  -m Qwen/Qwen3-1.7B \
  -n 20 \
  -a '{"n_windows": 20, "n_symbols": 4, "n_days": 120, "max_episode_bars": 16, "allow_short": true}'
```

You should see the rubric's `realized_return_reward` (the dense GRPO objective) and the
real `deflated_sharpe_reward`, plus the zero-weight `pass_k_reward` / `process_check_reward`
/ `format_reward` diagnostics.

## 3. Train

```bash
uv run rl @ rl.toml
```

This launches the prime-rl orchestrator + trainer + inference server, samples
`group_size = 16` rollouts per scenario, and runs GRPO. Checkpoints land in the
trainer's output dir (the `[ckpt]` block; checkpoint at end of training by default).

For a fast smoke test, shrink the run: `max_steps = 5`, `args.n_windows = 16`
(`num_tasks` must stay > 1 for GRPO), and a smaller `batch_size`.

## Values you must adjust

| Key | Why |
|:--|:--|
| `[model].name` | Any model you can serve. If it is not in prime-rl's `MODEL_RENDERER_MAP`, keep an explicit `[orchestrator.renderer]`. |
| `max_steps`, `[trainer.optim].lr`, `[orchestrator].batch_size` | Hardware- and budget-dependent. `batch_size` must be a multiple of `group_size`. |
| `[deployment]` (not set) | Add `num_train_gpus` / `num_infer_gpus` for your topology (see prime-rl's `examples/wordle`). |
| `args.n_windows` | The dataset length, i.e. `num_tasks`. A few hundred for a real run, >= 16 for a smoke. |

## Caveat: the held-out eval split is yours to enforce

The env is point-in-time and leak-free **within** a scenario (the data layer has no API
to read a future bar). Leak-freedom **across** train and eval - a strictly disjoint
seed band - is the operator's responsibility.

`build_scenario_dataset` already supports it (`mode="train"` draws from seed base `0`,
`mode="eval"` from `EVAL_SEED_BASE = 1_000_000`, asserted disjoint). **But** as of
`openoutcry` 0.1.0, `load_environment` hardcodes `mode="train"` and does not forward
`mode`/`seed_start`. The `mode = "eval"` key in the `[[orchestrator.eval.env]]` `args`
is therefore a **silent no-op today** - the eval set currently reuses the train seed
band. Until `load_environment` forwards `mode` to `build_scenario_dataset`, do not
report this eval as held-out. See [`docs/training.md`](../../docs/training.md).
