/**
 * `@general-liquidity/openoutcry` — the trading-agent environment (a leak-free,
 * point-in-time market simulator + reference baselines + recompute-to-verify) as a
 * typed JS API over the *identical* Rust engine that powers the SharpeBench harness
 * (compiled to WebAssembly).
 *
 * The harness sends a {@link MarketObservation}, the agent returns a {@link Decision},
 * repeat — and look-ahead is structurally impossible. A run's returns are recomputed
 * from raw decisions ({@link replayRun}), never trusted from the agent's word.
 */
import * as kernel from "../pkg/openoutcry.js";

import type {
  BaselineConfig,
  CostModel,
  Dataset,
  Regime,
  Run,
  RunTrajectory,
  StressScenario,
  SyntheticParams,
  WalkForwardParams,
  Window,
} from "./types.js";

export * from "./types.js";

/** Parse a kernel JSON string, surfacing the kernel's `{error}` as a thrown Error. */
function parse<T>(json: string): T {
  const value = JSON.parse(json) as unknown;
  if (value && typeof value === "object" && "error" in value) {
    throw new Error(String((value as { error: unknown }).error));
  }
  return value as T;
}

/** Empty/absent config → `""`, which the kernel reads as "use defaults". */
function optJson(value: object | undefined): string {
  if (!value || Object.keys(value).length === 0) return "";
  return JSON.stringify(value);
}

/**
 * Run a named in-process baseline (`buy_and_hold` | `hold` | `momentum` | `random`)
 * over a dataset for a window + seed + cost model, returning the {@link Run} — its
 * per-period returns, decision trace, and per-step confidences/outcomes.
 */
export function runBaseline(config: BaselineConfig): Run {
  return parse(kernel.run_baseline(JSON.stringify(config)));
}

/**
 * Replay a captured {@link RunTrajectory}'s raw decisions through the identical
 * engine to regenerate its {@link Run} — the tamper-evidence path. The result is
 * byte-identical to the originally captured run iff `dataset` and `costs` match what
 * it was captured against.
 */
export function replayRun(
  dataset: Dataset,
  trajectory: RunTrajectory,
  costs?: CostModel,
): Run {
  return parse(
    kernel.replay_run(
      JSON.stringify(dataset),
      JSON.stringify(trajectory),
      optJson(costs),
    ),
  );
}

/** Build a deterministic synthetic {@link Dataset} from `{n_symbols, n_days, seed}`. */
export function datasetSynthetic(params?: SyntheticParams): Dataset {
  return parse(kernel.dataset_synthetic(optJson(params)));
}

/** The named adversarial stress suite (flash-crash, whipsaw, …) for a seed. */
export function stressSuite(seed = 0): StressScenario[] {
  return parse(kernel.stress_suite(JSON.stringify({ seed })));
}

/** Generate disjoint walk-forward out-of-sample {@link Window}s. */
export function walkForward(params: WalkForwardParams): Window[] {
  return parse(kernel.walk_forward(JSON.stringify(params)));
}

/** Tag a window's coarse market regime (bull / bear / chop). */
export function tagRegime(dataset: Dataset, window: Window): Regime {
  const out = parse<{ regime: Regime }>(
    kernel.tag_regime(JSON.stringify({ dataset, window })),
  );
  return out.regime;
}
