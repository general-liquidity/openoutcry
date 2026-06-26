// Typed views of the OpenOutcry engine's JSON shapes. The wire contract (what an
// agent sees / returns) is typed precisely; engine outputs (Run/trace) carry the
// headline fields plus an index signature so they stay forward-compatible as the
// engine adds reported axes.

// --- The agent ⇄ harness wire contract (OpenOutcry Agent Interface v1.0) ---------

/** Discrete action label; sizing is carried by `target_weight`. */
export type Action = "buy" | "sell" | "hold" | "close";

/** A single per-instrument instruction. */
export interface Order {
  symbol: string;
  action: Action;
  /** Target portfolio weight in [0, 1] (signed for shorts). */
  target_weight: number;
  /** Stated conviction in [0, 1]; scored for calibration. Defaults to 0.5. */
  confidence?: number;
  /** Optional one-line rationale for this order, captured into the run trace. */
  rationale?: string;
}

/** What the agent returns at one decision point. */
export interface Decision {
  orders: Order[];
  /** Free-text rationale, captured into the trajectory for auditability. */
  reasoning?: string;
}

/** Point-in-time data for one instrument (only data at/before `date`). */
export interface SymbolSnapshot {
  symbol: string;
  /** Trailing closes up to and including `date` (oldest first). */
  close_history: number[];
  fundamentals?: Record<string, number>;
  news?: string[];
}

/** The agent's current holding in one instrument. */
export interface PositionState {
  symbol: string;
  shares: number;
  avg_price: number;
}

/** What the agent sees at one decision point. */
export interface MarketObservation {
  date: string;
  cash: number;
  symbols: SymbolSnapshot[];
  portfolio: PositionState[];
}

// --- The scored engine output ----------------------------------------------------

/** One observable event in a decision trace (tagged union on `event`). */
export interface TraceEvent {
  event: string;
  [k: string]: unknown;
}
export interface Trace {
  events: TraceEvent[];
}

/**
 * One backtest run's output: per-period returns + the decision trace + per-step
 * confidences/outcomes. Recomputed from raw decisions, never self-reported.
 */
export interface Run {
  returns: number[];
  trace: Trace;
  confidences: number[];
  outcomes: boolean[];
  /** Compute/token cost (any consistent unit); 0 = not reported. */
  cost: number;
  [k: string]: unknown;
}

// --- Trajectory (the recompute-to-verify artifact) -------------------------------

/** One captured decision step: the agent's raw output at one observation. */
export interface DecisionStep {
  step: number;
  observation_id: string;
  decision: Decision;
}

/** One captured backtest run (window × seed): the raw decisions + replay coords. */
export interface RunTrajectory {
  window_start: number;
  window_end: number;
  seed: number;
  steps: DecisionStep[];
}

// --- Dataset + engine inputs -----------------------------------------------------

/** A leak-free point-in-time price panel: a shared date axis + per-symbol closes. */
export interface Dataset {
  dates: string[];
  /** symbol → closes, each array aligned to `dates`. */
  closes: Record<string, number[]>;
  /** symbol → per-step cash dividend, aligned to `dates`. May be omitted. */
  dividends?: Record<string, number[]>;
}

/** A simulation window over the date axis: half-open `[start, end)`. */
export interface Window {
  start: number;
  end: number;
}

/** Basis-point transaction-cost model; every field falls back to the engine default. */
export interface CostModel {
  fee_bps?: number;
  slippage_bps?: number;
  impact_bps?: number;
  financing_bps?: number;
  /** Max fraction of NAV traded per step; omit for unlimited liquidity. */
  max_participation?: number;
}

/** Deterministic synthetic-panel parameters. */
export interface SyntheticParams {
  n_symbols?: number;
  n_days?: number;
  seed?: number;
}

/** Where a baseline run's prices come from: synthetic params OR raw CSV text. */
export interface DatasetSource {
  synthetic?: SyntheticParams;
  /** `date,symbol,close[,dividend]` long-format CSV (header required). */
  csv?: string;
}

/** The named baseline agents shipped in-process by the engine. */
export type BaselineAgent = "buy_and_hold" | "hold" | "momentum" | "random";

/** Config for {@link runBaseline}. */
export interface BaselineConfig {
  agent: BaselineAgent;
  /** Defaults to a 4×120 synthetic panel (seed 0). */
  dataset?: DatasetSource;
  /** Defaults to `{ start: 20, end: dataset.length }` (20-bar warm-up). */
  window?: Window;
  /** Execution seed (slippage noise; also seeds the `random` agent). */
  seed?: number;
  costs?: CostModel;
  /** Trailing window for the `momentum` baseline (default 10). */
  momentum_lookback?: number;
}

/** Walk-forward window-generation parameters. */
export interface WalkForwardParams {
  n_days: number;
  warmup: number;
  test: number;
  step: number;
}

/** A coarse market-regime label over a window. */
export type Regime = "bull" | "bear" | "chop";

/** One named adversarial stress scenario. */
export interface StressScenario {
  name: string;
  dataset: Dataset;
}
