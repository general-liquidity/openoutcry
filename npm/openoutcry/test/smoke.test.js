const test = require("node:test");
const assert = require("node:assert");
const oo = require("../dist/index.js");

/** Assert `r` has the shape of a valid {@link Run}. */
function assertValidRun(r, expectedLen) {
  assert.ok(Array.isArray(r.returns), "returns is an array");
  assert.equal(r.returns.length, expectedLen, "returns has the window length");
  assert.ok(
    r.returns.every((x) => typeof x === "number" && Number.isFinite(x)),
    "every return is a finite number",
  );
  assert.ok(r.trace && Array.isArray(r.trace.events), "trace.events is an array");
  assert.equal(r.confidences.length, expectedLen, "one confidence per step");
  assert.equal(r.outcomes.length, expectedLen, "one outcome per step");
  assert.equal(typeof r.cost, "number", "cost is a number");
}

test("runBaseline buy_and_hold produces a valid Run", () => {
  // Default synthetic panel is 120 days with a 20-bar warm-up → 100 steps.
  const run = oo.runBaseline({
    agent: "buy_and_hold",
    dataset: { synthetic: { n_symbols: 4, n_days: 120, seed: 11 } },
    seed: 1,
  });
  assertValidRun(run, 100);
  assert.ok(run.trace.events.length > 0, "buy-and-hold places orders, so the trace is non-empty");
});

test("every named baseline agent runs", () => {
  for (const agent of ["buy_and_hold", "hold", "momentum", "random"]) {
    const run = oo.runBaseline({ agent, seed: 3 });
    assertValidRun(run, 100);
  }
});

test("runBaseline accepts CSV data + an explicit window", () => {
  const csv =
    "date,symbol,close\n2025-01-01,AAA,10\n2025-01-02,AAA,11\n2025-01-03,AAA,12\n2025-01-04,AAA,13";
  const run = oo.runBaseline({
    agent: "buy_and_hold",
    dataset: { csv },
    window: { start: 1, end: 4 },
    seed: 0,
  });
  assertValidRun(run, 3);
});

test("replayRun recomputes a Run from a captured trajectory", () => {
  const dataset = oo.datasetSynthetic({ n_symbols: 4, n_days: 120, seed: 11 });
  const sym = Object.keys(dataset.closes)[0];
  // A hand-built trajectory: buy one name at 25% every step of a 10-bar window.
  const steps = Array.from({ length: 10 }, (_, i) => ({
    step: i,
    observation_id: dataset.dates[20 + i],
    decision: {
      orders: [
        { symbol: sym, action: "buy", target_weight: 0.25, confidence: 0.6, rationale: "test" },
      ],
      reasoning: "fixed allocation",
    },
  }));
  const trajectory = { window_start: 20, window_end: 30, seed: 7, steps };
  const run = oo.replayRun(dataset, trajectory);
  assertValidRun(run, 10);

  // Recompute-to-verify: replaying the same artifact is byte-identical.
  const again = oo.replayRun(dataset, trajectory);
  assert.deepEqual(again.returns, run.returns, "replay is deterministic");

  // Tamper with the artifact → the honest recompute yields different returns.
  const tampered = JSON.parse(JSON.stringify(trajectory));
  for (const s of tampered.steps) for (const o of s.decision.orders) o.target_weight *= 2;
  const tamperedRun = oo.replayRun(dataset, tampered);
  assert.notDeepEqual(tamperedRun.returns, run.returns, "a tampered trajectory cannot reproduce the returns");
});

test("datasetSynthetic is deterministic", () => {
  const a = oo.datasetSynthetic({ n_symbols: 3, n_days: 40, seed: 99 });
  const b = oo.datasetSynthetic({ n_symbols: 3, n_days: 40, seed: 99 });
  assert.deepEqual(a, b);
  assert.equal(a.dates.length, 40);
  assert.equal(Object.keys(a.closes).length, 3);
});

test("stressSuite, walkForward and tagRegime bridge", () => {
  const suite = oo.stressSuite(1);
  assert.equal(suite.length, 2);
  assert.equal(suite[0].name, "flash_crash");

  const windows = oo.walkForward({ n_days: 200, warmup: 20, test: 60, step: 60 });
  assert.equal(windows.length, 3);
  assert.equal(windows[0].start, 20);

  const ds = oo.datasetSynthetic({ n_symbols: 2, n_days: 120, seed: 7 });
  const regime = oo.tagRegime(ds, { start: 0, end: 120 });
  assert.ok(["bull", "bear", "chop"].includes(regime));
});

test("an unknown baseline agent throws (errors do not cross as panics)", () => {
  assert.throws(() => oo.runBaseline({ agent: "nope" }), /unknown baseline agent/);
});
