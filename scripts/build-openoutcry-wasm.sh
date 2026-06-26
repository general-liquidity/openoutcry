#!/usr/bin/env bash
# Build the OpenOutcry trading-agent environment (openoutcry, via openoutcry-wasm) to
# WASM so TypeScript/Bun can run baselines + recompute-to-verify with the IDENTICAL
# engine as the Rust harness.
#
# Output: ./pkg with a JS module exposing `run_baseline(configJson)` /
# `replay_run(datasetJson, trajectoryJson, costsJson)` (+ dataset/stress/walk-forward
# helpers), which the `@general-liquidity/openoutcry` npm wrapper imports.
set -euo pipefail

rustup target add wasm32-unknown-unknown
cargo build -p openoutcry-wasm --release --target wasm32-unknown-unknown
WASM="target/wasm32-unknown-unknown/release/openoutcry_wasm.wasm"

if command -v wasm-bindgen >/dev/null 2>&1; then
  wasm-bindgen "$WASM" --out-dir pkg --target bundler
  echo "wrote ./pkg  — import { run_baseline } from './pkg/openoutcry_wasm'"
else
  echo "built $WASM"
  echo "To generate the JS bindings, install the CLI then re-run:"
  echo "  cargo install wasm-bindgen-cli"
fi
