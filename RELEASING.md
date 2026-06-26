# Releasing OpenOutcry

OpenOutcry ships from **one Rust engine** to three surfaces: the `openoutcry` crate
(crates.io), the `@general-liquidity/openoutcry` npm package (the WASM build), and the
`openoutcry` PyPI wheel (the pyo3 binding + Gymnasium adapter). It depends on the
**published** `sharpebench-*` crates (the simulator engine) rather than vendoring them.

## Cutting a version

```bash
# green checks (cargo-release will not run these)
cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo deny check

cargo release patch            # DRY RUN
cargo release patch --execute  # bump shared version + rewrite pins + tag vX.Y.Z + push
```

`release.toml` sets `publish = false` — the local machine never publishes. The `v*`
tag triggers CI, which publishes via **OIDC Trusted Publishing** (no stored tokens).

## One-time publishing setup (pending)

Before the first CI publish, each registry needs its trusted publisher configured —
mirroring the SharpeBench process:

- **crates.io** — `openoutcry`, `openoutcry-wasm`: a crate must exist before a trusted
  publisher can be added, so the **first** publish of each name needs a token
  (`cargo publish -p openoutcry` then `-p openoutcry-wasm`); then add the trusted
  publisher (owner `general-liquidity`, repo `openoutcry`, workflow `release.yml`) and
  never use a token again.
- **npm** — `@general-liquidity/openoutcry`: claim once (`npm publish --access public`),
  then add the trusted publisher.
- **PyPI** — `openoutcry`: configure a trusted publisher (GitHub → repo `openoutcry`,
  workflow `release.yml`, environment `pypi`); maturin builds + uploads the wheel.

Then set repo variables `PUBLISH_CRATES=true`, `PUBLISH_NPM=true`, `PUBLISH_PYPI=true`
and create the `crates` / `npm` / `pypi` GitHub Environments (with whatever review
protection you want on the gate). `release.yml` is added as part of the first publish.
