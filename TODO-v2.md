# ELM Tool v2 release to-do

Status: alpha. Updated 2026-09-15. Check boxes represent verified evidence, not planned support.
Oracle existing-table REPLACE uses the explicitly approved **non-atomic table swap**; never label it atomic.

## 1. Cross-database correctness — first implementation priority

- [ ] Build and run a reusable matrix suite for all 16 source/target database combinations.
  - [x] PostgreSQL ↔ MySQL: shared typed fixture, exact readback, APPEND, FAIL preservation,
    empty REPLACE with type preservation. Both live tests passed on 2026-09-15; CI workflow added
    (remote CI execution not yet verified). See `tests/cross-database/README.md`.
  - [ ] Extend the same assertions to Oracle and SQL Server with native-driver fixtures.
  - [ ] Cover self-copies and all three write modes for every pair; explicitly test Oracle swap semantics.
- [ ] Exercise every database ↔ CSV/NDJSON/Parquet direction with nulls, Unicode, exact decimals,
  binary, timestamps, reserved identifiers, empty results, and explicit conversion failures.
- [ ] Complete fault injection: connection loss, full disk, malformed rows, duplicate keys,
  privilege failures, publication failures, cancellation, and daemon termination/restart.
- [ ] Prove no partial target visibility in atomic mode and no duplicate rows after supported resumes.

## 2. Performance and memory

- [x] Add a read-only benchmark-report gate checker with regression tests.
- [x] Run a small PostgreSQL probe: three 258 MB transfers, exact verification, observed process memory.
- [x] Compare installed ELM 1.0.5 using the same small dataset. Rust median 19.34 s vs Python 49.02 s;
  observed worker peaks 121.6 MiB vs 414.9 MiB. Different batching/publication paths; not a release gate.
- [ ] Provision adequate disk and dedicated benchmark hardware; freeze reproducible datasets and Python baselines.
- [ ] Run fixed 10 GB self-copies for PostgreSQL, Oracle, MySQL, SQL Server; require ≥2× Python throughput.
- [ ] Verify configured 512 MiB RSS ceiling, with independently documented native-client allowances.
- [ ] Establish reviewed Rust baselines and enforce ≤10% subsequent throughput regression.
- [ ] Run 100 GB soak and interrupted/resumed transfers; verify bounded memory, exact data, no duplicates.
- [ ] Measure daemon/SQLite checkpoint overhead and representative narrow/wide/LOB workloads.

## 3. Connector completeness and recovery

- [x] Oracle native bulk source/sink and explicit table-swap REPLACE implemented and live-tested.
- [x] Oracle recovery: interrupted swap completion/restoration, checkpoint trimming, exact append rollback.
- [x] Oracle end-to-end daemon transfer, native keychain, table replacement, Parquet export verified on Linux.
- [ ] Audit remaining unsupported canonical types, LOB spill, timezone/precision mappings across all connectors.
- [ ] Implement outstanding database-source resume support or explicitly narrow documented release capabilities.
- [ ] Verify stable-source/key-bound resume warnings and fail-closed preflight for all unsupported operations.
- [ ] Validate staging/backup cleanup permissions and recovery instructions for each database.

## 4. Desktop acceptance

- [ ] Complete/verify connection diagnostics, transfer preview, and wizard validation.
- [ ] Test live progress, close/reconnect, cancellation, resume/restart, history filtering, and actionable failures.
- [ ] Verify masking configuration and deterministic retry/resume behavior through the UI.
- [ ] Resolve pause semantics and UI behavior; do not advertise unimplemented actions.
- [ ] Test keyboard-only navigation, focus, labels, contrast, and accessibility on all supported platforms.

## 5. Native platforms and distribution

- [ ] Run Windows/macOS/Linux native database-driver and OS-keychain acceptance, including Oracle and ODBC setup failures.
- [ ] Verify TLS/certificate validation and credentials containing URL metacharacters.
- [ ] Build and smoke-test standalone CLI/daemon plus Windows installers, macOS DMGs, Linux AppImage/deb.
- [ ] Configure and verify platform signing/notarization; updater signing is not OS code signing.
- [ ] Test install/upgrade/uninstall and daemon lifecycle without losing jobs or credentials.

## 6. Release readiness

- [ ] Pass formatting, Clippy with warnings denied, unit/integration tests on Windows/macOS/Linux.
- [ ] Pass dependency vulnerability and license policy audits; review native dependency distribution obligations.
- [ ] Review documentation: supported mappings, prerequisites, performance presets, recovery, safety modes,
  production precautions, and explicit Oracle non-atomic replacement tradeoffs.
- [ ] Audit clean-break cutover and remaining legacy assets/configuration references; no Python application shim.
- [ ] Collect all release evidence before calling v2 stable. Do not substitute small smoke tests for acceptance gates.

## Evidence locations

- `benchmarks/README.md`: benchmark methodology, checker, Rust and installed-Python probes.
- `benchmark-results/`: local ignored measured results (not guaranteed to exist in a fresh clone).
- `crates/elm-connectors/tests/`, `crates/elm-daemon/tests/`: correctness/recovery tests.
- `docs/connectors.md`, `docs/operations.md`: current restrictions and recovery behavior.
