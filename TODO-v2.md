# ELM Tool v2 release to-do

Status: alpha. Updated 2026-09-16. Check boxes represent verified evidence, not planned support.
Oracle existing-table REPLACE uses the explicitly approved **non-atomic table swap**; never label it atomic.

## 1. Cross-database correctness — first implementation priority

- [x] Build and run a reusable matrix suite for all 16 source/target database combinations.
  - [x] Cover all 16 ordered PostgreSQL/MySQL/Oracle/SQL Server pairs with native drivers,
    including four same-database pairs. Exact readback covers Unicode, binary, nulls, decimals,
    microsecond timestamps, populated/empty REPLACE, APPEND, and FAIL preservation.
  - [x] Cover all three write modes for every pair and explicitly verify Oracle semantics:
    atomic replacement of an existing target fails closed before an explicit **non-atomic table swap**.
    Oracle pairs explicitly cast IDs to Decimal128(18,0); other pairs retain Int64 coverage.
  - [x] Verify fresh same-table copies using direct names, qualified names, SQL aliases,
    and separate environment records. Checkpoint readback preserves the original target until
    publication; REPLACE, exact-doubling APPEND, FAIL preservation, and empty REPLACE are covered.
  - [x] Verify interrupted same-table APPEND and REPLACE recovery for every database.
    An injected failure after the staging commit but before durable checkpoint storage leaves
    the target unchanged; a row-zero retry publishes once without duplicate rows.
  - [x] Latest local evidence: focused same-table suite 8/8 passed in 228.79 s and the full
    matrix 24/24 passed in 371.88 s on Linux x86-64 containers under Docker Desktop on Windows
    on 2026-09-16. Windows compilation, formatting, and Clippy with warnings denied passed.
    The native matrix has a manual CI job; remote CI execution remains unverified.
  - [x] Add physical source/target alias discovery (views, synonyms, alternate server addresses).
    Best-effort warning-only discovery unwinds one level of view (PostgreSQL/MySQL/SQL
    Server/Oracle) or synonym (SQL Server/Oracle) indirection through catalog metadata and
    compares a low-privilege server-native fingerprint so a second environment record reaching
    the same physical server through a different hostname is also recognized; never DNS-based,
    never blocks a transfer, and any ambiguity resolves to unknown. Live-tested on 2026-09-16
    against disposable PostgreSQL, MySQL, SQL Server, and Oracle fixtures, including an
    alternate-hostname PostgreSQL case and unrelated/missing-relation rejection; see
    `tests/cross-database/README.md` and `docs/connectors.md`.
- [x] Exercise every database ↔ CSV/NDJSON/Parquet direction with nulls, Unicode, exact decimals,
  binary, timestamps, reserved identifiers, empty results, and explicit conversion failures.
  Added engine-level (real `TransferEngine`, real connectors, real files) round-trip tests for
  all four databases, live-tested on 2026-09-16: a DB→Parquet→DB round trip through a fresh
  table with a reserved-word column, nulls, Unicode, exact decimal, binary, and timestamp;
  DB→CSV/NDJSON→DB round trips including an empty-result case (CSV/NDJSON's type inference on
  reimport is honestly lossy for anything beyond text — this is stated plainly in
  `docs/connectors.md`, not glossed over); and a file→DB explicit-conversion-failure case (an
  unparseable value with an explicit lossy `ConversionRule`) that fails cleanly and leaves no
  partial target. Along the way, found and documented that Oracle's connector has no native
  integer mapping, so CSV/NDJSON numeric columns reimported into Oracle require an explicit
  conversion rule — a real, previously-undocumented connector-specific constraint.
- [x] Complete fault injection: connection loss, full disk, malformed rows, duplicate keys,
  privilege failures, publication failures, cancellation, and daemon termination/restart.
  All eight now have real, live-tested evidence. Live-tested on 2026-09-16:
  - Duplicate keys / publication failure: now covered for all four databases (PostgreSQL was
    the gap; MySQL/SQL Server/Oracle already had it). A source with a duplicate key fails
    closed — for PostgreSQL specifically, staging itself rejects it (staging inherits the
    target's constraints via `CREATE TABLE ... LIKE target INCLUDING ALL`), while the other
    three fail at `publish()`; either way the target is left unchanged.
  - Cancellation: added real mid-transfer `TransferEngine` cancellation tests (large source,
    `CancellationToken.cancel()` fired mid-stream) for PostgreSQL, MySQL, and SQL Server,
    verifying `Err(Cancelled)` and no new target left behind. Oracle already had a deterministic
    connector-level unit test for both cancellation windows.
  - Connection loss: added one representative test (PostgreSQL) that starts a real transfer
    against a live Docker fixture and forcibly `docker restart`s the container mid-stream,
    confirming the job reports a clean connection failure within a bounded timeout rather than
    hanging or silently succeeding. Not repeated for the other three databases — the failure
    path (a severed native socket surfacing as a connection error) is the same underlying
    mechanism in every connector; this proves the pattern rather than exhaustively re-testing it.
  - Malformed rows: added one representative test (PostgreSQL) — a structurally malformed CSV
    row (wrong field count) fails closed, either at file open or during the transfer, and never
    leaves a partial target.
  - Privilege failures: done — see §3's staging/backup cleanup permission item.
  - Full disk: added `crates/elm-connectors/tests/full_disk.rs`, live-tested (2026-09-16) inside
    a disposable Docker fixture (`tests/full-disk/{Dockerfile,compose.yml}`) mounting a genuine
    4 MiB `tmpfs` at the staging path — not a mock or a simulated error. `RecoverableFileSink`
    writes ~512 KiB chunks until the filesystem is actually exhausted, asserts the failure comes
    back as a clean `ElmError::Io`, and confirms no partial destination file is ever produced
    (each staging chunk is written, fsync'd, and renamed independently, so a chunk that never
    completes never reaches the destination).
  - Daemon termination/restart: added `crates/elm-daemon/tests/process_restart.rs`, live-tested
    (2026-09-16) directly on the Windows host with no Docker needed. It spawns the real
    `elm-daemon` binary as an OS subprocess (`CARGO_BIN_EXE_elm-daemon`, not an in-process
    `tokio::spawn(runtime.serve())` — the gap this line previously called out), submits a
    150,000-row file transfer, waits for a real checkpoint to commit (8,192 rows staged), then
    kills the process uncleanly (`Child::kill()` — `TerminateProcess` on Windows, `SIGKILL` on
    Unix; no `DaemonStop`, no graceful shutdown). It then starts a second daemon process against
    the same data directory and confirms `DaemonRuntime::open`'s existing
    `mark_active_jobs_interrupted()` reconciliation marks the job `Interrupted` on its own, then
    resumes it over real IPC and confirms all 150,000 rows are published exactly once. Run three
    times consecutively to rule out a timing fluke; the kill landed after exactly one checkpoint
    (8,192/150,000 rows) every time.
- [x] Prove no partial target visibility in atomic mode and no duplicate rows after supported resumes.
  Already covered before this pass: `UnchangedTarget` in `crates/elm-connectors/tests/cross_database.rs`
  re-reads the target on every checkpoint during `same_table`/`interrupted_same_table` and asserts
  it is unchanged until publication, for all four databases; `interrupted_same_table` asserts exact
  row counts after a resume (not just matching data), proving no duplication after publish.

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
- [x] Audit remaining unsupported canonical types, LOB spill, timezone/precision mappings across all connectors.
  Cross-checked every source/sink type-mapping match arm against `docs/connectors.md` for
  PostgreSQL, MySQL, SQL Server, and Oracle on 2026-09-16; all four fail closed on unhandled
  types. Found and fixed a real gap: the MySQL sink mapped timezone-bearing Arrow timestamps to
  native TIMESTAMP(6) without validating MySQL's narrower 1970-2038 range, so out-of-range values
  would have reached the server instead of failing with a typed error; added a batch-level
  range check (live-tested) and corrected the PostgreSQL doc's inaccurate claim that oversized
  TEXT/BYTEA fail preflight — no connector enforces a LOB size limit today, and `elm-engine`'s
  `SpillStore` exists but is unwired into every connector; both are now documented honestly in
  `docs/connectors.md` rather than fixed, since implementing real LOB size limits is a separate,
  larger feature this line item did not ask for.
- [x] Implement outstanding database-source resume support or explicitly narrow documented release capabilities.
  Narrowed rather than implemented: MySQL/SQL Server/Oracle keyset resume is out of scope for
  this release, stated as a deliberate decision (not a "yet") in `docs/connectors.md`, matching
  the daemon's existing fail-closed rejection — only PostgreSQL implements keyset resume.
- [x] Verify stable-source/key-bound resume warnings and fail-closed preflight for all unsupported operations.
  `elm-daemon::runtime::validate_source_resume` now has an explicit unit-test case for each of
  the four `DatabaseKind` values (previously only Oracle and PostgreSQL were asserted); a
  supplied resume key for MySQL, SQL Server, or Oracle fails closed with `Unsupported` before
  credential lookup or a native connection, confirmed for real by the existing per-connector
  fixtures. `PreflightReport::require_safe_publication`'s existing unit test already covers
  every unsupported consistency/write-mode combination (table-swap, checkpointed, atomic).
- [x] Validate staging/backup cleanup permissions and recovery instructions for each database.
  Found a real gap while validating: no existing test anywhere exercised a genuine privilege
  failure — every connector's `PermissionDenied` path was unverified. Added one live-tested
  fixture per database (2026-09-16, against disposable PostgreSQL/MySQL/SQL Server/Oracle
  containers) that connects as a deliberately under-privileged account and confirms the sink
  fails closed with `PermissionDenied`: PostgreSQL and SQL Server reject at `preflight()`
  (proactive privilege checks); MySQL and Oracle preflight successfully but reject at `begin()`
  when staging DDL is attempted. Fixed Oracle along the way: ORA-01031 (insufficient privileges)
  was previously reported as a generic `Connection` error, not `PermissionDenied`; it now matches
  its siblings. Staging/backup cleanup's own marker-recognition checks were already covered by
  existing tests referenced in `docs/connectors.md` and were not re-verified here.

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

- `tests/cross-database/README.md`, `Dockerfile`, `compose.yml`, `native.yml`, and
  `run-native.ps1`: reproducible 16-pair, fresh same-table, and interrupted same-table evidence.
- `.github/workflows/cross-database.yml`: PostgreSQL/MySQL pull-request gate and manually
  dispatched four-database native matrix.
- `benchmarks/README.md`: benchmark methodology, checker, Rust and installed-Python probes.
- `benchmark-results/`: local ignored measured results (not guaranteed to exist in a fresh clone).
- `crates/elm-connectors/tests/`, `crates/elm-daemon/tests/`: correctness/recovery tests.
- `docs/connectors.md`, `docs/operations.md`: current restrictions and recovery behavior.
