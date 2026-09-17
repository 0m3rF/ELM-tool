# ELM Tool v2 release to-do

Status: alpha. Updated 2026-09-17. Check boxes represent verified evidence, not planned support.
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
  Transfer preview added on 2026-09-16 and live-tested: a new `job.preview` daemon operation
  (`crates/elm-daemon/src/runtime.rs`) opens the job's real source and destination and runs the
  same real, read-only `preflight()` a submitted job runs — reporting destination columns
  (source/output type per column after conversions), connector capabilities/warnings, and
  whether the requested write mode/consistency can publish safely — without calling `begin()`,
  so no staging DDL, file creation, or job record is ever produced. Exposed as `--dry-run` on
  `elm copy ...` (elm-cli) and as a "Run preview" step gating "Submit transfer" on the desktop
  New Transfer wizard (`crates/elm-desktop/src/App.tsx`; any field change invalidates a prior
  preview). Live-tested 2026-09-16: two fast file-based daemon IPC tests
  (`crates/elm-daemon/tests/ipc.rs`) plus one live test against the disposable PostgreSQL
  fixture (`crates/elm-daemon/tests/postgres_preview.rs`, `--ignored`, matching the existing
  Oracle daemon test's convention) confirming a preview never creates the destination table,
  never appears in `job.list`, and reports (not throws) an unsafe write-mode/consistency
  combination. `cargo fmt`, `cargo clippy -D warnings`, and the full `cargo test --workspace
  --all-features` suite pass; `npm run build` (tsc + vite) passes for the desktop frontend.
  Connection diagnostics (`environment.test`, a real live connection attempt with a typed error)
  already existed and were not changed. **Not done**: wizard validation is still only HTML5
  `required` fields plus basic relation-string parsing — no deeper checks (e.g. duplicate
  masking rules, oversized batch settings) were added, and none of this was exercised through
  an actual running GUI window in this session (no GUI automation tool was available), only
  through the daemon/CLI layer and a `tsc`/`vite` build of the frontend.

  Follow-up on 2026-09-17: investigated both callouts from the note above.
  - Oversized batch settings: not a real gap. `RuntimeSettings::validate` (`crates/elm-core/src/
    types.rs`) already fails closed if `memory_budget_bytes` can't hold at least two target
    batches, and it's invoked from `JobSpec::validate` on every submission path (`elm-state`,
    `elm-cli`, `elm-engine::pipeline`, `elm-daemon::preview_job`). The desktop's own Settings
    screen can't construct an invalid combination in the first place — memory is a 128-2048 MiB
    range and batch is an 8/16/32 MiB fixed choice, so batch is always well under half the
    minimum budget. Nothing to fix.
  - Duplicate masking rules: a real, previously unverified gap, confirmed by reading
    `apply_masks` (`crates/elm-core/src/masking.rs`): it applies rules in order and overwrites
    the same column index for each match, so selecting two masking rules for one column silently
    discards the first and keeps only the last, with no warning anywhere. Fixed at the root
    (`JobSpec::validate` in `crates/elm-core/src/types.rs`), so every entry point (CLI, daemon
    preview, daemon submit, and the engine's own pre-run check) now rejects a spec with more
    than one masking rule targeting the same column, naming the column in the error. Added
    `types::tests::duplicate_masked_column_is_rejected`. Also added a client-side pre-check in
    the desktop wizard (`crates/elm-desktop/src/App.tsx`): a computed `hasMaskConflict` disables
    both "Run preview" and "Submit transfer" and shows an inline alert naming the conflicting
    column(s) the moment two checked rules collide, instead of waiting for a server round trip;
    toggling masks now also invalidates a prior preview, matching the wizard's existing "any
    field that changes what would be transferred invalidates the preview" rule (mask selection
    was missing from that invalidation list before this pass). Verified with `cargo fmt --all --
    check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace
    --all-features` (0 failures), and `npm run build` (tsc + vite, passes). Not exercised through
    an actual running GUI window (no GUI automation tool was available in this session).
- [ ] Test live progress, close/reconnect, cancellation, resume/restart, history filtering, and actionable failures.
  Partial pass on 2026-09-17, daemon/IPC layer only (the desktop command surface is a thin
  passthrough to these same operations, but no GUI automation tool was available in this
  session, so nothing below was exercised through an actual running GUI window).
  Added `crates/elm-daemon/tests/job_lifecycle.rs`: a real `elm-daemon` OS subprocess (not an
  in-process task — a single-threaded in-process daemon cannot reliably race a live transfer
  against a second client) is sent a 150k-row transfer, then cancelled from one `DaemonClient`
  connection while a different connection watches it; a fresh connection opened afterward
  ("window reopened") still sees the correct terminal state, and a terminal job can then be
  deleted and disappears from `job.list`. Extended `crates/elm-daemon/tests/ipc.rs` with a
  real actionable-failure case (an explicit lossy conversion of an unparseable value) proving
  `PublicError.remediation` reaches a live watcher and still reads back correctly from
  `job.list` after a reconnect. Ran the new cancellation test 12 consecutive times to confirm
  it is not a timing fluke.
  Found and fixed three real, reproducible bugs surfaced by writing these tests (none were
  previously covered by any test, native or otherwise):
  1. `elm-state`'s `record_progress` used a deferred SQLite transaction that reads the job's
     current state and then writes based on it; under real concurrent writers (a running job's
     own progress updates racing a `job.cancel`) this produced an immediate "database is
     locked" even with `busy_timeout` set, because a deferred read-then-upgrade conflict is a
     different case than the initial-lock-acquisition wait `busy_timeout` covers. Fixed by
     opening that transaction `Immediate` (`crates/elm-state/src/store.rs`).
  2. `DaemonRuntime::cancel_job` persisted the `Cancelling` state before flipping the in-memory
     `CancellationToken`, and `TransferEngine::run` (`crates/elm-engine/src/pipeline.rs`)
     classified any resulting error as `Failed` unless it was literally `ElmError::Cancelled`.
     Because `tokio::select!` does not have to prefer a just-ready cancellation branch over an
     already-buffered batch, one more batch can legitimately be mid-flight when cancellation is
     requested; its own state write then loses to `Cancelling` and comes back as a state-machine
     `Conflict`, not `Cancelled` — so a normal, correctly-requested cancellation was being
     reported to the user as a failed job with a confusing internal error message. Fixed by
     cancelling the token before the store write, and by having the pipeline capture whether
     cancellation was already requested before its own unconditional self-cancel-on-any-error
     and treating that the same as `ElmError::Cancelled`.
  3. `produce_batches`' call to acquire a memory permit for the next batch was the one blocking
     point in that loop not raced against cancellation (every other wait in the pipeline is).
     Reproduced once in 6 runs before the fix (the job never reached a terminal state within
     30s and the daemon process went idle, not busy); fixed by selecting it against the
     cancellation token like the loop's other awaits (`crates/elm-engine/src/pipeline.rs`).
  Verified with the full `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
  -- -D warnings`, and `cargo test --workspace --all-features` (0 failures) after the fixes.
  **Not done in this pass**: resume/restart already has separate coverage (`process_restart.rs`,
  §1); nothing here touched masking-through-the-UI, pause semantics, history *filtering* as a
  UI feature (the desktop's client-side search/state filters in `App.tsx` were not exercised),
  or live-progress rendering/accessibility, and none of it ran against an actual GUI window.

  Follow-up on 2026-09-17, closing the history-filtering gap named above: no GUI automation tool
  is available for the Tauri desktop app in this environment (checked again — no `tauri-driver`,
  Edge/Chrome WebDriver, or existing test harness), so building real click-through GUI coverage
  wasn't attempted as a workaround. Instead, extracted the desktop's filtering and validation
  logic out of the React components and into a plain module (`crates/elm-desktop/src/logic.ts`:
  `filterJobs`, `findDuplicateMaskColumns`, `parseRelation`, the last two also lifted out of
  inline component code from earlier passes) so it's independently testable without a DOM. Added
  `vitest` as a dev dependency (pinned at 5.0.1 directly — `^3` initially resolved a version with
  a known moderate `@vitest/mocker` path-traversal advisory; `npm audit` now reports zero
  vulnerabilities) and `crates/elm-desktop/src/logic.test.ts`: 19 cases covering the Jobs
  screen's exact filter behavior (name match, id-substring match, case-insensitivity, a nameless
  job falling back to id matching, state-only filtering, combined query+state filtering, and no
  match), `parseRelation`'s three accepted forms plus its three rejection cases, and
  `findDuplicateMaskColumns`'s selected/unselected and single/multiple-conflict cases. All 19
  pass (`npm run test`); `npm run build` still passes and produces an identical-size bundle,
  confirming the test file isn't swept into the shipped app. **Still not done**: this is unit
  coverage of the logic, not of rendering or user interaction — no test opens the Jobs screen,
  types into the filter input, or clicks a checkbox, and accessibility through an actual GUI
  window remains unverified for the reason stated above.

  Follow-up on 2026-09-17, investigating "live-progress rendering": found a real, previously
  unnoticed bug rather than just a test gap. The Jobs screen's `<progress>` bar
  (`crates/elm-desktop/src/App.tsx`) was hardcoded to `value={state === "succeeded" ? 100 :
  state === "running" ? 55 : 10}` — a fabricated number derived only from coarse state, never
  from the real `rows`/`bytes` a running job had actually moved. A job 95% through a transfer and
  one that had just started both showed exactly 55%. Confirmed there's no way to compute a real
  percentage today: `JobProgress` (`crates/elm-core/src/types.rs`) carries no total-rows or
  total-bytes field, and its one field that could imply a completion estimate, `eta: Option
  <Duration>`, is unconditionally `None` everywhere it's set (`elm-engine/src/pipeline.rs`,
  `elm-daemon/src/runtime.rs` — grepped for every assignment). So a real percentage isn't
  knowable yet, and showing one anyway is actively misleading. Fixed by making the bar honest
  instead of inventing a fix for data that doesn't exist: a succeeded job shows a full
  `value="100"` bar; any other active state (`queued`/`preflighting`/`running`/`publishing`/
  `cancelling`) shows a native indeterminate `<progress>` (no `value` attribute, which renders as
  an animated bar rather than a fixed position) labeled "exact completion percentage isn't
  tracked yet"; a terminal non-success state (`failed`/`cancelled`/`interrupted`) shows no bar at
  all. The real per-second row/byte counters next to it were already accurate and are unchanged.
  Also deduplicated the active-state list into `isActiveJobState`/`ACTIVE_JOB_STATES` in
  `logic.ts` (it previously existed twice, once inline for the `job.watch` subscription effect
  and once — inconsistently, as three of five states — inline in the progress-bar condition) and
  added `logic.test.ts` coverage for it. `npm run test` (21 cases) and `npm run build` pass.
  **Still not done**: not exercised through an actual GUI window; a real total-rows/percentage
  field would need daemon/engine changes and is a separate, larger feature this fix didn't add.
- [ ] Verify masking configuration and deterministic retry/resume behavior through the UI.
  Verification found a real, product-level gap on 2026-09-17, not just a test gap: masking rules
  could be created, edited, tested, and removed (CRUD), but **no client ever attached a saved
  rule to an actual transfer**. `elm-desktop`'s `NewTransfer` hardcoded `masks: []` when building
  every job spec, and `elm-cli`'s `copy` command did the same — configuring a masking rule had
  zero effect on any real transfer, in either client. The daemon and engine were never at fault:
  `Operation::JobSubmit` faithfully applies whatever `spec.masks` it is given, and `apply_masks`
  is correctly deterministic; the rules just never reached `spec.masks` from any real workflow.
  Asked the user how mask rules should get attached; chosen approach was an explicit per-transfer
  picker, not silent auto-attachment by environment. Implemented:
  - `crates/elm-desktop/src/App.tsx`: the New Transfer wizard now fetches saved masking rules and
    shows a checkbox table (name, column, algorithm, scope) under a new "Masking" section; nothing
    is masked unless checked. Rules scoped to the transfer's source or target connection (plus
    global rules) are pre-checked as a suggestion when those connections change, but the user's
    own choices are the ones that reach `buildSpec()`'s `masks` field. Verified with `npm run
    build` (tsc + vite); not exercised through an actual running GUI window (no GUI automation
    tool was available in this session).
  - `crates/elm-cli/src/main.rs`: added a repeatable `--mask-rule <id>` flag (comma-separated or
    repeated) to `elm copy db-to-db|db-to-file|file-to-db`, resolved against `mask.list` and
    attached to the submitted spec; an unknown id fails closed before submission.
  Added real test coverage that did not exist at all before this pass (every prior masking test
  exercised `apply_masks`/`mask_text` directly, never a client-facing operation):
  - `crates/elm-daemon/tests/masks.rs`: a full `mask.add` → `mask.list` → `mask.test` →
    `mask.edit` → `mask.list` → `mask.test` → `mask.remove` → `mask.remove` (fails `NotFound`)
    round trip over real IPC — the exact sequence the desktop "Masking rules" screen and `elm
    mask ...` drive.
  - `crates/elm-daemon/tests/masking_resume.rs`: submits one identical `JobSpec` (same job id,
    same derived seed, same `Random`-algorithm masking rule) twice against a real `elm-daemon`
    subprocess — once straight through, once killed uncleanly mid-transfer and resumed (matching
    `process_restart.rs`'s pattern) — and asserts the published output is byte-for-byte identical
    either way. Run 4 consecutive times to rule out a timing fluke.
  Verified with `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, and `cargo test --workspace --all-features` (0 failures) after these changes.
  **Not done in this pass**: history filtering as a UI feature, live-progress rendering, and
  accessibility are untouched; nothing here ran against an actual GUI window.
- [x] Resolve pause semantics and UI behavior; do not advertise unimplemented actions.
  Verified 2026-09-17: a repo-wide search (`crates/`, `docs/`, `README.md`, excluding build
  output) found zero references to "pause" anywhere in the actual product — no `Operation`
  variant, no `JobState`, no CLI subcommand, no desktop button or copy. Nothing advertises an
  unimplemented pause action today; there was nothing to remove or disable. The real ambiguity
  this line was guarding against is more subtle: `cancel` and `resume` can look like a
  pause/continue pair, but `queue_retry` (`crates/elm-state/src/store.rs`) only allows resuming
  a job the daemon marked `Interrupted` or `Failed` — a `Cancelled` job can never be resumed, only
  deleted, and neither the desktop nor the CLI said so anywhere before this pass. Fixed by making
  that explicit everywhere a user could be confused: the desktop's Cancel button now confirms
  ("There is no pause: this cannot be resumed afterward...") before acting, and both action
  buttons carry a `title` explaining what they actually do; `elm jobs cancel --help` and `elm
  jobs resume --help` (`crates/elm-cli/src/main.rs`) state the same; and `README.md`'s safety
  model gained an explicit "There is no pause" line. `npm run build`, `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace
  --all-features` (0 failures) all pass. Not exercised through an actual running GUI window (no
  GUI automation tool was available in this session).
- [ ] Test keyboard-only navigation, focus, labels, contrast, and accessibility on all supported platforms.
  Partial pass on 2026-09-17, static-code audit only (no GUI automation tool or screen reader was
  available in this session, so nothing here ran against an actual running GUI window or assistive
  technology). Reviewed `crates/elm-desktop/src/App.tsx` and `styles.css`: every interactive
  control is a native `<button>`/`<input>`/`<select>`/`<label>` (no custom widgets, no `tabindex`
  hacks), so Tab order follows DOM order and all controls are reachable and operable from the
  keyboard already; screen-reader affordances (`role="alert"`/`role="status"` on error/success
  banners, `aria-label` on icon-only and filter controls, `aria-current` on the active nav item)
  were already present.
  Computed actual WCAG relative-luminance contrast ratios for every foreground/background color
  pair in `styles.css` (script not committed; ad hoc verification, reproducible from the formula
  in WCAG 2.x SC 1.4.3/1.4.11) and found five real, previously unverified failures, all fixed:
  - The global `:focus-visible` outline (`#9cc8ff`) was only 1.73:1 against the white/light
    backgrounds used by nearly every button, input, and select in the app — well under the 3:1
    non-text-contrast minimum, meaning keyboard focus was barely visible outside the dark sidebar.
    Changed the default outline to `#1d4ed8` (6.70:1 on white) and kept `#9cc8ff` as an
    `aside :focus-visible` override, since it already passes (10.49:1) on the dark sidebar
    background where the darker blue would fail.
  - Four secondary-text colors were below the 4.5:1 normal-text minimum: `.job-list small`
    (3.84:1), `.empty` paragraph text (3.57:1), `.job-detail dt` (4.02:1), and the dark-sidebar
    `kbd` shortcut hints (3.84:1). Darkened the first three to `#5b6b80` (5.07-5.44:1) and
    lightened `kbd` to `#8ea0b8` (6.81:1 on the dark sidebar).
  All other checked pairs (nav text, callout text, card text, pill/state badges) already passed.
  Verified with `npm run build` (tsc + vite, passes) after the change. **Not done**: no actual
  screen reader (NVDA/JAWS/VoiceOver/Orca) was run against the app, no real keyboard-only walk-
  through of a live window happened, and platform-specific accessibility behavior (Windows
  Narrator, macOS VoiceOver, Linux Orca/AT-SPI) is entirely unverified — this line stays open
  until that happens on each supported platform.

## 5. Native platforms and distribution

- [ ] Run Windows/macOS/Linux native database-driver and OS-keychain acceptance, including Oracle and ODBC setup failures.
- [ ] Verify TLS/certificate validation and credentials containing URL metacharacters.
- [ ] Build and smoke-test standalone CLI/daemon plus Windows installers, macOS DMGs, Linux AppImage/deb.
- [ ] Configure and verify platform signing/notarization; updater signing is not OS code signing.
- [ ] Test install/upgrade/uninstall and daemon lifecycle without losing jobs or credentials.

## 6. Release readiness

- [ ] Pass formatting, Clippy with warnings denied, unit/integration tests on Windows/macOS/Linux.
  Windows: verified continuously throughout this alpha's development, including this session.
  Linux: verified 2026-09-16 by locally reproducing `.github/workflows/rust.yml`'s `ubuntu-latest`
  job exactly (same apt packages, same three commands) in a disposable `rust:1-bookworm`
  container — not yet the same as a confirmed green run on GitHub's own runners, but a real
  Linux toolchain and OS, not a simulation. `cargo fmt --all -- --check` and
  `cargo clippy --workspace --all-targets -- -D warnings` passed clean; `cargo test --workspace
  --all-targets` initially failed two elm-daemon IPC tests with a real, reproducible bug: the
  Unix domain socket client (`crates/elm-daemon/src/transport.rs`) had no connection retry, so
  connecting before the daemon finished `UnixListener::bind` failed immediately with ENOENT,
  unlike the Windows named-pipe client two functions above it, which already retried exactly
  this race for 1 second. Fixed to match; full suite then passed, including
  `process_restart.rs`'s real-subprocess kill/restart test running on Linux via `SIGKILL`.
  **Still unverified: macOS.** No Apple hardware or CI access in this session; this gate
  remains open until it runs on real macOS, per this file's own evidence standard.
- [x] Pass dependency vulnerability and license policy audits; review native dependency distribution obligations.
  Found and fixed a real gap while auditing: `deny.toml`'s license allowlist did not include
  `GPL-3.0-or-later` — this workspace's own declared license — so `cargo deny check licenses`
  had been failing on every one of the project's seven crates, and the `dependency-policy` CI
  job (`.github/workflows/rust.yml`, `EmbarkStudios/cargo-deny-action@v2`) had presumably been
  red since it was introduced. Fixed by adding the workspace's own license plus two other
  legitimately-encountered transitive licenses (`Apache-2.0 WITH LLVM-exception`, `CC0-1.0`);
  marked all seven workspace crates `publish = false` (they are an internal application, not
  published libraries) so `allow-wildcard-paths` correctly covers their path-only internal
  dependencies; and reviewed the 7 "unmaintained" (not vulnerability) advisories that remained
  after that — all transitive via Tauri or the Oracle driver crate, all with no safe upgrade
  available — and recorded them in `deny.toml`'s `ignore` list with the reasoning and exact
  dependency chain for each, rather than silently widening the whole category. Verified
  2026-09-16: `cargo deny check` now exits 0 with `advisories ok, bans ok, licenses ok, sources
  ok`. Native dependency distribution reviewed: `.github/workflows/release.yml` only installs
  the open-source `unixodbc-dev`/`unixodbc` build-time headers (not a Cargo dependency, so
  outside `cargo deny`'s scope); no proprietary client (Oracle Instant Client, Microsoft ODBC
  Driver 18) is bundled, downloaded, or distributed by any release artifact, matching
  `docs/connectors.md`'s existing "ELM does not bundle or download proprietary clients" claim.
- [x] Review documentation: supported mappings, prerequisites, performance presets, recovery, safety modes,
  production precautions, and explicit Oracle non-atomic replacement tradeoffs.
  Partial pass on 2026-09-16: cross-checked a sample of specific claims in `docs/connectors.md`
  and `docs/operations.md` against the current source (SQL Server source/sink type lists,
  resource-tuning byte constants, `validate_source_resume` coverage, Oracle non-atomic-swap
  language) — all matched exactly. Found and fixed one real bug from this session's own
  editing: a prose paragraph had been inserted in the middle of `docs/connectors.md`'s
  connector table, splitting it into two broken markdown fragments. Not exhaustive — this did
  not check every mapping table entry line-by-line, did not read `docs/architecture.md` or
  `docs/correctness-contract.md` at all, and did not review production-precautions prose
  specifically. Leaving unchecked rather than claiming a full pass on a partial sample.

  Follow-up on 2026-09-17: read `docs/architecture.md` and `docs/correctness-contract.md` in
  full and cross-checked their most safety/correctness-relevant claims against current source —
  the 64 KiB memory-permit quantum and 2x reserve multiplier for masked/converted batches
  (`crates/elm-engine/src/memory.rs`, `pipeline.rs`), the two bounded `mpsc::channel(2)` pipeline
  stages, the 244-bit IPC token (two concatenated UUIDv4s = 2×122 random bits) and its `0o600`
  Unix file/socket permissions (`crates/elm-daemon/src/paths.rs`, `transport.rs`), and every
  dialect's identifier-quoting and parameter-placeholder style (`crates/elm-connectors/src/
  dialect.rs`: doubled double quotes for PostgreSQL/Oracle, doubled backticks for MySQL, doubled
  closing brackets for SQL Server; `$n`/`?`/`?`/`:n` parameters respectively). All matched
  exactly. Also read `README.md`'s "Production precautions" section specifically; its claims
  (non-production destination advisory, fail-closed lossy conversions and staging privileges,
  key-bounded resume not being a snapshot) are consistent with already-verified behavior.
  Found and fixed one real, previously unnoticed inaccuracy: `docs/architecture.md` described
  "job-scoped spill files" as if they actively let "a connector that advertised LOB spill
  support" stream oversized values today — but every real connector's `preflight()` reports
  `supports_lob_spill: false` (confirmed by reading all five), so `elm-engine::SpillStore` is
  dead code in the current alpha, exactly as `docs/connectors.md` already stated honestly
  elsewhere. `docs/architecture.md` just hadn't been updated to match when that was fixed.
  Reworded to state plainly that `SpillStore` exists but isn't wired into any connector, and
  pointed to `docs/connectors.md` for the per-connector detail. Also tightened the memory-permit
  sentence, which described the 2x reserve as mask-specific when the source
  (`self.spec.masks.is_empty() && self.spec.conversions.is_empty()`) triggers it for either
  masking or conversion. Not exhaustive — this pass verified specific technical claims, not
  every sentence in either document, and did not re-review `docs/connectors.md` or
  `docs/operations.md` beyond the prior pass's sample.

  Follow-up on 2026-09-17, closing the remaining gap: did the line-by-line numeric-constant sweep
  of `docs/connectors.md`'s type-mapping tables the first pass explicitly skipped, against the
  actual match arms in all four connectors. Confirmed exact matches for: SQL Server sink text
  4000 UTF-16 units / binary 8000 bytes and its 2 MiB parameter-array buffer limit
  (`sql_server_sink.rs`); SQL Server source's lossless type list — `TinyInt`→`UInt8`,
  `Decimal`/`Numeric` precision 1..=38, bounded `Char`/`Varchar`/`Binary`/`Varbinary` ≤8000 —
  matching the doc's bounded-type list exactly (`sql_server.rs`); Oracle sink NVARCHAR2(2000) /
  RAW(2000) and the empty-string/binary rejection (`oracle_sink.rs`); Oracle source's 8 MiB Arrow
  batch cap (`8 * 1024 * 1024` literal in three places) and the "native array reserves at most
  2 MiB" claim, which isn't a separate literal but falls out exactly from `batch_capacity`'s
  `row_bytes * 4` conservative reserve against an 8 MiB budget (`oracle_source.rs`); and MySQL's
  `TINYINT(1)` claim — `MYSQL_TYPE_TINY` maps to `Int8`/`UInt8` unconditionally, with no
  display-width or `BOOLEAN`-alias special case (`mysql.rs`). Re-read `docs/operations.md` in
  full; its specific numeric claims (Oracle's 15s/1-300000ms call timeout, the 8/16/32 MiB batch
  guidance, the 512 MiB default budget) all matched values already verified in the prior two
  passes, and its recovery/fault-injection narrative is consistent with the tests §1 and §3 cite
  as evidence. No further inaccuracies found. Checking this line off: every category it names has
  now had at least one substantive, source-verified pass, even though not literally every prose
  sentence in every doc was independently re-derived — a truly exhaustive word-by-word audit was
  never this line's bar, and re-verification should resume if the underlying source changes.
- [x] Audit clean-break cutover and remaining legacy assets/configuration references; no Python application shim.
  Clean result (2026-09-16): the only tracked Python file anywhere in the repo is
  `benchmarks/installed_elm_performance.py`, the intentional ELM 1.0.5 comparison script
  documented in §2's own evidence — not an application shim. No `requirements.txt`,
  `setup.py`, `pyproject.toml`, or other Python packaging artifact exists. No CI workflow,
  the Tauri build config, or any doc references a Python runtime, pip, or virtualenv. No
  legacy v1 config schema, migration shim, or orphaned v1-era asset was found; the desktop
  app's icon set is Tauri's own generated output, not a leftover. Separately noted, not a
  finding for this item: the repo also tracks `.augment/` (~200 files), unrelated
  AI-assistant tooling configuration for the Augment editor — not Python, not part of the
  product, but worth a human decision on whether it belongs in version control.
- [ ] Collect all release evidence before calling v2 stable. Do not substitute small smoke tests for acceptance gates.

## Evidence locations

- `tests/cross-database/README.md`, `Dockerfile`, `compose.yml`, `native.yml`, and
  `run-native.ps1`: reproducible 16-pair, fresh same-table, and interrupted same-table evidence.
- `.github/workflows/cross-database.yml`: PostgreSQL/MySQL pull-request gate and manually
  dispatched four-database native matrix.
- `benchmarks/README.md`: benchmark methodology, checker, Rust and installed-Python probes.
- `benchmark-results/`: local ignored measured results (not guaranteed to exist in a fresh clone).
- `crates/elm-connectors/tests/`, `crates/elm-daemon/tests/`: correctness/recovery tests.
  `crates/elm-daemon/tests/ipc.rs` and `postgres_preview.rs` cover the `job.preview` operation.
  `crates/elm-daemon/tests/job_lifecycle.rs` covers cancel/delete/reconnect against a real
  daemon subprocess; `ipc.rs`'s `a_failed_job_carries_an_actionable_remediation_...` test
  covers actionable failures surviving in history. `crates/elm-daemon/tests/masks.rs` covers
  the masking-rule CRUD operations; `masking_resume.rs` covers masked output determinism
  across a real interrupted-and-resumed transfer.
- `docs/connectors.md`, `docs/operations.md`: current restrictions and recovery behavior; see
  `docs/operations.md`'s "Transfer preview" section for `job.preview`/`--dry-run`.
