# Operations and recovery

## Per-user daemon

`elm daemon start` starts one hidden daemon for the current OS user. Any normal CLI or desktop
operation also starts it on demand. `elm daemon status` reports its PID and active count;
`elm daemon stop` requests a clean shutdown. A lock file and first-instance named-pipe semantics
prevent duplicates.

State is stored in the OS per-user local data directory as `elm-v2.sqlite3`. `daemon.token` is an
authentication secret, not a portable configuration file. Do not copy either file between users.

## Transfer preview

`job.preview` (CLI: `elm copy <db-to-db|db-to-file|file-to-db> ... --dry-run`; desktop: the "Run
preview" button on the New Transfer wizard's Review step) runs the same real preflight a submitted
job runs — opening the actual source and destination, checking privileges, and computing the
destination schema — without moving any rows, creating any job, or leaving anything behind. It
never calls a connector's `begin()`, which is the only place staging DDL or file creation happens;
preflight itself is a read-only privilege/schema check by contract (`docs/connectors.md`'s own
matrix depends on this). A preview response reports the destination columns (source and output
type per column, after any configured conversions), the sink's capabilities and warnings, and
whether the requested write mode and consistency combination can publish safely — the same check
`require_safe_publication` runs immediately before every real job's `begin()`, surfaced here as
data instead of a hard failure so the wizard can show it before submission. Live-tested on
2026-09-16 against the disposable PostgreSQL fixture and file source/sink pairs: a preview never
creates the destination table or file, is never visible through `job.list`/`job.get`, and an
unsafe write-mode/consistency combination is reported, not thrown.

## Same-table copies

Live acceptance tests verify fresh staged jobs that read and write the same base table
on PostgreSQL, MySQL, SQL Server, and Oracle. REPLACE publishes the selected rows, including
an empty selection; APPEND doubles the original rows exactly; FAIL preserves the existing
table and returns a conflict. Separate-connection readback at staging checkpoints verifies
that the target still contains its original values before publication.

The tests cover direct table selection, qualified names, SQL table aliases, and separate
environment records pointing to the same database. `elm_connectors::physical_identity::resolve`
additionally unwinds one level of view or synonym indirection and compares a server-native
fingerprint, so a source and destination reached through a view/synonym or a different
configured hostname for the same physical server are also recognized; this is a best-effort
warning surfaced on the job, not a blocking check, since staged publication is already safe
independent of whether source and destination are the same physical table. See
`docs/connectors.md`. Concurrent writers remain outside this acceptance evidence.

Interrupted same-table APPEND and REPLACE are accepted for unkeyed database sources. Live tests
inject failure after a staging batch commits but before its checkpoint is stored. The target
remains unchanged; retry with the same job identity restarts the source from row zero, reconciles
the private staging data, and publishes once without duplicate rows. This verifies the engine and
connector recovery boundary, not forced daemon-process termination.

Oracle existing-table REPLACE requires explicit `table-swap` consistency and remains
**non-atomic**, including when the source and target are the same table. Connector type
restrictions and dependency restrictions still apply.

## Interrupted jobs

After an unclean stop, an active job becomes `interrupted`. Inspect its last event and checkpoint:

```powershell
elm jobs show <job-id>
elm jobs resume <job-id>
elm jobs watch <job-id>
```

Resume creates a new attempt. File retries validate the source fingerprint, reopen the last logical
row offset, verify the staging manifest and contiguous committed chunk sequence, and then publish
the old and new chunks exactly once. A changed input or missing/unrecognized staging resource fails
closed. File-to-PostgreSQL retries validate the same file checkpoint and continue into the marked,
row-count-verified staging table. A private batch-sequence column lets recovery remove a database
batch committed just after SQLite's last checkpoint. Structured PostgreSQL table sources continue
from configured non-null unique keys up to their captured initial bound; sources without keys and
explicit SQL restart private staging from zero. File-to-MySQL supports the same staging-batch
reconciliation; MySQL sources restart from zero. Before reopening the source, MySQL retries check
their publication receipt. A committed APPEND or confirmed atomic rename completes the retry without
reading or writing any source rows again. An inconclusive receipt or older publication fence blocks
replay and requires manual destination reconciliation. SQL Server sources restart from zero and
reject resume keys. SQL Server file-source retries trim uncheckpointed staging batches and validate
row counts. Its publication transaction commits the target change and receipt together, so a
confirmed receipt completes a retry without reopening the source or duplicating an APPEND.
Oracle native calls default to a 15-second round-trip timeout. For a slow environment, explicitly
set `elm env edit <environment> --oracle-call-timeout-ms 120000` (also available on `env add`).
The Oracle-only setting accepts 1–300000 milliseconds; zero/unbounded timeouts are rejected.
It applies to reading, staging, publication, recovery, and connection-test native calls.
Desktop edits preserve this advanced setting. Cancellation can wait for an in-flight native call
and driver cleanup; this is not a whole-operation wall-clock deadline. The disposable cold-start
Oracle fixture uses 120000 ms after its initial journal write exceeded the default (DPI-1067).
No automatic retry or consistency downgrade is introduced.

Oracle sources restart from zero; file-to-Oracle retries validate schema hashes, trim staging to
the checkpoint, and verify row counts. Atomic APPEND receipts prevent replay. Explicit Oracle
`table-swap` REPLACE is non-atomic: a validated interrupted rename is completed before the source
is reopened. Unknown object identities stop recovery for manual inspection. The backup retains
the old table's indexes, grants, and constraints; these are not copied onto the replacement.

`elm jobs delete` applies that identity check to file staging, PostgreSQL staging, MySQL staging
and publication fences, SQL Server staging and receipts, and Oracle journals/staging/backups.
Oracle job deletion restores a renamed original if publication is incomplete; after successful
replacement it deletes the retained backup. Inspect or copy that backup before deleting the job
if you need to keep the old data. Oracle DROP uses the database's normal recycle-bin policy,
but recovery from a dropped backup must not be assumed. If a
directory contains an unrecognized entry, a manifest differs, or a PostgreSQL table lacks the exact
job comment, deletion fails and leaves the resource untouched for manual inspection. Connection
environments referenced by job history cannot be removed; delete the jobs first so keychain-backed
cleanup remains possible.

## Fault injection coverage

Live-tested for all four databases: privilege failures (staging fails closed with
`PermissionDenied`), and duplicate keys / publication failure (a constraint violation fails the
transfer and leaves the target unchanged — PostgreSQL rejects it while staging, since its
staging table inherits the target's constraints; the other three reject it at publish).
Live-tested for PostgreSQL, MySQL, and SQL Server: cancellation mid-transfer, via a real
`TransferEngine` run cancelled partway through a multi-batch transfer. Oracle already had a
deterministic connector-level cancellation unit test covering both cancellation windows.
Live-tested for PostgreSQL only, as a representative case rather than repeated per database:
connection loss (the database container is forcibly restarted mid-transfer; the job reports a
clean connection failure within a bounded timeout) and a structurally malformed source row (a
CSV row with the wrong field count fails closed, at open or during the transfer, without a
partial target).

Full disk and daemon process termination/restart are now covered too, each by a test that
exercises the real failure mode rather than a simulated one. `crates/elm-connectors/tests/full_disk.rs`
runs inside a disposable Docker fixture (`tests/full-disk/`) that mounts a genuine 4 MiB `tmpfs`
at the staging path, writes chunks until the filesystem is actually exhausted, and confirms the
failure surfaces as a clean `ElmError::Io` with no partial destination file.
`crates/elm-daemon/tests/process_restart.rs` spawns the real `elm-daemon` binary as an OS
subprocess (not the in-process `tokio::spawn(runtime.serve())` used by the interrupted-job
recovery tests above), submits a transfer, waits for a checkpoint to actually commit, kills the
process uncleanly (`TerminateProcess`/`SIGKILL`, no `DaemonStop`), starts a second daemon process
against the same data directory, and confirms `DaemonRuntime::open`'s
`mark_active_jobs_interrupted()` reconciliation marks the job `Interrupted` before resuming it
over real IPC to a successful, non-duplicated completion.

## Resource tuning

The default 512 MiB budget includes queued input and transformed output Arrow buffers but excludes a
documented fixed native-client allowance. Keep target batches at 8 MiB for very wide rows or high
latency variance, 16 MiB normally, and 32 MiB for narrow rows on high-bandwidth local networks.
Increasing batch size does not increase the process budget.

Parquet can buffer encoded row groups; the sink flushes on its configured threshold. A full disk,
failed `fsync`, or publication error fails the attempt and leaves the original output in place.
