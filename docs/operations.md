# Operations and recovery

## Per-user daemon

`elm daemon start` starts one hidden daemon for the current OS user. Any normal CLI or desktop
operation also starts it on demand. `elm daemon status` reports its PID and active count;
`elm daemon stop` requests a clean shutdown. A lock file and first-instance named-pipe semantics
prevent duplicates.

State is stored in the OS per-user local data directory as `elm-v2.sqlite3`. `daemon.token` is an
authentication secret, not a portable configuration file. Do not copy either file between users.

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

## Resource tuning

The default 512 MiB budget includes queued input and transformed output Arrow buffers but excludes a
documented fixed native-client allowance. Keep target batches at 8 MiB for very wide rows or high
latency variance, 16 MiB normally, and 32 MiB for narrow rows on high-bandwidth local networks.
Increasing batch size does not increase the process budget.

Parquet can buffer encoded row groups; the sink flushes on its configured threshold. A full disk,
failed `fsync`, or publication error fails the attempt and leaves the original output in place.
