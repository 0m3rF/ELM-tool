# Architecture

The CLI and desktop are daemon clients. They do not own transfers, credentials, or database
connections, so closing either interface does not interrupt an active job.

```text
elm CLI ───────────────┐
                      ├─ authenticated per-user IPC ─ elm-daemon ─ SQLite + OS keychain
Tauri desktop ────────┘                                  │
                                                         ▼
               bounded Source → Arrow → Mask/Convert → staged Sink
```

## Process and memory boundary

`elm-engine` admits each batch through a semaphore measured in 64 KiB permits. A batch that will be
masked or converted reserves twice its observed Arrow allocation until the old and transformed
arrays can be released; an untransformed batch reserves once. Two bounded Tokio channels connect
acquisition, transform, and sink stages. This provides backpressure independently of connector row
counts. A batch larger than the configured process budget fails with `resource_exhausted`; it is
never accepted optimistically.

Connectors receive a byte target, not a fixed row count. `AdaptiveBatchSizer` learns observed row
width and converts the 8–32 MiB byte target into a row hint. `elm-engine::SpillStore` exists as a
job-scoped, disk-backed store for a value too large to hold in the batch memory budget, deleting its
temporary directory on drop, but it is not wired into any connector or the pipeline today: every
connector reports `supports_lob_spill: false`, so an oversized value is bounded only by the ambient
per-batch memory budget today, not by a dedicated spill path. See `docs/connectors.md` for the
per-connector detail.

Mask and conversion transforms execute from the versioned job contract. Conversion planning derives
the sink schema before loading, rejects unsupported casts, and conservatively classifies lossless
conversions. Other casts require a per-column `allow_lossy` opt-in. Runtime value failures abort the
batch rather than silently introducing nulls.

## Publication

Sinks perform preflight before loading. Capability negotiation includes atomic append/replace,
checkpointing, keyset resume, native bulk paths, and LOB spill. Atomic database jobs use a unique
staging relation and a recovery record. APPEND inserts staging rows in one publication transaction;
REPLACE swaps a validated staging object; FAIL checks existence before load. Dialect code accepts
only structured `Relation` and `Identifier` values and quotes each component.

PostgreSQL implements this contract for its documented alpha type subset. Each Arrow batch is a
separate completed binary COPY into a job-marked staging table, making the persisted checkpoint
durable. APPEND inserts and drops staging in one transaction; REPLACE transactionally renames the
old target to a private backup, renames staging, and drops the backup. Any failure rolls the whole
publication transaction back. Other database connectors remain feature-gated.

File sinks follow the same visibility rule with a sibling temporary file and atomic persistence.
APPEND first copies/rewrites the existing file into staging and verifies its schema. Parquet never
concatenates physical files; old batches are decoded and written into the new staged file.

## Durability and recovery

SQLite uses WAL, foreign keys, full synchronous commits, and a busy timeout. Jobs, attempts,
checkpoints, progress events, and abandoned staging resources are separate tables. A checkpoint is
saved only after the sink reports that its staging batch is committed. On daemon start, jobs left in
RUNNING or PUBLISHING become INTERRUPTED.

`FileSource::resume` validates a fingerprint and replays to an exact logical row offset. Committed
sink batches are individually finalized as Parquet chunks under a job-scoped staging directory.
After restart, the daemon verifies the staging manifest and contiguous chunk sequence, discards only
recognized uncommitted tail chunks, continues global batch numbering, and publishes all chunks once.
Deleting a file job removes its staging directory only after the manifest matches the job's target,
format, and write mode and every contained entry is a recognized chunk resource.
PostgreSQL sink batches carry a private batch-sequence column. On resume, rows newer than SQLite's
last checkpoint are trimmed only after the table's job marker is verified, closing the database/SQLite
commit window without duplicates. File sources then continue into that stage. Structured PostgreSQL
table sources can resume from non-null, unique, ordered keys: checkpoint values are parameterized via
a temporary bounds table and an initial upper bound excludes later inserts. Sources without keys and
explicit user SQL restart private staging from zero; ELM never rewrites explicit SQL.

## IPC

Protocol version 1 is newline-delimited JSON on a Windows named pipe or Unix-domain socket. A
random 244-bit token is created in the per-user data directory; Unix token/socket permissions are
`0600`. The token is compared through fixed-size BLAKE3 digests. Requests are capped at 1 MiB.
`job.watch` keeps the connection open and streams progress events. Public errors are categorized,
sanitized, and may include remediation, but never internal connection strings.

Runtime memory and batch presets use the same IPC boundary and are validated before SQLite upsert.
The desktop transfer wizard snapshots the current preset into every submitted job, so later settings
changes cannot alter an existing or resumed attempt.
