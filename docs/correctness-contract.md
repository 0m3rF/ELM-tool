# v2 correctness contract

This contract defines the required stable-v2 behavior. A connector cannot override it with a
convenience fallback. The alpha implements the documented type subsets for file, PostgreSQL,
MySQL, SQL Server, and Oracle transfers. Consult the connector matrix for current limits.

## Write modes

| Mode | Destination absent | Destination present | Empty source |
| --- | --- | --- | --- |
| `append` | Create through staging | Stage, validate, then insert in one publication transaction | No visible change |
| `replace` | Create through staging | Preserve old target until a validated staging target is swapped | Publish an empty target with the source schema |
| `fail` | Create through staging | Fail before loading | Publish an empty target with the source schema |

In atomic mode, cancellation or failure before publication leaves the original destination visible.
Abandoned staging objects are recorded for recovery cleanup. If privileges do not permit staging and
safe publication, preflight fails. `checkpointed` consistency permits partial committed visibility
only when the job explicitly requests it.

Oracle existing-table REPLACE has a separate explicit `table_swap` mode. It stages and validates
the replacement, journals object identities, renames the original to a backup, then renames the
stage to the target. These renames are **not atomic**: the target name may be temporarily absent.
Recovery finishes a validated swap without replaying its source. Unexpected identities stop
recovery. Atomic mode never falls back to this strategy. Existing indexes, grants, and constraints
remain on the backup; detected dependencies that would continue referencing the old object reject
the swap. Backups are retained until job deletion. Exclusive control of target DDL is required.

## Schema and values

- Canonical interchange uses Arrow fields and arrays. Nullability, decimal precision/scale,
  timestamp unit/time zone, and binary/string distinctions are preserved.
- A target is validated before source loading begins. Name matching is exact after connector
  metadata normalization.
- Unsupported or lossy mappings fail preflight. `allow_lossy` must appear on an explicit per-column
  conversion rule; no global permissive switch exists.
- Conversion target names accept common scalar aliases or Arrow's canonical `DataType` syntax.
  Lossless widening is accepted automatically; any other Arrow-supported cast requires
  `allow_lossy` on that column. Failed values abort the batch and never silently become null.
- CSV and NDJSON readers sample at most 10,000 rows for schema inference and continue streaming.
  Parquet reads its embedded Arrow schema.
- JSON output is newline-delimited. A full JSON array is never assembled.
- Malformed rows fail the attempt with their record position. They are never silently skipped.

## Identifiers and SQL

Catalog, schema, table, and column identifiers are structured values. PostgreSQL/Oracle use doubled
double quotes, MySQL uses doubled backticks, and SQL Server uses doubled closing brackets. Resume
values use driver parameters (`$n`, `?`, or `:n`). User SQL is executed as an explicit source and is
not concatenated with identifiers, ordering, or checkpoint predicates.

## Cancellation and resume

Cancellation is cooperative at source reads, channel sends, transforms, sink writes, checkpoints,
and publication boundaries. A native PostgreSQL COPY must be aborted before its connection is
returned to a pool. Oracle blocking calls live on dedicated workers and observe cancellation between
batch DML calls.

File checkpoints bind a committed logical record offset to a file fingerprint. A changed file is a
conflict. A database table can resume in place only with user-supplied non-null unique ordered keys;
the initial maximum key bounds the source. This protects against duplicates but is not a database
snapshot. All other database sources restart the staging load from zero.

## Masking and errors

`star`, `star_length`, `random`, and `nullify` operate on Arrow arrays. Random output derives from the
job seed, batch sequence, column, and row, making a retried committed range deterministic. `nullify`
retains the original Arrow type.

Persisted and displayed errors redact URL credentials and password/token/secret fields. Job specs,
logs, events, and exports contain credential references only. Passwords exist solely in the native
keychain and in connector memory for the duration of connection establishment.
