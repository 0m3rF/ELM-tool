# Connector matrix

## Implementation status

| Connector | Source path | Sink path | Runtime prerequisite | Alpha status |
| --- | --- | --- | --- | --- |
| CSV | Streaming Arrow CSV reader | Recoverable chunks, sibling staged Arrow CSV writer | None | Implemented |
| NDJSON | Streaming line reader | Recoverable chunks, staged line-delimited writer | None | Implemented |
| Parquet | RecordBatch reader | Recoverable chunks, staged ArrowWriter | None | Implemented |
| PostgreSQL | Binary COPY OUT | Per-batch binary COPY into marked staging; transactional publish | None | Implemented alpha subset |
| MySQL | Prepared binary protocol, request-driven batches | In-memory `LOAD DATA LOCAL`, announced prepared fallback | InnoDB; LOCAL preferred | Implemented alpha subset |
| SQL Server | Typed ODBC block fetch | Columnar atomic staging sink | Microsoft ODBC Driver 18 | Alpha source/sink enabled; Linux native fixtures pass; limited types |
| Oracle | Byte-bounded native array reader | Batch DML staging | Oracle Instant Client | Experimental; atomic APPEND/new tables; existing REPLACE requires explicit non-atomic table swap |

`RecoverableFileSink` writes each batch as its own staged chunk file, fsyncs it, then renames it
into place, so a chunk that never finishes never reaches the destination. Live-tested against a
genuine 4 MiB `tmpfs` (`tests/full-disk/`, `crates/elm-connectors/tests/full_disk.rs`): staging
writes fail closed with a clean `ElmError::Io` when the filesystem is actually exhausted, and no
partial destination file is ever produced. See `docs/operations.md`.

The alpha daemon executes PostgreSQL-to-file, file-to-PostgreSQL, and PostgreSQL-to-PostgreSQL jobs.
It performs a live keychain-backed connection diagnostic, discovers the Arrow schema before loading,
finishes one binary COPY per committed staging batch, and publishes APPEND/REPLACE/FAIL atomically.
Staging tables carry a job marker; resume and deletion refuse to touch a relation without the exact
marker. Live PostgreSQL 16 tests cover Unicode, nulls, binary data, exact decimals (including
negative scales), dates, zoned timestamps, all write modes, empty REPLACE, target invisibility before
publication, crash-window staging trim, committed-stage resume, and upper-bounded keyset resume.
A role granted no privilege in the destination schema (PostgreSQL 15+'s default for a fresh,
non-superuser role) is live-tested to fail `preflight()` with `PermissionDenied` rather than
reaching staging DDL.

The current PostgreSQL subset is intentionally narrow:

- Lossless mappings are BOOLEAN, SMALLINT/INTEGER/BIGINT, REAL/DOUBLE PRECISION,
  constrained NUMERIC/DECIMAL with precision up to 38 and an Arrow-valid scale,
  TEXT/VARCHAR/CHAR/NAME, BYTEA, DATE, TIMESTAMP, and TIMESTAMPTZ. NUMERIC values use PostgreSQL's
  binary base-10000 representation and Arrow Decimal128 coefficients; they are never converted
  through floating point or text.
- Unconstrained NUMERIC, precision above 38, scale greater than precision, UUID, JSON, arrays,
  enums/domains, and intervals fail preflight; no text or row-by-row fallback is selected.
  TEXT and BYTEA have no declared maximum and stream without a size check: every connector
  reports `supports_lob_spill: false`, and `elm-engine`'s `SpillStore` (a disk-backed store for
  values too large to hold in the batch memory budget) exists but is not wired into any
  connector or the pipeline. An oversized TEXT/BYTEA value is bounded only by the ambient
  per-batch memory budget, not by a dedicated preflight or LOB-spill check.
- Structured PostgreSQL table sources can resume with user-supplied non-null, unique, ordered keys.
  The initial upper key is captured, checkpoint values are bound as parameters through a temporary
  bounds table, and rows added above that bound are excluded. Explicit SQL and sources without keys
  restart private staging from zero. `checkpointed` publication is not implemented.
- `ssl_mode=require` is the default and verifies the host and certificate chain with the native OS
  trust store or an optional PEM `root_certificate`. Plaintext requires explicit
  `ssl_mode=disable` and produces a job warning; no TLS failure is downgraded.

MySQL sources and sinks are enabled, including database/file transfers. Signed and unsigned integers,
floating point, constrained Decimal128, Unicode text, binary values, dates, and microsecond timestamps
have canonical mappings. ENUM/SET and unsupported types require explicit source conversion.
TINYINT(1), including BOOLEAN aliases, remains an integer because MySQL does not constrain it to 0/1.
TLS verifies host and certificate chain by default; plaintext requires explicit `ssl_mode=disable`.
Naive Arrow timestamps map to DATETIME(6) with no range restriction; timezone-bearing Arrow
timestamps map to native TIMESTAMP(6), whose documented range (1970-01-01 00:00:01 through
2038-01-19 03:14:07.999999 UTC) is narrower. The sink validates every row of a batch against
that range before attempting to write it, and fails closed with a typed error rather than
letting the server reject or reinterpret an out-of-range value; this is a per-batch check, not a
per-row error surfaced from inside the LOCAL INFILE stream, because a row-level error raised
while streaming that protocol is not preserved as a typed error and would otherwise arrive as a
generic I/O failure.
TEXT/BLOB columns of any declared MySQL width map to LONGTEXT/LONGBLOB and have no enforced
maximum size: unlike SQL Server (4000 UTF-16 units / 8000 bytes) and Oracle (NVARCHAR2(2000) /
RAW(2000)), which must declare a fixed native buffer size upfront, MySQL and PostgreSQL text and
binary values stream without a declared bound and are limited only by the ambient per-batch
memory budget.

MySQL loads marked InnoDB staging tables and validates row counts and conversion warnings before
publication. APPEND uses a transaction; REPLACE uses a multi-table rename. When LOCAL is disabled at
preflight, progress includes a prepared-insert fallback warning. A later policy change fails the
attempt so a new preflight can report the selected path. File checkpoints reconcile the private
batch sequence; MySQL source queries restart from zero. Resume keys are out of scope for MySQL,
SQL Server, and Oracle sources in this release — a deliberate scope decision, not a placeholder —
and the daemon rejects a supplied key for any of them with `Unsupported` before credential lookup
or a native connection; only PostgreSQL's keyset resume is implemented. `elm-daemon`'s
`validate_source_resume` is unit-tested against all four database kinds.

A durable MySQL publication receipt prevents replay after an uncertain commit or daemon crash.
APPEND records completion in its publication transaction. REPLACE records completion after rename;
if that update was interrupted, the target's job marker and missing stage prove the rename completed.
On resume, the daemon confirms publication before opening the source and records success without
copying again. An inconclusive or legacy fence still requires manual destination reconciliation.
Receipts remain until job deletion. A user granted `SELECT` but not `CREATE` on the destination
database can preflight successfully (MySQL has no proactive privilege check like PostgreSQL's
and SQL Server's) but is live-tested to fail `begin()` with `PermissionDenied` when staging table
creation is denied. Broader fault-injection, backup cleanup recovery, memory-budget accounting
for encoded buffers, and performance gates remain outstanding.
SQL Server source execution is wired to the daemon for file, PostgreSQL, MySQL, and SQL Server
destinations, subject to each sink's supported types.
The experimental source maps BIT, TINYINT (unsigned), SMALLINT, INTEGER, BIGINT, REAL/FLOAT,
bounded CHAR/VARCHAR/NCHAR/NVARCHAR, bounded BINARY/VARBINARY, DECIMAL/NUMERIC up to precision 38,
DATE, and timestamps with up to six fractional digits. Decimal coefficients are parsed from the
driver's exact character representation without floating point. Dates and timestamps use typed
ODBC buffers; invalid values or sub-microsecond precision fail conversion. UTF-16 decoding rejects
invalid surrogates, and block fetching checks for truncation. Declared LOBs, TIME, DATETIMEOFFSET,
timestamp precision above six, and native/custom types fail schema discovery. Source resume keys are unavailable: retry restarts
the private load from zero. Oracle publication and recovery are described below.

The SQL Server sink is wired to the daemon, with native parameter-array inserts,
job-marked staging, checkpoint sequence reconciliation, and transactional publication receipts.
APPEND validates the schema, rejects enabled triggers and IGNORE_DUP_KEY indexes, and checks target
row counts under an exclusive publication lock. REPLACE drops the old target and renames staging
inside one transaction; dependencies that prevent dropping the target make publication fail.
Empty REPLACE and FAIL modes have integration fixtures. CREATE TABLE and schema ALTER privileges
are checked before staging; a live-tested fixture confirms a login granted neither fails
`preflight()` with `PermissionDenied` rather than reaching staging DDL. Receipt and staging
cleanup refuse unrecognized job markers.

This sink currently supports boolean, UInt8, Int16/32/64, finite floats, Decimal128 with precision
1..38 and scale 0..precision, Date32, timezone-free microsecond timestamps, text up to 4000 UTF-16
units, and binary up to 8000 bytes. Decimal
coefficients are validated before a batch transaction and encoded as exact fixed-point ASCII in
bounded parameter-array buffers, with no floating-point intermediate. Negative scale, scale above
precision, and out-of-precision values fail explicitly. Dates map to DATE and timestamps to
DATETIME2(6), using typed native buffers. Values outside years 0001..9999 fail before the staging
batch transaction; no timezone is assigned or discarded. Existing timestamp destinations must
also be DATETIME2(6): DATETIME, SMALLDATETIME, and lower-precision DATETIME2 are rejected before
APPEND to prevent rounding. Existing text and binary targets must be NVARCHAR(4000) and
VARBINARY(8000), respectively. Non-Unicode, fixed-width, and narrower targets fail preflight:
the canonical Arrow schema alone cannot detect character loss, padding, or truncation risks.
Timezone-bearing timestamps, other timestamp units, Date64, TIME,
wider unsigned, and LOB sink types remain unsupported without explicit conversion.
Decimal sink encoding has unit coverage across every supported precision/scale.
Live native tests additionally cover positive/negative 38-digit coefficients, scales 0, 4, and 38,
nulls, and rejection of an out-of-precision batch followed by a successful valid write. Temporal
fixtures cover nulls, year 0001/year 9999 boundaries, pre-epoch microseconds, leap-day values, and
out-of-range rejection followed by a valid batch.
Insertion buffers are reused across native chunks within an Arrow batch, with a 2 MiB buffer limit.
The daemon exposes this alpha sink and checks its publication receipt before reopening a source on
retry. Confirmed commits complete without replay; job deletion removes marked staging and receipts.
SQL text, identifiers, and connection attributes use UTF-16 ODBC APIs on all platforms, independent
of the process locale. Destination names that alias private staging resources are rejected.

One blocking worker owns the SQL Server connection and reusable columnar buffers. Each request
fetches one bounded block; capacity reserves space for native and decoded buffers and caps native
block allocation. Cancellation closes the request channel and checks a stop flag between native
calls; an in-flight call relies on the driver's 15-second query timeout. This is not immediate
native cancellation. Unit tests cover Arrow conversion and sizing. On 2026-09-10, all five live
integration tests passed against SQL Server 2022 CU20 on Linux with Driver 18.7.1.1, including exact
Unicode/NUL/binary/decimal/date/timestamp multi-chunk round trips and lossy APPEND-target rejection.
Windows/macOS live tests, verified-TLS integration,
comprehensive privilege/fault tests, performance, and cross-database acceptance remain pending.

The experimental Oracle source is wired to daemon jobs for all destinations, subject to sink
type compatibility. Jobs persist and broadcast a warning that native array throughput is not
performance-qualified. Retries discard the source row-count
checkpoint and restart private staging from zero; supplied resume keys fail before credential
lookup or native connection. Confirmed destination publication receipts are still reconciled
before any source is reopened. One
blocking worker owns all native handles and fetches only on batch requests. Supported mappings are
bounded character/RAW values, BINARY_FLOAT/BINARY_DOUBLE, declared NUMBER with precision 1..38 and
scale -38..precision, DATE, and TIMESTAMP with at most six fractional digits. NUMBER conversion
parses exact decimal/scientific text without a floating-point intermediate or rounding. Oracle
DATE includes time of day and therefore maps to a timezone-free microsecond timestamp, not Date32.
Unconstrained NUMBER, Oracle FLOAT, LOBs, LONG, timezones, intervals, and custom types fail schema
discovery. Duplicate column aliases are rejected. Oracle keyset resume is not implemented.

The reader prepares and verifies SELECT, then uses bound SQL with `DBMS_SQL` to describe the
cursor before execution. It executes once, rechecks metadata, and transfers that same cursor to
a byte-sized native REF CURSOR array. `DBMS_SQL` execution privileges are required; there is no
one-row fallback. Prefetch is disabled. Locator mode prevents eager LOB materialization; see the
[rust-oracle statement builder](https://docs.rs/oracle/latest/oracle/struct.StatementBuilder.html).
Arrow batches reserve space conservatively from declared widths and are capped at 8 MiB.
The native array reserves at most 2 MiB from conservative declared widths. This is not a
verified process-RSS or throughput guarantee.
Cancellation checks occur between native calls; in-flight calls rely on the configured native call
timeout (15 seconds by default). Cancelling an individual batch future also closes the worker channel and permanently
invalidates that source instance, including when its fetched reply was delivered but not consumed.
Subsequent fetches fail instead of skipping unobserved rows; checkpoints count only batches returned
to the caller. Failed requests likewise invalidate the cursor. Deterministic channel tests cover
both cancellation windows, normal completion, and conversion failure without requiring a client.
Unit tests cover exact decimals across supported precision/scale combinations, temporal
boundaries, metadata rejection, identifier quoting, and batch sizing. On 2026-09-15 (Europe/Istanbul),
all eleven live connector tests and the daemon end-to-end test passed on Linux against Oracle XE 21
with Instant Client Basic 21.23. Coverage includes engine-driven CSV/NDJSON/Parquet publication,
exact sink readback, atomic APPEND, duplicate-key rollback, empty table-swap REPLACE, and recovery
between renames, rollback of inconsistent checkpoint trimming, replay without duplicate rows,
and restoration of the original table when deleting an interrupted swap.
The daemon test uses authenticated IPC and a real native Secret Service keychain,
then transfers Oracle-to-Oracle, replaces the table, exports Parquet, and reconnects to history.
Windows/macOS native acceptance, verified TLS, large-workload tuning, and RSS measurements remain pending.

Oracle sinks use batch DML on a dedicated worker. Supported destination values are Decimal128,
binary floats, timezone-free microsecond timestamps, strings fitting NVARCHAR2(2000), and binary
values fitting RAW(2000). Empty strings/binary fail because Oracle would turn them into NULL.
Other types require explicit conversions. There is no native integer (Int8/16/32/64,
UInt8/16/32/64) mapping: a plain integer column from a CSV/NDJSON file — which Arrow schema
inference always types as Int64, never Decimal128 — needs an explicit `ConversionRule` to
Decimal128 or a float type before it can reach an Oracle sink; a database source with a properly
typed NUMBER column is unaffected, since that already arrives as Decimal128. Writes are
restricted to the connected user's schema.
A staging operation that fails with ORA-01031 (insufficient privileges) is reported as
`PermissionDenied` rather than a generic connection error, matching PostgreSQL's and SQL Server's
own proactive privilege checks; a live-tested fixture connects as a user granted only
CREATE SESSION and confirms `begin()` fails this way when it attempts to create the staging table.
An invisible `__elm_batch` column (default 0) remains on newly published tables for recovery;
ordinary `SELECT *` does not expose it. Checkpoints verify the source schema hash and staging
row count; uncheckpointed batches are trimmed before resuming file sources.

APPEND to an existing compatible target inserts the validated staging rows and publication
receipt in one transaction. Targets with triggers, generated/identity columns, or mismatched
canonical schemas are rejected. Publishing a new target uses a single rename.

Existing-table REPLACE is **not atomic** and requires `--consistency table-swap` (JSON:
`"table_swap"`). The original is renamed to a job-specific backup, then staging takes the target
name. The name can be unavailable between renames. Keep exclusive control of target DDL during
publication. Existing indexes, grants, and constraints stay with the backup, not the replacement;
detected incoming foreign keys and dependent objects cause preflight rejection. There is no
automatic downgrade from atomic mode and no DELETE-and-reinsert replacement path.

The journal stores original/staged object IDs. Resume completes a validated interrupted swap
before opening its source; confirmed receipts prevent replay. Unexpected identities fail closed.
Backups remain until job deletion. Deleting an interrupted job restores a renamed original before
cleaning staging; deleting a successfully published job removes its backup, not its current target.
Partial DDL setup without a valid journal requires manual inspection, not speculative deletion.

Compile-time connector features are `postgresql`, `mysql`, `sql-server`, `oracle`, and
`all-databases`. The daemon enables all four features for connection diagnostics and source
and destination execution. Oracle and SQL Server client libraries/drivers
remain external. Linux/macOS builds and daemon installations also require the unixODBC manager.

## Physical source/target identity discovery

Before running a database-to-database job, the daemon resolves each side's physical identity
through `elm_connectors::physical_identity::resolve` and warns, rather than blocks, when the
source and destination turn out to be the same physical table. Resolution unwinds exactly one
level of PostgreSQL/MySQL view, or Oracle/SQL Server view or synonym, indirection through catalog
metadata alone; it never parses view SQL text. It also compares a low-privilege, server-native
fingerprint — PostgreSQL `pg_control_system()`'s `system_identifier`, MySQL `@@server_uuid`, SQL
Server `SERVERPROPERTY('ServerName')` paired with the target database's creation timestamp, and
Oracle `SYS_CONTEXT('USERENV','DB_UNIQUE_NAME')`/`INSTANCE_NAME` — so two saved environment
records that reach the same physical server through different configured hostnames are still
recognized as the same server. This is not DNS resolution and never attempts any.

Anything ambiguous resolves to unknown, never to a guessed match or mismatch: a relation that does
not exist, a view depending on more than one base table, an Oracle synonym across a database
link, or a connected role that cannot read the server fingerprint. A raw SQL source (rather than a
selected table) is never compared, since it has no single relation identity to resolve. This
warning changes nothing about how a transfer runs: staged publication already isolates the copied
rows from the destination before the destination is touched, independent of whether source and
sink are the same table, so the warning exists purely to make that condition visible to the job's
own progress and history. Live fixtures cover view resolution on PostgreSQL and MySQL, view and
synonym resolution on SQL Server and Oracle, an alternate-hostname PostgreSQL environment record
resolving to the same server, and correct rejection of unrelated and missing relations; see
`tests/cross-database/README.md`.

## Native prerequisite diagnostics

- Windows SQL Server: install Microsoft ODBC Driver 18 and confirm it appears in the ODBC Data
  Sources driver list.
- Linux/macOS SQL Server: install `msodbcsql18` plus unixODBC and ensure the driver is registered.
- Oracle: install a supported Instant Client Basic package. On Windows place its directory on PATH;
  on Linux configure the dynamic loader; on macOS follow Oracle's signed-library guidance.

Connection testing checks the registered SQL Server Driver 18 or loads Oracle Instant Client before
attempting a login. Native calls run on blocking workers. SQL Server sets a 15-second login timeout;
Oracle sets connect/transport timeouts and a configurable native round-trip timeout (default 15 seconds). Driver timeout enforcement is
native-client-dependent. Diagnostic failures never include native error text or credentials.

SQL Server uses SQL authentication with `Encrypt=yes;TrustServerCertificate=no` by default. Oracle
interprets Database as a service name and uses TCPS with server DN matching by default. Configure
Oracle wallet/trust through the installed client. Both accept explicit `ssl_mode=disable`; the
custom root-certificate field remains limited to PostgreSQL/MySQL. ELM does not bundle or download
proprietary clients. These settings follow the [Microsoft connection-string reference](https://learn.microsoft.com/en-us/sql/connect/odbc/dsn-connection-string-attribute)
and [Oracle Net descriptor reference](https://docs.oracle.com/en/database/oracle/oracle-database/19/netrf/local-naming-parameters-in-tns-ora-file.html).

The Oracle fixtures create uniquely named test tables and exercise source, sink, swap recovery,
and daemon transfers. They require Instant Client and an isolated Oracle service with CREATE TABLE,
own-schema DDL/DML, DUAL, and DBMS_SQL privileges. Daemon tests also need an unlocked OS keychain:

```powershell
$env:ELM_TEST_ORACLE_PASSWORD = "..."
# Optional: ELM_TEST_ORACLE_HOST, PORT, DATABASE (service), USER, SSL_MODE.
# Defaults: localhost, 1521, XEPDB1, elm_test, require.
cargo test -p elm-connectors --features oracle --test oracle -- --ignored --test-threads=1
```

Use explicit SSL_MODE=disable only for a disposable isolated plaintext fixture. Tests cover Unicode,
RAW, exact decimals, negative scale, DATE time-of-day, pre-epoch timestamps, floats, nulls, repeated
EOF, unsupported-type rejection, empty-result schema preservation, sanitized native failures,
and 1,025 ordered rows across small requested batches. No fixture credentials are stored in the
repository. Engine-level fixtures additionally exercise recoverable CSV/NDJSON/Parquet exports,
ordered IDs and Unicode readback, empty REPLACE, FAIL against an existing destination, and an
injected checkpoint-store failure after a staging batch commits. They assert that the visible
destination remains unchanged at every checkpoint and after failure. These use the real source
and sink through the engine. A separate daemon fixture covers IPC and the native keychain.
The Oracle integration workflow runs these tests on relevant pull requests and manual
dispatch; remote CI execution has not been verified locally.

For Linux x86-64 Docker, `tests/oracle/compose.yml` provides a disposable Oracle XE 21 fixture and
a Rust test container with Instant Client Basic 21.23, unixODBC, and a disposable D-Bus/Secret
Service session for the daemon test. The client archive is SHA-256 checked against
[Oracle's published download](https://www.oracle.com/database/technologies/instant-client/linux-x86-64-downloads.html).
This development-only image downloads proprietary software; review its included license before use.
The database uses the image's [application-user and health-check configuration](https://github.com/gvenzl/oci-oracle-xe#how-to-use-this-image).
No host ports are exposed, the repository is mounted read-only, and the test client is not bundled
with ELM. Database files use a dedicated disposable volume; diagnostic/audit directories use
size-limited temporary memory filesystems to avoid overlay journal stalls during native startup.
Those diagnostic files disappear when the container stops. Never attach this fixture to a
production network. Generated credentials stay below Oracle XE 21's 30-byte password limit.

```powershell
$env:ELM_TEST_ORACLE_PASSWORD = "Elm-" + [guid]::NewGuid().ToString("N").Substring(0, 24) + "!"
docker compose -p elm-oracle-acceptance -f tests/oracle/compose.yml up --build --abort-on-container-exit --exit-code-from tests
# Capture the preceding exit code before cleanup when scripting.
docker compose -p elm-oracle-acceptance -f tests/oracle/compose.yml down --volumes
```

Cleanup removes the disposable database and Rust build/registry volumes, not repository files.

Run the ignored PostgreSQL integration suite against an isolated test instance with database
`elm_test`, user `postgres`, and the configured test password:

```powershell
$env:ELM_TEST_POSTGRES_PORT = "55432"
$env:ELM_TEST_POSTGRES_PASSWORD = "..."
cargo test -p elm-connectors --features postgresql --test postgresql -- --ignored
```

Run MySQL tests against an isolated MySQL 8.4 instance (`elm_test`, user `root`). Run serially because
the fallback test temporarily changes the server-wide LOCAL policy:

```powershell
$env:ELM_TEST_MYSQL_PORT = "53306"
$env:ELM_TEST_MYSQL_PASSWORD = "..."
cargo test -p elm-connectors --features mysql --test mysql -- --ignored --test-threads=1
```

The SQL Server source test executes a constant SELECT. The sink test creates and removes a uniquely
named `dbo.elm_sink_test_*` fixture, and requires CREATE TABLE and schema ALTER in an isolated test
database. Both require Driver 18 and a test login:

```powershell
$env:ELM_TEST_SQL_SERVER_PASSWORD = "..."
# Optional: ELM_TEST_SQL_SERVER_HOST, PORT, DATABASE, USER, SSL_MODE.
# Defaults: localhost, 1433, elm_test, sa, require.
cargo test -p elm-connectors --features sql-server --test sql_server -- --ignored
```

Alternatively, run the same tests entirely in Linux Docker containers, without installing a host
driver. The development image installs Driver 18 using [Microsoft's Debian instructions](https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server).
This fixture accepts the SQL Server Developer and ODBC client EULAs, publishes no host ports,
mounts the repository read-only, and uses only disposable generated test tables. Use a fresh
Compose project and never connect this fixture to a production network:

```powershell
$env:ELM_TEST_SQL_SERVER_PASSWORD = "Elm-test-" + [guid]::NewGuid().ToString("N") + "!"
docker compose -p elm-sql-acceptance -f tests/sql-server/compose.yml up --build --abort-on-container-exit --exit-code-from tests
# Capture the preceding exit code before cleanup when scripting.
docker compose -p elm-sql-acceptance -f tests/sql-server/compose.yml down --volumes
```

Cleanup removes the fixture database and Rust build/registry volumes, not repository files.
The first run downloads a Rust image and dependencies. These tests cover typed source values,
all three publication modes, empty REPLACE, duplicate-key rollback, trimming an uncheckpointed
batch on resume, publication receipts, and exact multi-chunk round trips for supported sink types.
The SQL Server integration workflow runs this fixture on relevant pull requests and manual dispatch.
