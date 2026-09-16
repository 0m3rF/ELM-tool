# Cross-database acceptance fixtures

The reusable suite covers all 16 ordered pairs of PostgreSQL 16, MySQL 8.4, Oracle XE 21,
and SQL Server 2022 using the real transfer engine, including four same-database pairs.
It checks Unicode, binary bytes, exact decimal scale/value, microsecond timestamps, NULLs,
new-target REPLACE and FAIL, staged APPEND, FAIL preserving an existing target, populated REPLACE,
and empty-source REPLACE preserving canonical types.

Pairs involving Oracle explicitly cast IDs to DECIMAL/NUMBER(18,0); other pairs retain
Int64 coverage. This reflects Oracle's current supported mappings and does not establish
automatic integer conversion or full type coverage.
For each Oracle destination, atomic REPLACE of an existing target must return Unsupported
and preserve its rows. The subsequent REPLACE explicitly selects the **non-atomic table swap**.
Crash recovery and dependency handling remain covered by the separate Oracle connector suite.

Four `same_table_*` tests copy a table onto itself. Each exercises direct table
selection, schema-qualified selection (also catalog-qualified on SQL Server), and a SQL
table alias. Qualified and query cases use distinct environment IDs for the same database.
They require populated REPLACE to retain all rows, APPEND to double the original rows exactly,
FAIL to preserve existing data, and an empty query to publish an empty schema. A checkpoint
observer reads the target through a separate connection and requires its original values
to remain visible until publication. Oracle REPLACE explicitly uses non-atomic table swap.
Four `interrupted_same_table_*` tests inject failure after a private staging batch commits but
before its checkpoint is persisted. For APPEND and REPLACE on each database, the public target
must retain its original values after failure. A retry with the same job identity restarts the
unkeyed database source from row zero, trims or recreates its private stage, and publishes once
without duplicate rows. These tests model the engine and connector recovery boundary; they do
not terminate and restart the daemon process. Concurrent writers remain unverified.

Physical source/target identity discovery is implemented as a best-effort diagnostic warning,
never a blocking check: `elm_connectors::physical_identity::resolve` unwinds exactly one level of
PostgreSQL/MySQL view, or Oracle/SQL Server view or synonym, indirection through catalog metadata
(never by parsing view SQL text), and pairs the resolved base relation with a low-privilege,
server-native fingerprint (PostgreSQL `pg_control_system()`, MySQL `@@server_uuid`, SQL Server
`SERVERPROPERTY('ServerName')` plus the database's creation timestamp, Oracle
`SYS_CONTEXT('USERENV','DB_UNIQUE_NAME')`/`INSTANCE_NAME`) so two environment records that reach
the same physical server through different configured hostnames are also recognized as the same
table, without resolving DNS. Any ambiguity — ownership mismatches, a view depending on more than
one base table, a synonym across a database link, or insufficient catalog privilege — resolves to
"unknown" rather than a guess, and unknown is never treated as "different." A daemon job whose
source and destination resolve to the same physical table gets a warning naming both sides; the
transfer's own safety already comes from staging the source fully before the destination is
touched (see the `same_table` fixtures above), so this never blocks or changes a transfer.

## Verified evidence

On 2026-09-16, the focused `same_table` run passed **8 tests, 0 failures, 0 ignored**
in 228.79 seconds, covering fresh and interrupted same-table behavior on all four databases.
The complete matrix run is recorded separately below.

On 2026-09-16, one live `physical_identity_*` integration test per database passed against each
database's own disposable fixture: PostgreSQL (view resolution plus a second environment record
reaching the same server through `localhost` instead of `127.0.0.1`), MySQL (view resolution via
`information_schema.view_table_usage`), and SQL Server and Oracle (view and synonym resolution,
run inside `tests/sql-server/compose.yml` and `tests/oracle/compose.yml` alongside every other
ignored test in each connector's suite, all of which continued to pass). Each test also asserted
that an unrelated table and a missing relation are correctly rejected or reported unknown.

On 2026-09-16, `./tests/cross-database/run-native.ps1` then passed **24 tests, 0 failures,
0 ignored** in 371.88 seconds of test execution on Linux x86-64 containers under Docker
Desktop on Windows. Startup and cold compilation time are additional. The runner completed
fixture and volume cleanup successfully.

| Source → target | PostgreSQL | MySQL | Oracle | SQL Server |
| --- | --- | --- | --- | --- |
| PostgreSQL | Pass | Pass | Pass | Pass |
| MySQL | Pass | Pass | Pass | Pass |
| Oracle | Pass | Pass | Pass | Pass |
| SQL Server | Pass | Pass | Pass | Pass |

Windows compilation and Clippy with warnings denied also passed for the full matrix.
The broader local core, engine, and connector tests passed; live database tests in that
broader command were ignored and are separate from the 24 live results above.
This is not Windows/macOS native-driver acceptance, remote CI evidence, or a performance gate.

PostgreSQL/MySQL fixtures bind only loopback ports 55441 and 53341. Use a generated disposable `ELM_MATRIX_PASSWORD`.
Never substitute production databases. The ignored tests create uniquely named tables in the fixture databases;
fixture teardown removes their journals and data. Tests do not read saved ELM connections or keychains.

## PostgreSQL/MySQL subset (four pairs and two same-table tests)

From the repository root:

```powershell
$env:ELM_MATRIX_PASSWORD = 'Elm-' + [Guid]::NewGuid().ToString('N')
docker compose -p elm-cross-acceptance -f tests/cross-database/compose.yml up -d --wait --wait-timeout 180
cargo test -p elm-connectors --features postgresql,mysql --test cross_database -- --ignored --test-threads=1
docker compose -p elm-cross-acceptance -f tests/cross-database/compose.yml down --volumes
```

## Full native-driver matrix (16 pairs and four same-table tests)

Requires Docker with Linux x86-64 containers and enough memory for all four databases plus
the Rust build. Oracle Instant Client and Microsoft ODBC Driver 18 are installed only in
the test image. Oracle and SQL Server expose no host ports; tests use the private Compose
network. SQL Server uses its disposable `master` database; Oracle uses the `elm_matrix` app user.

From the repository root, in PowerShell (also available as `pwsh` on Linux):

```powershell
./tests/cross-database/run-native.ps1
```

The runner generates an Oracle-compatible password, builds the native-client image, compiles
before database startup to reduce peak memory, waits for database health,
runs the eight focused same-table tests followed by all 24 tests sequentially, and removes
the named fixtures and volumes in a
`finally` block. The image download needs access to Oracle and Microsoft package servers.
The GitHub workflow runs this full matrix on manual dispatch; remote execution remains
unverified. Pull requests run the PostgreSQL/MySQL subset, including its two same-table tests.

If the runner is forcibly terminated, remove only its fixtures:

```powershell
$env:ELM_MATRIX_PASSWORD = 'cleanup-placeholder'
docker compose -p elm-cross-native -f tests/cross-database/compose.yml -f tests/cross-database/native.yml down --volumes
Remove-Item Env:ELM_MATRIX_PASSWORD
```

Always tear down the named fixture after testing, including after a failure. Do not use broad
Docker cleanup/prune commands; other projects may share the Docker daemon. Keep the generated
password out of reports.
