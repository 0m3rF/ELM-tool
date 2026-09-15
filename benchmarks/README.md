# Performance gates

Run baseline and v2 candidates on the same dedicated host. Record CPU model/count, RAM, storage,
database/server versions, client drivers, network path, schema digest, generator digest, command,
wall time, rows, bytes, throughput, and peak RSS in `benchmark-results/`.

Required gates:

- each fixed 10 GB PostgreSQL, Oracle, MySQL, and SQL Server self-copy reaches at least 2× the frozen
  Python v1 throughput;
- peak process RSS stays at or below 512 MiB plus the connector's documented native-client allowance;
- after a Rust connector baseline is accepted, it may not regress by more than 10%;
- a 100 GB soak and interrupt/resume run completes without unbounded memory, duplicates, or visible
  target corruption.

The repository intentionally contains no invented baseline numbers. `baseline-template.json` must be
filled from actual dedicated-host runs and reviewed before a stable connector is enabled.

## Check a measured report

Run `cargo run -p elm-cli --example check_benchmarks -- benchmark-results/candidate.json`.
For subsequent candidates, add `--previous benchmark-results/accepted-rust.json` to enforce
the 10% regression limit. Without that argument, the checker explicitly reports regression
as not evaluated. Reports are read-only; this command never connects to a database.

The checker requires schema version 1, an RFC3339 `captured_at`, nonempty host/dataset provenance,
exactly one result for each supported connector, the fixed 10,000,000,000-byte dataset,
positive finite throughput, and measured nonzero peak RSS. Null template values fail.
The 2× Python and 512 MiB gates are inclusive. Previous baselines must themselves pass,
use identical host/dataset metadata, retain the frozen Python rates and memory allowances,
and not postdate the candidate.

Each result can optionally include `native_client_allowance_bytes` (defaults to zero) and
`native_client_allowance_evidence` (a nonempty reference to reviewed measurements, required
for any nonzero allowance). Review this allowance independently before accepting a report;
the checker cannot authenticate evidence or establish that a submitted measurement is genuine.
Do not include credentials or credential-bearing commands in reports or evidence.

This checks report arithmetic and basic comparability only. Archive raw timing/RSS measurements,
server/client versions, schema digests, sanitized commands, and correctness evidence with the report.
The shared host/dataset fields assert that Python and Rust ran under identical conditions; reviewers
must verify this against those artifacts. A passing report is **not** a release approval and does
not verify the separate 100 GB soak, interrupted-resume, correctness, or installer gates.

Checker regression tests run with `cargo test -p elm-cli --example check_benchmarks` and use
explicitly synthetic numbers, never accepted benchmark results.

## Small PostgreSQL performance probe (Windows)

This probe uses 250,000 rows of bigint IDs and deterministic 1,024-byte text (258 MB logical data),
three sequential copies, default engine budgets, and atomic staged publication into new targets.
It measures connector/engine performance, not daemon/SQLite overhead, and is not the 10 GB gate.
Setup and full ID/payload verification are outside each transfer timer; connection and publication
are inside. Subsequent runs are warm-cache runs. No Python comparison is implied.

Use only a **fresh disposable** PostgreSQL 16 container named `elm-perf-postgres`, database
`elm_perf`, user `postgres`, bound to `127.0.0.1:55439`. The probe uses test-only trust authentication,
so do not expose the port beyond loopback or run it on a shared/untrusted host.
It deliberately refuses an existing `perf_source` table. Do not point production services at this port.

Build with `cargo build --release -p elm-connectors --features postgresql --example postgres_performance`,
then run `./benchmarks/run-postgres-performance.ps1`. JSON is printed to stdout. The runner queries
Windows' process peak working set every 100 ms, including setup/verification but excluding the
database server. This is not a whole-system RSS or long-run memory-growth guarantee.
Remove only the dedicated fixture and its test volume after recording results.

### Compare an installed Python ELM Tool

`./benchmarks/run-postgres-performance.ps1 -InstalledElm -PythonExecutable <absolute-python.exe>`
invokes the unchanged installed `elm.core.copy.copy_db_to_db` with the installed CLI's 1,000-row
batch default, one worker, REPLACE, and no masks. The Python interpreter runs in isolated import
mode (`-I`); settings/history use a temporary `ELM_TOOL_HOME`, not saved user environments.
Use a fresh copy of the same database fixture. Imports and fixture generation are outside the
timer; normal core history recording is included. Direct Python publication is not equivalent
to Rust's staged atomic publication, so record this safety difference with timing comparisons.

If the installed interpreter lacks the PostgreSQL driver, a disposable virtual environment with
`--system-site-packages` can reuse the installed ELM package while providing `psycopg2-binary`.
Do not reinstall or patch ELM for the comparison. Record the added driver version. Windows venv
launchers may spawn a separate Python interpreter; the runner discovers and measures that worker.
This benchmark-only Python script is not an application runtime or compatibility shim.
