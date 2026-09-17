# ELM Tool v2

ELM Tool is a local-first Rust application for bounded-memory, recoverable data transfers. It is
organized as a reusable Arrow engine, a per-user daemon, an automation CLI, and a Tauri 2 desktop
application.

> **Development status:** `2.0.0-alpha.1`. The Arrow engine, CSV/NDJSON/Parquet connectors,
> SQLite state, authenticated local IPC, CLI surface, and Tauri shell are implemented. PostgreSQL
> sources and atomic sinks use binary COPY; MySQL streaming sources and staged sinks are enabled.
> SQL Server typed ODBC sources and atomic staged sinks are enabled with limited types; Linux native
> integration tests pass. The experimental Oracle source is wired to the daemon with explicit
> performance warnings and restart-from-zero recovery. Oracle uses bounded native array fetching
> and batch DML staging; existing-table REPLACE requires explicit non-atomic `--consistency table-swap`.
> Native client checks and connection diagnostics are enabled. Desktop connection and masking CRUD,
> an explicit per-transfer masking-rule picker, structured job submission and actions, settings
> persistence, history filtering, and live events are wired; schema preview remains pending.
> PostgreSQL connection diagnostics, exact Decimal128 conversion,
> explicit Arrow conversion rules, and durable file resume are implemented. See the
> [connector status](docs/connectors.md).

## Safety model

- Transfers stream `Source -> Arrow RecordBatch -> Mask/Convert -> Sink` through byte-bounded
  channels. The default process budget is 512 MiB and batches target 16 MiB (allowed range:
  8–32 MiB).
- Atomic mode stages output and leaves the visible target unchanged until schema and row-count
  validation succeeds. If safe publication is unavailable, preflight fails with remediation.
- `--consistency checkpointed` is an explicit advanced choice. ELM never selects it automatically.
- Oracle existing-table REPLACE requires `--mode replace --consistency table-swap`. This is a
  non-atomic rename swap, not DELETE-and-reinsert. The old table remains as a recovery backup
  until job deletion; its indexes, grants, and constraints are not copied to the replacement.
- Environment metadata, jobs, attempts, checkpoints, and events live in per-user SQLite. Database
  passwords live only in the operating system keychain.
- Random masking is deterministic from the persisted job seed and batch/row position, so retries
  produce the same result. Masking is opt-in per transfer: saving a rule has no effect until it is
  explicitly attached, via the desktop New Transfer wizard's "Masking" step or `elm copy
  --mask-rule <id>`.
- File attempts persist committed batches as job-scoped Parquet chunks. Resume validates the input
  fingerprint and continues from the committed logical row without losing or duplicating output.
- Conversion rules are schema-preflighted. Potentially lossy casts require `allow_lossy` on the
  individual column, and value-level cast failures fail the attempt rather than producing nulls.
- User SQL is an explicit source and is never extended with identifiers or checkpoint literals.
- There is no pause action. `cancel` stops a job permanently (it cannot be resumed, only
  deleted); `resume` only applies to a job the daemon marked `interrupted` (e.g. an unclean
  daemon stop) or `failed`, and continues from its last durable checkpoint.

The frozen behavior for APPEND, REPLACE, FAIL, empty inputs, cancellation, resume, and redaction is
in [the correctness contract](docs/correctness-contract.md).

## Workspace

| Crate | Responsibility |
| --- | --- |
| `elm-core` | Versioned job types, validation, masking, errors, IPC schema, connector traits |
| `elm-engine` | Backpressure, memory permits, transforms, cancellation, checkpoints, spill files |
| `elm-connectors` | File streams, dialect quoting/publication plans, native connector feature gates |
| `elm-state` | SQLite durability and native keychain integration |
| `elm-daemon` | Persistent executor on an authenticated named pipe or Unix socket |
| `elm-cli` | `elm` automation interface and daemon auto-start client |
| `elm-desktop` | Tauri 2 commands plus a React/TypeScript/Vite frontend |

More detail is in [the architecture guide](docs/architecture.md).

## Build and test

Install stable Rust, Node.js 22+, and platform prerequisites for Tauri 2, then run:

```powershell
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

cd crates/elm-desktop
npm ci
npm run build
```

Run the service and CLI from a development build:

```powershell
cargo run -p elm-daemon
cargo run -p elm-cli -- daemon status
cargo run -p elm-cli -- env list
```

## CLI

```text
elm daemon start|status|stop
elm env add|edit|list|test|remove
elm mask add|edit|list|remove|test
elm copy db-to-db|db-to-file|file-to-db [--detach] [--mask-rule <id>,...]
elm jobs list|show|watch|cancel|resume|delete
```

Copy commands attach to live progress by default. `--detach` leaves execution with the daemon. Use
`--json` for automation. Passwords are prompted securely or read from stdin; they are not accepted
as command-line values.

## Desktop

```powershell
cd crates/elm-desktop
npm run dev
# in another terminal from the repository root
cargo run -p elm-desktop
```

The desktop has Connections, New Transfer, Jobs, Masking Rules, and Settings screens. Active job
events use a Tauri channel backed by the daemon's streaming `job.watch` request rather than stdout
capture or polling.

## Database prerequisites

The PostgreSQL backend uses the native Rust protocol and binary COPY. TLS is required and verifies
the host and certificate chain by default; an optional PEM root can be configured for private CAs.
Plaintext transport requires explicit `ssl_mode=disable`. The MySQL backend also uses a native Rust
client with verified TLS by default. SQL Server requires Microsoft ODBC Driver 18, and Oracle
requires Oracle Instant Client; ELM does not bundle these proprietary clients, but on Windows it
can install them itself after you confirm: `elm native status` reports what's installed, and
`elm native install <oracle|sql-server>` (or the desktop's Settings screen) downloads and installs
one, verifying its SHA-256 against a pinned value first. Oracle needs no elevation (its files are
placed next to `elm-daemon`, which Windows checks before `PATH`); SQL Server's driver registers
itself with the OS, so installing it triggers one Windows UAC elevation prompt you approve
separately. macOS/Linux don't have this automation yet and still need a manual install. Consult
[the connector matrix](docs/connectors.md) for the current type and recovery boundaries.
Linux/macOS installations also require unixODBC for the daemon's native diagnostics. SQL Server
and Oracle connection tests report missing clients before login and run native calls on blocking
workers; successful live login verification requires those clients and a reachable test database.

## Production precautions

Use a non-production destination until schema mapping and publication privileges have been verified.
ELM deliberately refuses lossy conversions and unavailable staging privileges. A resumed database
source is key-bounded, not a recoverable database snapshot; exact point-in-time output requires the
source dataset to remain stable.

Licensed under GPL-3.0-or-later.
