# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is ELM Tool

ELM (Environment & Lifecycle Manager) is a Python database utility for copying, masking, and generating data across multi-database environments (PostgreSQL, Oracle, MySQL, MSSQL). It exposes three interfaces over a shared core: a Click-based CLI, a CustomTkinter GUI, and a Python API.

## Commands

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the app (no args → GUI; with args → CLI)
python elm-tool.py
python elm-tool.py environment list
python elm-tool.py copy db2db --source SRC --target TGT --table my_table

# Run all tests (parallel, with coverage)
pytest tests

# Run a single test file
pytest tests/core/test_copy.py

# Run tests with a specific marker
pytest tests -m integration
pytest tests -m "not integration"

# Run a specific test by name
pytest tests/core/test_copy.py::test_copy_db_to_file -v
```

**pytest.ini markers:** `integration`, `db_access`, `file_io`, `serial`  
Tests run in parallel via `pytest-xdist` with `loadscope` strategy and a 30-second per-test timeout.

## Architecture

```
User Input (CLI / GUI / API)
        ↓
elm/elm.py          → CLI entry point (Click); launches GUI when no args given
elm/gui/app.py      → CustomTkinter 3-tab GUI (Environments, Operations, History)
elm/api.py          → High-level Python API (wraps core directly)
        ↓
elm/core/           → All business logic lives here
        ↓
elm/elm_utils/      → DB connectivity (SQLAlchemy), encryption, Faker wrappers, file locking
        ↓
External: SQLAlchemy, Pandas, databases
```

### Core modules (`elm/core/`)

| Module | Responsibility |
|---|---|
| `environment.py` | Database environment CRUD, credential storage |
| `copy.py` | Data transfer: `db2file`, `file2db`, `db2db` |
| `streaming.py` | Memory-efficient streaming for LOB/large datasets |
| `masking.py` | Data masking with mtime-aware disk cache |
| `generation.py` | Faker-based test data generation |
| `history.py` | JSON audit trail with file locking |
| `metadata_sync.py` | Schema synchronization between environments |
| `config.py` | App-level configuration management |
| `types.py` | Dataclass types, enums, `OperationResult` |
| `exceptions.py` | Custom exception hierarchy |

### Interface layers

- **CLI** (`elm/elm_commands/`) — thin Click command wrappers that call `elm.core.*`. Supports short aliases (`env`, `cpy`, `msk`, `gen`, `cfg`, `syn`).
- **GUI** (`elm/gui/`) — `EnvironmentManagerFrame`, `OperationsPanel`, `HistoryPanel` tabs. Uses polling for history refresh.
- **API** (`elm/api.py`) — callable functions for all operations; same signatures as CLI but Python-native.

### Key patterns

- **`OperationResult`** — every core operation returns this standard dataclass (success, record count, message).
- **Streaming batches** — `streaming.py` feeds data in configurable batch sizes; masking cache (`masking.py`) lives per-batch to avoid repeated disk I/O.
- **Multi-DB bulk loaders** — each database has an optimized path: PostgreSQL uses COPY protocol or `execute_values`; Oracle uses array binding; MSSQL uses `fast_executemany`.
- **File locking** (`elm_utils/file_lock.py`) — cross-platform lock guards history and config writes.
- **Encryption** (`elm_utils/encryption.py`) — optional encryption for environment credentials stored on disk.

### Test layout

```
tests/
  core/        # Unit tests for each elm/core module
  cli/         # Click command parsing and execution
  api/         # Python API surface tests
  integration/ # Full database workflow tests (need live DB)
  utils/       # elm_utils helpers
  conftest.py  # Fixtures: worker_id, unique_env_name, unique_table_name, sample_dataframe
```

`unique_env_name` and `unique_table_name` fixtures use `worker_id` to isolate parallel test runs from colliding on shared database state.


## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.