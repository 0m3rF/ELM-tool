---
phase: 05-cli-history-commands
plan: 01
status: complete
completed: "2026-05-04"
---

# Summary: Phase 05 Plan 01 — CLI History Commands

## What Was Built

### 1. Extended HistoryRecorder (`elm/core/history.py`)
- Added `last_run_date: Optional[str] = None` to `HistoryRecord` dataclass (backward-compatible with existing JSON files)
- Added `read_records()` → returns all history records via `HistoryRecorder`
- Added `get_record(record_id)` → finds a single record by ID, or `None`
- Added `update_record(record_id, **kwargs)` → thread-safe field updates using existing `file_lock`

### 2. New CLI Subcommands (`elm/elm_commands/copy.py`)
- **`elm copy list`** — tabular or JSON output of past operations with filtering (`--status`, `--source`, `--target`, `--operation-type`, `--table`, `--limit`, `--sort`, `--format`)
- **`elm copy re-run <id>`** — re-executes a stored operation using its original parameters, updates the original record's `last_run_date` on success; missing environments/files create a new failure entry
- **`elm copy edit <id>`** — shows a parameter preview, applies any CLI overrides, then executes and always records a **new** history entry (never mutates original)
- Aliases registered: `history` → list, `ls` → list, `rerun` → re-run

### 3. Comprehensive Test Suite (`tests/test_history_cli.py`)
- 14 pytest test functions covering list, re-run, and edit commands
- Tests for empty history, table/JSON output, filtering, limits, sorting
- Tests for re-run success, missing record, missing-env failure behavior
- Tests for edit preview, parameter override, and new-record creation

## Key Files

| File | Change |
|------|--------|
| `elm/core/history.py` | Added `last_run_date`, `read_records()`, `get_record()`, `update_record()` |
| `elm/elm_commands/copy.py` | Added `list_history`, `re_run`, `edit_history` commands + aliases |
| `tests/test_history_cli.py` | New test file (14 tests) |

## Acceptance Criteria

| Check | Result |
|-------|--------|
| `python -c "from elm.elm_commands.copy import list_history, re_run, edit_history; print('OK')"` | ✅ OK |
| 6 `@copy.command` decorators in `copy.py` | ✅ 6 |
| `def list_history`, `def re_run`, `def edit_history` each present once | ✅ |
| `HistoryRecorder` has `read_records`, `get_record`, `update_record` | ✅ |
| `pytest tests/test_history_cli.py -v` passes all 14 tests | ✅ |
| Existing `tests/test_history.py` still passes (no regressions) | ✅ |

## Deviations

- None.
