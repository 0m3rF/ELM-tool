# Phase 04: Storage Layer & Recording - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement persistent storage that records every copy operation with full metadata — operation type, parameters, timestamps, status, and outcome. This phase covers the recording layer only; CLI list/re-run commands are Phase 5, and the GUI History tab is Phase 6.

**Requirements:** STOR-01, STOR-02, STOR-03, STOR-04, STOR-05

</domain>

<decisions>
## Implementation Decisions

### Storage Location
- **D-01:** History file is stored inside `ELM_TOOL_HOME` directory, co-located with `environments.ini` and `masking.json`.
- **D-02:** File name: `history.json` (resolved via `ConfigManager.get_history_file()`).

### Storage Format
- **D-03:** Simple flat JSON array (`List[Dict]`) — one object per history record.
- **D-04:** Each record must have a unique auto-incrementing integer `id`. Before writing, read the file, compute `max(existing ids) + 1`, and assign; this guarantees no duplicate IDs across all writers.
- **D-05:** Each record includes a `date` field (ISO-8601 timestamp string, e.g. `2026-05-04T14:30:00`).
- **D-06:** Record schema fields: `id`, `date`, `operation_type`, `source`, `target`, `query`, `table`, `mode`, `batch_size`, `parallel_workers`, `start_time`, `end_time`, `status` (`success` | `failure`), `record_count`, `error_message` (nullable).

### FIFO Eviction & Thread Safety
- **D-07:** All file reads, in-memory append/trim, and writes are performed under the existing `file_lock` from `elm/core/utils.py` (already used for `environments.ini`).
- **D-08:** Read → parse → append new record → if `len(records) > max_entries` (default 100), slice `records[-max_entries:]` → write back.
- **D-09:** Max entries is configurable via `ConfigManager` key `history.max_entries`, default `100`.

### Storage Failure Handling
- **D-10:** Before overwriting `history.json`, create a backup named `history.json.bak` in the same directory.
- **D-11:** After writing, read the file back and verify at least one record is present and the JSON parses cleanly.
- **D-12:** On successful verification, delete `history.json.bak`.
- **D-13:** On write or verification failure: restore from `history.json.bak` (if it exists), then silently proceed — the copy operation itself must never be blocked by a history-write failure.
- **D-14:** Return an `OperationResult` with a `history_saved` flag so callers (CLI, GUI) can warn the user if history persistence failed.

### Recording Layer Integration
- **D-15:** History recording is triggered inside `elm/core/copy.py` at the end of each public copy function (`copy_db_to_db`, `copy_db_to_file`, `copy_file_to_db`), immediately after the `OperationResult` is built.
- **D-16:** Recording wraps the entire call in a `try/except` so that any history-related exception is caught and the original `OperationResult` is still returned.

### Claude's Discretion
- Exact backup verification implementation (checksum vs re-read JSON).
- Whether to log history-write failures to stderr or a separate log file.
- Default `history.max_entries` fallback if config key is missing.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Logic
- `elm/core/copy.py` — `copy_db_to_db()`, `copy_db_to_file()`, `copy_file_to_db()` — where recording hooks are inserted
- `elm/core/types.py` — `OperationResult` dataclass; may need `history_saved: Optional[bool]` field
- `elm/core/utils.py` — `file_lock` context manager for thread-safe file access
- `elm/core/config.py` — `ConfigManager` for `get_history_file()` and `history.max_entries` config key

### API & CLI
- `elm/api.py` — Thin wrappers around core copy functions; recording happens in core, no API change needed
- `elm/elm_commands/copy.py` — Click command wrappers; also no change needed for Phase 4

### Existing Storage Patterns
- `elm/core/environment.py` — Uses `file_lock` + INI format; reference for safe concurrent file mutation
- `elm/elm_utils/variables.py` — `ENVS_FILE` pattern; add `HISTORY_FILE` following same convention

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `file_lock` (`elm/core/utils.py`) — Already used for `environments.ini`; reuse for `history.json`.
- `ConfigManager` (`elm/core/config.py`) — Add `get_history_file()` method following `get_envs_file()` / `get_mask_file()` pattern.
- `OperationResult` (`elm/core/types.py`) — Extend with `history_saved` flag (default `None` to preserve backward compatibility).

### Established Patterns
- JSON persistence: `ConfigManager` already reads/writes JSON (`config.json`). Use `json.dump(..., indent=2)` for human-readable history.
- File backup pattern: No existing backup-before-write pattern in codebase; this is a new pattern to implement.

### Integration Points
- Hook point: End of each `copy_*` function in `elm/core/copy.py`, right before `return create_success_result(...)` / `return handle_exception(...)`.
- The `OperationResult` from core is consumed by `elm/api.py` (returns `.to_dict()`) and `elm/elm_commands/copy.py` (checks `.success`). The new `history_saved` field will surface to both.
</code_context>

<specifics>
## Specific Ideas

- User explicitly wants backup-before-write with verification: "take a backup before writing. Ensure it has written correctly and remove backup."
- User wants date field in every record (beyond the existing start/end timestamps).
- No duplicate IDs must ever occur, even with concurrent file access.

</specifics>

<deferred>
## Deferred Ideas

- CLI `copy list`, `copy re-run`, `copy edit` commands — Phase 5
- GUI History tab with scrollable list and re-run/edit buttons — Phase 6
- History export to CSV/JSON — out of scope for v1.1 (per REQUIREMENTS.md)

</deferred>

---

*Phase: 04-storage-layer-recording*
*Context gathered: 2026-05-04*
