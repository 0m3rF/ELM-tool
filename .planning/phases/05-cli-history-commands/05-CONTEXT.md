# Phase 05: CLI History Commands - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Add `elm copy list`, `elm copy re-run <id>`, and `elm copy edit <id>` CLI commands that read from and interact with the `history.json` persisted by Phase 4. This phase covers CLI-only interaction with history; the GUI History panel is Phase 6.

**Requirements:** CLIC-01, CLIC-02, CLIC-03, CLIC-04

</domain>

<decisions>
## Implementation Decisions

### List Output Format
- **D-01:** Default output is a human-readable table (rich/tabulate style). Columns: ID, date, operation_type, source, target, table, status.
- **D-02:** JSON mode available via `--format json` flag for programmatic use.
- **D-03:** Default sort order: newest first (by `date` descending).

### Re-run Behavior
- **D-04:** `elm copy re-run <id>` executes immediately — no confirmation prompt.
- **D-05:** Successful re-run updates the existing history entry (does NOT create a new record).
- **D-06:** On re-run, update the original record with a `last_run_date` field (ISO-8601 timestamp).
- **D-07:** If the original environment or file no longer exists, the failed attempt is recorded as a new history entry with failure status.

### Edit & Re-run Workflow
- **D-08:** All original parameters are overridable via CLI flags (e.g., `--target new_env`, `--query "SELECT *"`).
- **D-09:** Before executing, show a preview of the modified command/parameters.
- **D-10:** The edited run is always recorded as a new history entry (never updates the original).

### Filtering & Sorting
- **D-11:** `elm copy list` supports filtering by any field in the record schema: `--status`, `--source`, `--target`, `--operation-type`, `--table`.
- **D-12:** Default limit is 10 entries. Override with `--limit N`; `--limit 0` means unlimited.
- **D-13:** Default sort: newest first (`date` descending). `--sort asc` for oldest first.

### Error Handling
- **D-14:** Non-existent ID referenced in `re-run` or `edit` → clear error message to stderr, exit with non-zero code. No history entry created for ID-not-found errors.
- **D-15:** Missing environment/file during re-run → record failure as new history entry per D-07.

### Claude's Discretion
- Exact table formatting library choice (tabulate, rich.table, or click.echo with manual formatting).
- Preview display format (key-value pairs vs command-line reconstruction).
- Whether to expose `--all` as an alias for `--limit 0`.
- Exact error message wording.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Core Logic
- `elm/core/history.py` — `HistoryRecorder`, `HistoryRecord` dataclass; read/parse `history.json`
- `elm/core/config.py` — `ConfigManager.get_history_file()` for history file path resolution
- `elm/core/types.py` — `OperationResult` dataclass (already extended with `history_saved` in Phase 4)

### CLI Patterns
- `elm/elm_commands/copy.py` — Existing Click command group structure; new subcommands (`list`, `re-run`, `edit`) added here
- `elm/elm.py` — CLI entrypoint; no changes needed (commands auto-register via Click group)

### Requirements
- `.planning/REQUIREMENTS.md` § CLI Commands (CLIC-01 through CLIC-04)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `HistoryRecorder._read_records()` in `elm/core/history.py` — can be reused or exposed as a public helper for reading history.
- `ConfigManager.get_history_file()` — already returns the path.
- `file_lock` from `elm/core/utils.py` — reuse for thread-safe history.json reads.

### Established Patterns
- Click command group with subcommands in `elm/elm_commands/copy.py`.
- `OperationResult` return pattern from core functions consumed by Click commands.
- JSON persistence with `json.dump(..., indent=2)` for human-readable files.

### Integration Points
- New subcommands (`list`, `re-run`, `edit`) register under the existing `copy` Click group in `elm/elm_commands/copy.py`.
- Re-run/edit commands invoke the same core `copy_db_to_db`, `copy_db_to_file`, `copy_file_to_db` functions used by the existing CLI commands.
- History recording is already hooked into core copy functions (Phase 4); CLI commands do not need to call `HistoryRecorder` directly for normal runs.

</code_context>

<specifics>
## Specific Ideas

- "table layout" for list output — user explicitly wants tabular human-readable output.
- "immediately run" — no confirmation prompts for re-run.
- "update the old one add last run date" — re-run mutates the original record.
- "everything should be overridable" — edit supports overriding any parameter from the original record.
- "show preview" — edit workflow must display the command/parameters before execution.
- "add as new entry" — edit always creates a new history record.
- "default 10 is ok" — list limit default is 10.
- "make it filtrable with every field" — all record fields usable as filters.

</specifics>

<deferred>
## Deferred Ideas

- GUI History tab with scrollable list and re-run/edit buttons — Phase 6
- History export to CSV/JSON — out of scope for v1.1 (per REQUIREMENTS.md)
- Full undo/rollback — out of scope for v1.1

</deferred>

---

*Phase: 05-cli-history-commands*
*Context gathered: 2026-05-04*
