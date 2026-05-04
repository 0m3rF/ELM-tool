# Phase 04: Storage Layer & Recording - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the analysis.

**Date:** 2026-05-04
**Phase:** 04-storage-layer-recording
**Mode:** discuss
**Areas discussed:** 4

## Decisions Captured

### Recording Layer (D-01, D-02)
- **Decision:** History stored in `ELM_TOOL_HOME` as `history.json`.
- **Rationale:** Co-located with existing config (`environments.ini`, `masking.json`), consistent with project patterns.

### Storage Format & Record Schema (D-03 – D-06)
- **Decision:** Simple flat JSON array with unique auto-incrementing integer `id`, plus `date` field (ISO-8601).
- **Rationale:** User explicitly requested flat array, no duplicate IDs, and date inclusion. Unique IDs are computed by reading existing records and taking `max(id) + 1` under lock.

### FIFO Eviction & Thread Safety (D-07 – D-09)
- **Decision:** All operations (read → parse → append → trim → write) performed under `file_lock`.
- **Rationale:** User explicitly requested "use all under lock". Existing `file_lock` utility in `elm/core/utils.py` is already used for `environments.ini`.

### Storage Failure Handling (D-10 – D-14)
- **Decision:** Backup-before-write with verification: create `history.json.bak`, write, verify JSON parses, delete backup. On failure, restore from backup. Never block the copy operation.
- **Rationale:** User explicitly requested: "take a backup before writing. Ensure it has written correctly and remove backup."

### Recording Layer Integration (D-15 – D-16)
- **Decision:** Recording triggered inside `elm/core/copy.py` at end of each `copy_*` function, wrapped in `try/except`.
- **Rationale:** Core-layer recording ensures all entrypoints (CLI, API, GUI) capture history without per-caller changes.

## Auto-Selected / Pre-Answered

- No auto-selections; all decisions came from direct user input.

## External Research

- None performed; codebase analysis provided sufficient context.
