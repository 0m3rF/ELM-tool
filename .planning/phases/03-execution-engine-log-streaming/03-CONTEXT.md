# Phase 03: Execution Engine & Log Streaming - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning
**Source:** Inferred from ROADMAP.md + Phase 02 UI-SPEC continuity

<domain>
## Phase Boundary

Build the execution thread and connect `sys.stdout` streaming to an observable log panel. This phase delivers:
- A Copy Operation visual form inside the "⚙ Operations" tab
- Background threading + queue.Queue so `elm/core` copy commands do not freeze the CustomTkinter event loop
- A persistent `CTkTextbox` log panel that consumes a thread-safe queue to render stdout/stderr in real time

</domain>

<decisions>
## Implementation Decisions

### Threading Model
- **D-01:** One dedicated `threading.Thread` per copy operation. The worker thread captures stdout/stderr via a custom file-like object that writes into a `queue.Queue`. The main thread polls the queue via `after()` at 100 ms intervals.

### Log Panel
- **D-02:** Persistent at bottom of Operations tab, fixed height ~200px, expandable via `pack(fill="both", expand=True)` within its container. Uses `CTkTextbox(state="disabled")` for read-only display, with `state="normal"` toggled only during insertions to prevent user editing.

### Copy Form Controls
- **D-03:** Source/Target environment dropdowns populated from `api.list_environments()`. "Execute" button triggers background copy. "Cancel" button (if operation running) sets a `threading.Event` to request graceful stop.

### Stdout Redirection
- **D-04:** Replace `sys.stdout` and `sys.stderr` inside the worker thread with a `queue.Queue`-backed file-like writer. Main thread drains queue into the `CTkTextbox`. Never replace global sys.stdout on the main thread to avoid breaking CustomTkinter internal logging.

### Claude's Discretion
- Exact queue polling interval (100 ms default)
- Log line color coding (optional: green for stdout, red for stderr)
- Cancel behavior: cooperative vs forceful
- Whether to auto-scroll log to bottom on new lines
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Implementation Reference
- `elm/gui/app.py` — Main CTkTabview with "⚙ Operations" placeholder label to replace
- `elm/api.py` — `copy_db_to_db()`, `copy_db_to_file()`, `list_environments()`
- `elm/core/copy.py` — Core copy logic that prints batch logs to stdout
- `elm/gui/environment_manager.py` — Pattern for CTk widgets and API integration (Phase 2 reference)

### Design System
- `.planning/phases/02-environment-management-visuals/02-UI-SPEC.md` — Spacing, color, typography tokens (60/30/10, dark-blue theme, 4px grid)

</canonical_refs>

<specifics>
## Specific Ideas

- Log panel should show timestamps if possible, or raw stdout lines as emitted by `elm.core.copy`
- Form should include: Source env dropdown, Target env dropdown, SQL query entry (multiline CTkTextbox), Target table name entry, Execute button, optional Cancel button
- Consider showing a "Running..." progress indicator (spinning text or simple label) while thread is alive
</specifics>

<deferred>
## Deferred Ideas

- Visual progress bars deriving percentages from log outputs (ADV-02, v2)
- Fine-grained masking rules via visual table builder (ADV-01, v2)
</deferred>

---

*Phase: 03-execution-engine-log-streaming*
*Context gathered: 2026-04-23 via bridge from ROADMAP*
