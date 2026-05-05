# Phase 06: GUI History Panel - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Add a "History" tab to the GUI with a scrollable list of past copy operations and Re-run / Edit & Re-run actions. Operations panel already handles log streaming and form controls — this phase wires history data and tab interaction into the existing GUI.

**Requirements:** GUIR-01, GUIR-02, GUIR-03, GUIR-04, GUIR-05

</domain>

<decisions>
## Implementation Decisions

### History Entry Display Format
- **D-01:** Compact table-style rows (not detailed cards). Each row shows: ID, date, operation_type, source → target, table, status — color-coded by status.
- **D-02:** Rows are clickable/selectable CTkButton or CTkFrame rows inside a CTkScrollableFrame, similar to EnvironmentListPanel button pattern.

### Re-run Log Streaming
- **D-03:** When Re-run is clicked from History, execution output streams to the **existing Operations tab log panel** — no separate log in History tab.
- **D-04:** The GUI automatically switches focus to the Operations tab when a Re-run starts from History, so the user sees the live log.

### Auto-Refresh Strategy
- **D-05:** History list refreshes **only on tab switch** (when user clicks onto the History tab).
- **D-06:** Additionally, a **scheduled poll every 15 seconds** (`self.after(15000, ...)`) checks for new records while the History tab is active.
- **D-07:** After a copy operation finishes in Operations, the next History tab switch or 15s poll will pick up the new record.

### Edit & Re-run Navigation
- **D-08:** Clicking "Edit & Re-run" on a history entry **automatically switches to the Operations tab** and **pre-fills the form** with that entry's original parameters (source, target, query, table, mode, batch_size).
- **D-09:** The user can modify the pre-filled form before executing.
- **D-10:** The pre-fill action is a method on OperationsPanel exposed to the History panel via the shared parent (ELMApp).

### Color Coding
- **D-11:** Success status → green (#28A745), failure → red (#DC3545), in-progress/running → blue (#3B8ED0). These match existing environment_manager.py status colors.

### Claude's Discretion
- Exact row height and padding in the scrollable list
- Whether to show record_count in the compact row (nice-to-have)
- Scroll-to-bottom behavior when new records appear
- Whether to add a "Clear History" button in this phase
- Exact implementation of the scheduled poll timer cleanup on app close

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### GUI Code
- `elm/gui/app.py` — CTkTabview setup, existing Environments + Operations tabs, tab switching mechanism
- `elm/gui/operations_panel.py` — log panel (`CTkTextbox`), form fields, background thread/queue pattern, `_copy_worker`
- `elm/gui/environment_manager.py` — scrollable button list pattern (`EnvironmentListPanel`)

### History Data Layer
- `elm/core/history.py` — `HistoryRecord` dataclass, `HistoryRecorder.read_records()`, `get_record()`
- `elm/core/config.py` — `ConfigManager.get_history_file()` for file path
- `elm/api.py` — may need `list_history()` / `get_history_record()` / `rerun_history_record()` API wrappers

### Prior Phase Decisions
- `.planning/phases/04-storage-layer-recording/04-CONTEXT.md` — record schema (id, date, operation_type, source, target, table, status, record_count, error_message)
- `.planning/phases/05-cli-history-commands/05-CONTEXT.md` — re-run updates original record with `last_run_date`; edit & re-run creates new record

### Requirements
- `.planning/REQUIREMENTS.md` — GUIR-01 through GUIR-05

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `CTkScrollableFrame` + `CTkButton` rows — used in `EnvironmentListPanel` for environment list; reuse for history list
- `QueueStream` + background thread pattern — in `OperationsPanel._copy_worker`; reuse for history-triggered re-runs
- `CTkTextbox` log panel — in `OperationsPanel`; history re-runs stream output here
- `CTkOptionMenu` — already used for source/target dropdowns in OperationsPanel

### Established Patterns
- Tab content is a `CTkFrame` packed `fill="both", expand=True` inside `tabview.tab("name")`
- Inside-method imports (`from elm.gui.operations_panel import OperationsPanel`) used in `ELMApp._build_tabs` to prevent circular imports
- Background operations use `threading.Thread(daemon=True)` + `self.after(100, self._poll_queue)`
- Status colors: `#28A745` (success), `#DC3545` (error), `#3B8ED0` (accent)
- API layer (`elm/api.py`) wraps core functions; GUI calls `api.*` not core directly

### Integration Points
- New tab added in `ELMApp._build_tabs()` alongside existing "🌐  Environments" and "⚙  Operations"
- HistoryPanel needs a reference to `ELMApp` (or at least `tabview` + `ops_panel`) to:
  - Switch tabs (`self.tabview.set("⚙  Operations")`)
  - Call `OperationsPanel.pre_fill_form(params)` for Edit & Re-run
  - Stream re-run logs into `OperationsPanel.log_textbox`
- `OperationsPanel._on_copy_finished()` may need to signal HistoryPanel that a new record is available (or rely on the 15s poll)
- API needs history read functions (not yet exposed in `elm/api.py`)

</code_context>

<specifics>
## Specific Ideas

- "like operations tab" — history re-run output should go to the existing Operations log panel, not a new one
- "refresh only on tab switch. also add a scheduled job to check every 15 seconds" — dual refresh strategy
- "the GUI automatically switch to the Operations tab and pre-fill the form" — Edit & Re-run is a one-click "take me to Operations with this data" action

</specifics>

<deferred>
## Deferred Ideas

- History export to CSV/JSON — out of scope for v1.1 (per REQUIREMENTS.md)
- Full undo/rollback — out of scope for v1.1
- "Clear History" button — deferred; could be added later
- Sorting/filtering within the History tab GUI — not required by GUIR-01..05

</deferred>

---

*Phase: 06-gui-history-panel*
*Context gathered: 2026-05-04*
