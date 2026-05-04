---
phase: 3
plan: 1
type: feature
wave: 1
depends_on: [02-environment-management-visuals]
files_modified:
  - elm/gui/operations_panel.py
  - elm/gui/app.py
autonomous: true
requirements: [EXEC-01, EXEC-02, EXEC-03, MON-01, MON-02]
---

<objective>
Build the Operations GUI inside the "⚙ Operations" tab. Replace the placeholder label with a live panel containing: (a) a copy-operation form with Source/Target environment dropdowns, a SQL query textbox, a target table entry, and an Execute button; and (b) a persistent read-only log panel (`CTkTextbox`) that displays stdout/stderr from background copy threads in real time. Implement `threading.Thread` + `queue.Queue` to run `api.copy_db_to_db()` without blocking the CustomTkinter main loop, and redirect the worker thread's `sys.stdout`/`sys.stderr` into the queue so core `safe_print` batch logs appear natively in the GUI.
</objective>

<tasks>

### Task 1: Create elm/gui/operations_panel.py

<type>create</type>
<files>elm/gui/operations_panel.py</files>

<read_first>
- elm/gui/app.py (understand CTkTabview tab structure for the "⚙ Operations" tab)
- elm/api.py (copy_db_to_db signature: source_env, target_env, query, table; list_environments signature)
- elm/core/types.py (OperationResult.to_dict() fields: success, message, error_details)
- elm/core/utils.py (safe_print implementation — uses print() which writes to sys.stdout)
- .planning/phases/03-execution-engine-log-streaming/03-UI-SPEC.md (design tokens, spacing, colors, component inventory)
- .planning/phases/03-execution-engine-log-streaming/03-CONTEXT.md (D-01 threading model, D-02 log panel, D-03 copy form, D-04 stdout redirect rules)
</read_first>

<action>
Create `elm/gui/operations_panel.py` containing two classes. The file must be importable without launching a GUI (no `mainloop` or window creation at module level).

**Imports at top:**
```python
import queue
import sys
import threading
import customtkinter as ctk
from elm import api
```

**Class 1: `QueueStream`**
- `__init__(self, log_queue)` stores the queue
- `write(self, text)` puts `text` into `self.log_queue` (if text is non-empty string)
- `flush(self)` is a no-op

**Class 2: `OperationsPanel(ctk.CTkFrame)`**
- `__init__(self, master)` builds the layout in two vertical sections:

**Top — Copy Form (`CTkFrame`, packed `fill="x"`, `padx=10`, `pady=(10,0)`):**
1. Source Environment label: `CTkLabel(text="Source Environment", font=("", 12))` packed `anchor="w"`
2. Source dropdown: `CTkOptionMenu(variable=self.source_var, values=[])` packed `fill="x"`, `pady=(0,8)`
3. Target Environment label: `CTkLabel(text="Target Environment", font=("", 12))` packed `anchor="w"`
4. Target dropdown: `CTkOptionMenu(variable=self.target_var, values=[])` packed `fill="x"`, `pady=(0,8)`
5. SQL Query label: `CTkLabel(text="SQL Query", font=("", 12))` packed `anchor="w"`
6. Query input: `CTkTextbox(height=80)` packed `fill="x"`, `pady=(0,8)`
7. Target Table label: `CTkLabel(text="Target Table", font=("", 12))` packed `anchor="w"`
8. Table entry: `CTkEntry(placeholder_text="target_table_name")` packed `fill="x"`, `pady=(0,8)`
9. Button row (`CTkFrame(fg_color="transparent")`):
   - Execute: `CTkButton(text="▶ Execute", command=self._on_execute)` packed `side="left"`, `padx=(0,8)`
   - Cancel: `CTkButton(text="⏹ Cancel", fg_color="#DC3545", hover_color="#C82333", command=self._on_cancel)` packed `side="left"`, initially `state="disabled"`
10. Status label: `CTkLabel(text="Ready", font=("", 12))` packed `anchor="w"`, `pady=(4,0)`

**Bottom — Log Panel (`CTkFrame`, packed `fill="both", expand=True`, `padx=10`, `pady=10`):**
1. Log title: `CTkLabel(text="Execution Log", font=("", 18, "bold"))` packed `anchor="w"`, `pady=(0,8)`
2. Log textbox: `CTkTextbox(state="disabled")` packed `fill="both", expand=True`
3. Clear button: `CTkButton(text="Clear Log", width=80, command=self._clear_log)` packed `anchor="e"`, `pady=(8,0)`

**Threading state (instance variables):**
- `self.worker_thread = None`
- `self.log_queue = queue.Queue()`
- `self.cancel_event = threading.Event()`

**Methods:**
- `_refresh_environments()`: calls `api.list_environments(show_all=True)`, extracts `name` from each dict, sets dropdown `values`, defaults source to first item and target to second (or first if only one). If list is empty, set both dropdowns to `values=["(no environments)"]` and disable Execute button.
- `_on_execute()`:
  1. Read `source = self.source_var.get()`, `target = self.target_var.get()`, `query = self.query_input.get("1.0", "end-1c").strip()`, `table = self.table_entry.get().strip()`
  2. Validate: if any field empty OR source == target, set status label to red error text (e.g., `"Please fill all fields"` or `"Source and target must differ"`) and return
  3. Set status `"Running..."` in default color, disable Execute, enable Cancel
  4. Clear log via `_clear_log()`
  5. Create fresh `queue.Queue()` and `threading.Event()`
  6. Spawn `threading.Thread(target=self._copy_worker, args=(source, target, query, table, self.log_queue, self.cancel_event), daemon=True)`
  7. Start thread and schedule `self.after(100, self._poll_queue)`
- `_copy_worker(source_env, target_env, query, table, log_queue, cancel_event)` (static method or bound method, must accept these args exactly):
  1. Save `old_stdout = sys.stdout`, `old_stderr = sys.stderr`
  2. Replace both with `QueueStream(log_queue)`
  3. Try: call `api.copy_db_to_db(source_env=source_env, target_env=target_env, query=query, table=table)` and capture return value
  4. If result dict has `success=True`, put `\n✓ Copy completed successfully.\n` into queue
  5. Else put `\n✗ Copy failed: {result.get('error_details', result.get('message', 'Unknown error'))}\n`
  6. Except Exception as e: put `\n✗ Error: {str(e)}\n`
  7. Finally: restore `sys.stdout = old_stdout`, `sys.stderr = old_stderr`
- `_poll_queue()`: drain queue with `get_nowait()` in a `while True` loop; for each item, set log textbox to `state="normal"`, `insert("end", text)`, `state="disabled"`, `see("end")`; break on `queue.Empty`. After draining, if `self.worker_thread` is alive, schedule `self.after(100, self._poll_queue)`; else call `_on_copy_finished()`
- `_on_copy_finished()`: re-enable Execute, disable Cancel, set status `"Ready"`, then call `_drain_remaining_queue()` to flush any last queue items into the textbox
- `_drain_remaining_queue()`: same drain loop as `_poll_queue` but does NOT schedule another after()
- `_on_cancel()`: call `self.cancel_event.set()`, set status `"Cancelling..."` in red. Note: this is cooperative — the core copy API does not check the event, so the worker will finish the current operation then exit.
- `_clear_log()`: set textbox `state="normal"`, `delete("1.0", "end")`, `state="disabled"`

At the end of `__init__`, call `self._refresh_environments()`.
</action>

<acceptance_criteria>
- File `elm/gui/operations_panel.py` exists and is importable via `python -c "from elm.gui.operations_panel import OperationsPanel, QueueStream; print('ok')"`
- Contains `class QueueStream` with `write(self, text)` and `flush(self)` methods
- Contains `class OperationsPanel(ctk.CTkFrame)` with `__init__(self, master)`
- `OperationsPanel` contains `CTkOptionMenu` widgets for source and target (grep `CTkOptionMenu`)
- `OperationsPanel` contains a `CTkTextbox` for query input with `height=80`
- `OperationsPanel` contains a `CTkEntry` for target table with `placeholder_text="target_table_name"`
- `OperationsPanel` contains Execute button with `text="▶ Execute"` and Cancel button with `fg_color="#DC3545"`
- `OperationsPanel` contains a `CTkTextbox` for log display with `state="disabled"`
- `OperationsPanel` contains a `CTkButton(text="Clear Log")`
- `OperationsPanel` imports `api` from `elm` and calls `api.list_environments(show_all=True)` inside `_refresh_environments`
- `OperationsPanel` calls `api.copy_db_to_db` inside the worker thread
- `QueueStream.write` puts text into a `queue.Queue`
- The worker thread redirects `sys.stdout` and `sys.stderr` to `QueueStream` instances
- `_poll_queue` method exists and uses `self.after(100, ...)` scheduling
</acceptance_criteria>

---

### Task 2: Wire OperationsPanel into app.py

<type>modify</type>
<files>elm/gui/app.py</files>

<read_first>
- elm/gui/app.py (MUST read current content before editing — verify exact placeholder lines)
- elm/gui/operations_panel.py (verify it exists and is importable)
</read_first>

<action>
Replace the placeholder label in the `_build_tabs` method of `ELMApp` with the live `OperationsPanel`.

Find these lines in `elm/gui/app.py`:
```python
        ops_label = ctk.CTkLabel(
            self.tabview.tab("⚙ Operations"),
            text="Copy operations will appear here.",
            font=("", 14),
        )
        ops_label.pack(expand=True)
```

Replace them with:
```python
        from elm.gui.operations_panel import OperationsPanel
        self.ops_panel = OperationsPanel(self.tabview.tab("⚙ Operations"))
        self.ops_panel.pack(fill="both", expand=True)
```

The import is placed inside `_build_tabs` (not at top of file) to prevent circular import issues during module-level import of `app.py`. `OperationsPanel` is a `ctk.CTkFrame` subclass that fills its parent tab via `pack(fill="both", expand=True)`.
</action>

<acceptance_criteria>
- `elm/gui/app.py` no longer contains the placeholder text `"Copy operations will appear here."`
- `elm/gui/app.py` contains `from elm.gui.operations_panel import OperationsPanel`
- `elm/gui/app.py` contains `self.ops_panel = OperationsPanel(...)`
- `elm/gui/app.py` contains `self.ops_panel.pack(fill="both", expand=True)`
- Running `python -c "from elm.gui.app import ELMApp; print('ok')"` succeeds (no import errors)
</acceptance_criteria>

</tasks>

<verification>
## Verification Steps

1. **Module importable (no GUI launch):**
   ```bash
   python -c "from elm.gui.operations_panel import OperationsPanel, QueueStream; print('ok')"
   ```
   Expected: prints `ok`, exit code 0.

2. **App module still importable after wiring:**
   ```bash
   python -c "from elm.gui.app import ELMApp; print('ok')"
   ```
   Expected: prints `ok`, exit code 0.

3. **GUI launches without crash (manual verification):**
   ```bash
   python -m elm.elm
   ```
   Expected: CustomTkinter window opens. The "⚙ Operations" tab shows a form with Source/Target dropdowns, SQL Query box, Target Table entry, Execute button, and a log panel below. No placeholder text is visible.

4. **Widget creation lines (≥8):**
   ```bash
   grep -E "CTkOptionMenu|CTkTextbox|CTkEntry|CTkButton|CTkLabel" elm/gui/operations_panel.py | wc -l
   ```
   Expected: ≥ 8 widget creation lines.

5. **Threading primitives present:**
   ```bash
   grep -E "threading\.Thread|queue\.Queue|threading\.Event" elm/gui/operations_panel.py
   ```
   Expected: matches for Thread, Queue, and Event.

6. **Stdout redirect present:**
   ```bash
   grep -E "sys\.stdout\s*=|sys\.stderr\s*=|QueueStream" elm/gui/operations_panel.py
   ```
   Expected: matches showing QueueStream assignment to both stdout and stderr.

7. **API integration present:**
   ```bash
   grep -E "api\.list_environments|api\.copy_db_to_db" elm/gui/operations_panel.py
   ```
   Expected: both calls found.

8. **Polling loop present:**
   ```bash
   grep -E "self\.after\(100|_poll_queue" elm/gui/operations_panel.py
   ```
   Expected: `self.after(100, self._poll_queue)` or similar found.
</verification>

<must_haves>
- [ ] Copy form contains Source and Target environment dropdowns populated from `api.list_environments()` (EXEC-01)
- [ ] Copy form contains SQL query input, target table entry, and an Execute button that triggers a background copy (EXEC-02)
- [ ] Background copy runs in a `threading.Thread` so the CustomTkinter main loop is not frozen (EXEC-03)
- [ ] A persistent `CTkTextbox` log panel exists at the bottom of the Operations tab (MON-01)
- [ ] The worker thread redirects `sys.stdout` and `sys.stderr` to a `queue.Queue`, and the main thread drains the queue into the log panel every 100 ms so core batch logs appear in real time (MON-02)
- [ ] Log panel is read-only (`state="disabled"` toggled only during inserts)
- [ ] Execute/Cancel button states toggle correctly during operation lifecycle
- [ ] Import of `OperationsPanel` is placed inside `_build_tabs` to avoid circular imports
- [ ] All UI text, spacing, and colors follow the Phase 3 UI-SPEC contract (inherited from Phase 2 design system)
</must_haves>

## PLANNING COMPLETE
