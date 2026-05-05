---
phase: 06-gui-history-panel
plan: gap-fixes
type: fix
based_on: 06-UAT.md
---

# Phase 6 Gap Fixes Plan

Generated from UAT gaps diagnosed 2026-05-05.

## Gap 1: UI Blink on History Refresh (minor, Test 2)

**Root cause:** `HistoryPanel._refresh_list()` unconditionally destroys all row widgets via `widget.destroy()` and rebuilds from scratch on every poll/tab switch.

**Fix:**
1. In `elm/gui/history_panel.py`, modify `_refresh_list()` to store a hash of the last-fetched records.
2. On refresh, compare the new records list against the cached hash.
3. Only call `widget.destroy()` and rebuild if the records have actually changed.
4. If records are identical, skip destruction/rebuild and only update the `status_label` timestamp.

**Files:** `elm/gui/history_panel.py`

## Gap 2: Re-run Lacks Pre-flight Validation + Environment Editing (major, Test 5)

**Root cause:** `OperationsPanel.run_history_record()` launches the background worker without validating that source/target environments exist. If they were deleted, the copy fails silently in the worker thread. Additionally, `EnvironmentFormPanel` may fail to populate fields if `api.get_environment()` returns an unexpected shape.

**Fix:**
1. In `elm/gui/operations_panel.py`, add a `_validate_record_envs(record)` helper that checks `api.list_environments(show_all=True)` for the record's source and target names.
2. In `run_history_record()`, call this helper before creating the worker thread. If environments are missing, show a clear error in `status_label` and return early without starting the thread.
3. In `elm/gui/environment_manager.py`, add defensive handling in `_populate_fields()` for missing keys and ensure `port` is coerced to string before `insert()`.

**Files:** `elm/gui/operations_panel.py`, `elm/gui/environment_manager.py`

## Gap 3: Edit & Re-run Form Pre-fill Fails (major, Test 6)

**Root cause:** `OperationsPanel.pre_fill_form()` sets `source_var` and `target_var` without first refreshing the OptionMenu `values` list. If the environment names from the history record aren't in the currently loaded dropdown values, the OptionMenu doesn't display them and `_on_execute` validation rejects the form.

**Fix:**
1. In `elm/gui/operations_panel.py`, at the top of `pre_fill_form()`, call `self._refresh_environments()` to repopulate the dropdown values from the current environment list.
2. After refreshing, check if `record.get("source")` and `record.get("target")` exist in the current environment names. If an environment is missing, show a warning in `status_label` but still allow the user to see the pre-filled values.
3. Ensure `source_menu` and `target_menu` have their `values` updated before `source_var.set()` / `target_var.set()` is called.

**Files:** `elm/gui/operations_panel.py`

## Gap 4: Auto-refresh After Operation Completion (major, Test 7)

**Root cause:** `OperationsPanel._on_copy_finished()` does not notify `HistoryPanel` to refresh. The History tab only refreshes via its 15-second poll while active. When an operation completes on the Operations tab and the user switches to History, they may see stale data until the next poll cycle.

**Fix:**
1. In `elm/gui/app.py`, store a weak reference or callback bridge so `OperationsPanel` can notify `HistoryPanel` when a copy finishes.
2. In `elm/gui/operations_panel.py`, modify `_on_copy_finished()` to call an optional `on_copy_complete_callback` if set.
3. In `elm/gui/app.py`, wire this callback during `_build_tabs()`: `self.ops_panel.on_copy_complete_callback = self.history_panel._refresh_list`.
4. Ensure the callback is called on the main thread (it already runs via `after` scheduling in `_poll_queue`).
5. Optionally reduce the poll interval from 15000ms to 5000ms for more responsive updates while the History tab is active.

**Files:** `elm/gui/app.py`, `elm/gui/operations_panel.py`, `elm/gui/history_panel.py`

## Verification

After all fixes are applied:
1. Run `python -m pytest tests/test_history_gui.py -v` to ensure existing tests still pass.
2. Launch the GUI and verify: History tab shows without blink, Re-run validates environments, Edit pre-fills correctly, and new operations appear immediately in History after completion.
