---
status: testing
phase: 06-gui-history-panel
source: 06-01-SUMMARY.md
started: 2026-05-05T00:00:00Z
updated: 2026-05-05T00:00:00Z
---

## Current Test

number: 1
name: History Tab Presence
expected: |
  Launch the ELM GUI. In the tab bar at the top of the window, there are three tabs: "Environments", "Operations", and "⏳  History". The History tab is clickable and switches to a new panel when selected.
awaiting: user response

## Tests

### 1. History Tab Presence
expected: Three tabs visible in GUI tab bar including "⏳  History". Clicking it shows the History panel.
result: pending

### 2. History List Display
expected: If past copy operations exist, the History panel shows a scrollable list of rows. Each row displays ID, date, operation type, source/target, table, and a color-coded status badge (green for success, red for failure, blue for running).
result: pending

### 3. Empty State
expected: If no copy operations have been run yet, the History panel shows "No history yet" heading with body text "Run a copy operation to see it here." instead of the list.
result: pending

### 4. Tab Switch Refresh
expected: Clicking onto the History tab refreshes the list from storage. A status label at the top shows "Last refreshed: {timestamp}" reflecting the latest load time.
result: pending

### 5. Re-run Operation
expected: Clicking the "▶ Re-run" button on a history entry switches to the Operations tab and executes the copy with original parameters in a background thread. Log output streams to the existing log panel.
result: pending

### 6. Edit & Re-run
expected: Clicking the "✎ Edit" button on a history entry switches to the Operations tab and pre-fills the copy form (source, target, query, table, mode, batch size) with the original parameters. User can modify and click Execute.
result: pending

### 7. Auto-refresh After New Operation
expected: After a copy operation completes from the Operations tab, switching to the History tab (or waiting ~15 seconds while on it) shows the new operation in the list automatically.
result: pending

## Summary

total: 7
passed: 0
issues: 0
pending: 7
skipped: 0
blocked: 0

## Gaps

[none yet]
