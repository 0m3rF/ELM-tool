# Phase 06: GUI History Panel - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-05-04
**Phase:** 06-gui-history-panel
**Mode:** discuss (default)
**Areas discussed:** Display format, Re-run log streaming, Auto-refresh, Edit & Re-run navigation

---

## Questions & Answers

### Area 1: History Entry Display Format

**Question:** Compact table rows or detailed cards?

**Options presented:**
- Compact table rows (reuses list button pattern from EnvironmentListPanel)
- Detailed cards (more info per entry, needs new component)

**Answer:** Compact table rows

**Rationale (user):** "use compact tables"

---

### Area 2: Re-run Log Streaming

**Question:** When re-run is clicked from History, where does output go?

**Options presented:**
- Operations tab log panel (existing, consistent with current behavior)
- New log area inside History tab
- Modal popup window with log

**Answer:** Operations tab log panel ("add like operations tab")

**Rationale (user):** "add like operations tab" — output should go to the existing Operations log panel

---

### Area 3: Auto-Refresh Strategy

**Question:** How does the History list stay up to date?

**Options presented:**
- Refresh only on tab switch
- Real-time push from worker thread
- Scheduled polling interval

**Answer:** Refresh on tab switch + scheduled poll every 15 seconds

**Rationale (user):** "refresh only on tab switch. also add a scheduled job to check every 15 seconds"

---

### Area 4: Edit & Re-run Navigation

**Question:** When "Edit & Re-run" is clicked, what happens?

**Options presented:**
- Auto-switch to Operations tab and pre-fill form
- Pre-fill form but let user manually switch tabs
- Open an inline edit dialog in History tab

**Answer:** Auto-switch to Operations tab and pre-fill the form

**Rationale (user):** "the GUI automatically switch to the Operations tab and pre-fill the form"

---

## Summary

| Area | Decision |
|------|----------|
| Display format | Compact table rows |
| Re-run log streaming | Operations tab log panel |
| Auto-refresh | Tab switch + 15s poll |
| Edit & Re-run | Auto-switch + pre-fill Operations form |

All four gray areas resolved in a single turn. No scope creep detected.
