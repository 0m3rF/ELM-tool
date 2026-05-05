# Phase 05: CLI History Commands - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions captured in CONTEXT.md — this log preserves the discussion.

**Date:** 2026-05-04
**Phase:** 05-cli-history-commands
**Mode:** discuss (default)
**Areas discussed:** 5/5

## Gray Areas Discussed

### 1. List output format
**Question:** How should `elm copy list` display results?
**User answer:** "table layout"
**Captured as:** D-01, D-02, D-03 — human-readable table by default, JSON via `--format json`, newest first sort.

### 2. Re-run behavior
**Question:** Should `elm copy re-run <id>` execute immediately or confirm? Update old entry or create new?
**User answer:** "immediately run. update the old one add last run date."
**Captured as:** D-04, D-05, D-06 — immediate execution, update existing record, add `last_run_date`.

### 3. Edit workflow
**Question:** Which parameters overridable? Preview? New entry or update old?
**User answer:** "everythink should be overridable. Yes show preview, yes add as new entry"
**Captured as:** D-08, D-09, D-10 — all parameters overridable, preview before run, new history entry.

### 4. Filtering & sorting
**Question:** Default limit, filterable fields, sort order?
**User answer:** "default 10 is ok. make it filtrable with every field."
**Captured as:** D-11, D-12, D-13 — filter by any field, default limit 10, newest first.

### 5. Error handling
**Question:** What happens on non-existent ID or missing environment/file?
**User answer:** "add it as new."
**Captured as:** D-14, D-15 — ID-not-found = clear error (no new entry); missing env/file during re-run = record failure as new entry.

## Decisions Summary

| Decision | Choice | Notes |
|----------|--------|-------|
| Default list format | Table | Human-readable |
| Alternative format | JSON | `--format json` flag |
| Re-run confirmation | None | Immediate execution |
| Re-run history update | Update original | Add `last_run_date` |
| Edit history update | New entry | Never updates original |
| Edit preview | Yes | Show modified params before run |
| Overridable params | All | Any original param can be changed |
| Default list limit | 10 | Override with `--limit` |
| Filterable fields | All record fields | `--status`, `--source`, etc. |
| Non-existent ID | Error | Exit non-zero, no history entry |
| Missing env/file on re-run | Record failure | New history entry with failure status |

## Auto-Resolved

None — all areas were discussed interactively.

## External Research

None — no web or library research was needed.

---

*Phase: 05-cli-history-commands*
*Discussion completed: 2026-05-04*
