# Roadmap: ELM-tool GUI

**Granularity:** Standard

## Milestones

- ✅ **v1.0 MVP** — Phases 1-3 (shipped 2026-04-24)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-3) — SHIPPED 2026-04-24</summary>

- [x] Phase 1: Foundation & GUI Bootstrap (1/1 plans) — completed 2026-04-14
  - Intercept empty arguments in `elm.py` to trigger GUI launch
  - CustomTkinter root window with tabbed layout (Environments, Operations)
  - Window closed protocol cleanly terminates process
  - **Covers:** BOOT-01, BOOT-02
- [x] Phase 2: Environment Management Visuals (1/1 plans) — completed 2026-04-23
  - Scrollable environment list with selection highlight
  - Full form for adding/editing environments (write-only passwords)
  - Connection testing and delete confirmation
  - **Covers:** ENV-01, ENV-02, ENV-03, ENV-04
- [x] Phase 3: Execution Engine & Log Streaming (1/1 plans) — completed 2026-04-23
  - Copy Operation visual form with Source/Target dropdowns
  - Background threading with `queue.Queue` stdout redirect
  - Persistent log panel with real-time streaming
  - **Covers:** EXEC-01, EXEC-02, EXEC-03, MON-01, MON-02

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Foundation & GUI Bootstrap | v1.0 | 1/1 | Complete | 2026-04-14 |
| 2. Environment Management Visuals | v1.0 | 1/1 | Complete | 2026-04-23 |
| 3. Execution Engine & Log Streaming | v1.0 | 1/1 | Complete | 2026-04-24 |
| 4. Storage Layer & Recording | v1.1 | 2/2 | Complete | 2026-05-04 |
| 5. CLI History Commands | v1.1 | 1/1 | Complete | 2026-05-04 |
| 6. GUI History Panel | v1.1 | 0/1 | Pending | — |

---

## Upcoming: v1.1 Copy History & Re-Run

### Phase 4: Storage Layer & Recording

**Goal:** Implement persistent storage that records every copy operation with full metadata.

**Requirements:** STOR-01, STOR-02, STOR-03, STOR-04, STOR-05

**Plans:** 2 plans

**Success criteria:**
1. Running `elm copy db2db` (or any copy command) creates a JSON history record
2. History JSON file is readable/writable and co-located with environment config
3. Each record has a unique incrementing integer ID
4. Records contain all operation parameters and timestamps
5. File size stays bounded (FIFO at 100 entries)

Plans:
- [x] 04-01-PLAN.md — Build HistoryRecorder core module and extend infrastructure
- [x] 04-02-PLAN.md — Wire recording into all three copy functions and add tests

### Phase 5: CLI History Commands

**Goal:** Add `copy list`, `copy re-run`, and `copy edit` CLI commands.

**Requirements:** CLIC-01, CLIC-02, CLIC-03, CLIC-04

**Success criteria:**
1. `elm copy list` displays a formatted table of past operations
2. `elm copy re-run <id>` executes the operation with original parameters
3. `elm copy edit <id> --target new_env` overrides specific fields and runs
4. Filtering flags (`--status`, `--source`, `--limit`) work as expected

### Phase 6: GUI History Panel

**Goal:** Add a "History" tab to the GUI with scrollable operations list and re-run/edit actions.

**Requirements:** GUIR-01, GUIR-02, GUIR-03, GUIR-04, GUIR-05

**Success criteria:**
1. New "History" tab appears in the GUI tab bar
2. List shows all past operations with color-coded status
3. Clicking "Re-run" on an entry executes the operation in background with log streaming
4. Clicking "Edit & Re-run" switches to Operations tab with form pre-filled
5. New operations automatically appear in the History list after completion
