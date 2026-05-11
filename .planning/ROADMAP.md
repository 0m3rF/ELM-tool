# Roadmap: ELM-tool GUI

**Granularity:** Standard

## Milestones

- ✅ **v1.0 MVP** — Phases 1-3 (shipped 2026-04-24)
- ✅ **v1.1 Copy History & Re-Run** — Phases 4-6 (shipped 2026-05-06)

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

<details>
<summary>✅ v1.1 Copy History & Re-Run (Phases 4-6) — SHIPPED 2026-05-06</summary>

- [x] Phase 4: Storage Layer & Recording (2/2 plans) — completed 2026-05-04
  - `HistoryRecorder` core module with thread-safe JSON storage and FIFO eviction
  - Wired recording into all three copy functions (`db2db`, `db2file`, `file2db`)
  - **Covers:** STOR-01, STOR-02, STOR-03, STOR-04, STOR-05
- [x] Phase 5: CLI History Commands (1/1 plans) — completed 2026-05-04
  - `elm copy list` with tabular/JSON output and filtering
  - `elm copy re-run <id>` and `elm copy edit <id>` with override flags
  - **Covers:** CLIC-01, CLIC-02, CLIC-03, CLIC-04
- [x] Phase 6: GUI History Panel (1/1 plans) — completed 2026-05-04
  - ⏳ History tab with scrollable operations list and color-coded status badges
  - Re-run and Edit & Re-run actions with auto-refresh on copy completion
  - **Covers:** GUIR-01, GUIR-02, GUIR-03, GUIR-04, GUIR-05

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|-----------------|--------|-----------|
| 1. Foundation & GUI Bootstrap | v1.0 | 1/1 | Complete | 2026-04-14 |
| 2. Environment Management Visuals | v1.0 | 1/1 | Complete | 2026-04-23 |
| 3. Execution Engine & Log Streaming | v1.0 | 1/1 | Complete | 2026-04-24 |
| 4. Storage Layer & Recording | v1.1 | 2/2 | Complete | 2026-05-04 |
| 5. CLI History Commands | v1.1 | 1/1 | Complete | 2026-05-04 |
| 6. GUI History Panel | v1.1 | 1/1 | Complete | 2026-05-04 |
