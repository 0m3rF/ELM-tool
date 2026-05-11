# ELM-tool GUI

## What This Is

A cross-platform native GUI for the ELM-tool (Extract, Load, Mask) that launches automatically when the CLI is executed without parameters. It provides visual forms and buttons for environment management and copying workflows, accompanied by a persistent log panel that streams real-time execution feedback.

## Core Value

A visual interface that makes ELM-tool's environment management and copying features accessible and intuitive, bridging the gap between command-line power and GUI ease-of-use.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- ✓ [CLI Core] Extract, Load, Mask data pipeline (`elm/core/`) — existing
- ✓ [CLI Config] Environment configuration and cross-platform storage — existing
- ✓ [CLI Logic] Command routing for core operations via Click — existing
- ✓ [GUI-01] GUI opens automatically when the CLI entrypoint is executed with no arguments/parameters — v1.0
- ✓ [GUI-02] GUI presents visual forms and buttons for editing, adding, removing, and listing environments — v1.0
- ✓ [GUI-03] GUI allows executing data copy tasks between environments via visual controls — v1.0
- ✓ [GUI-04] GUI includes a dedicated, persistent log panel to stream the raw text output of underlying command executions — v1.0
- ✓ [GUI-05] GUI window stays open until the user actively closes it or terminates the application — v1.0
- ✓ [GUI-06] GUI is cross-platform, natively supporting Windows, Linux, and macOS — v1.0 (CustomTkinter)
- ✓ [STOR-01..05] Persistent JSON history storage with auto-incrementing IDs and FIFO eviction — v1.1
- ✓ [CLIC-01..04] CLI `copy list`, `copy re-run`, `copy edit` commands with filtering and preview — v1.1
- ✓ [GUIR-01..05] GUI History tab with scrollable list, color-coded status, re-run/edit, auto-refresh — v1.1

### Active

<!-- Current scope. Building toward these. -->

- [ ] [GUI-07] Mock-based unit tests for GUI widgets and threading logic (headless CI support) — partial: 9 tests in test_history_gui.py
- [ ] [GUI-08] Visual masking rules builder (table-based, instead of JSON/YAML)
- [ ] [GUI-09] Progress bars for long-running copy operations
- [ ] [GUI-10] Export/import environment configurations
- [ ] [GUI-11] History entry delete/undo functionality
- [ ] [GUI-12] History export to CSV/JSON

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- [Web/Browser Application] — Excluded because the user specifically requested a native openable window directly tied to the CLI tool.
- [Background Service] — GUI is a foreground desktop app tied to interactive usage.
- [Cross-machine sync] — Requires server/cloud storage; out of scope for local-first tool.
- [Full undo/rollback] — Too complex; requires transaction log beyond history metadata.

## Context

ELM-tool is a Python-based CLI + GUI hybrid utilizing `click`, `pandas`, `sqlalchemy`, `Faker`, and `customtkinter`.

**v1.0** (2026-04-24): GUI MVP with environment management, copy execution, and real-time log streaming. ~865 LOC GUI code across 3 modules. 345 core tests passing.

**v1.1** (2026-05-06): Copy History & Re-Run. Persistent JSON history storage (`elm/core/history.py`), CLI history commands (`elm/elm_commands/copy.py`), and GUI History panel (`elm/gui/history_panel.py`). ~22,789 total LOC. 29+ tests for history features. All 14 v1.1 requirements satisfied.

## Constraints

- **Language/Toolkit**: Python-native GUI (CustomTkinter). Already shipped and working.
- **Compatibility**: Cross-platform (Windows, macOS, Linux) via CustomTkinter + tkinter.
- **Integration**: Must preserve existing `click` CLI headless functionality. Already verified.
- **Build System**: `hatchling` + PIP. `customtkinter>=5.2.0` added to dependencies.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Visual Form UI | Prioritize visual inputs for commands rather than a generic text command palette | ✓ Delivered in v1.0 |
| Dedicated Log Panel | Essential for streaming backend execution output directly into the UI | ✓ Delivered in v1.0 |
| CustomTkinter over PySide6 | Lighter dependency footprint, easier PIP install | ✓ Correct choice |
| Lazy GUI imports | Import GUI modules only when GUI is triggered, keeping CLI startup fast | ✓ Working well |
| Inside-method imports | Place GUI imports inside `_build_tabs` methods to prevent circular imports at module level | ✓ Working well |
| JSON file for history storage | Simple, human-readable, no external DB dependency | ✓ Delivered in v1.1 |
| FIFO eviction at 100 entries | Bounded growth, minimal config | ✓ Delivered in v1.1 |
| MagicMock with unbound method binding for headless GUI tests | Avoids Tk display requirements in CI | ✓ Working well in v1.1 |
| Record-dispatch pattern in `_copy_worker` | Single worker handles all three operation types from a record dict | ✓ Delivered in v1.1 |
| Diff-based widget refresh via JSON hash | Eliminates full destroy/rebuild cycle causing UI blinking | ✓ Fixed in v1.1 UAT gap |

---
*Last updated: 2026-05-06 after v1.1 milestone*

## Current State

**Version:** v1.0 shipped ✓ | v1.1 shipped ✓
**Status:** All 6 phases complete. 25 total tests for history features. Ready for next milestone planning.
**Next focus:** Planning v1.2 — potential areas: masking rules builder, progress bars, environment export/import, history delete/export.

## Milestone History

### v1.1 Copy History & Re-Run — SHIPPED 2026-05-06

**Phases:** 4, 5, 6 | **Plans:** 4 | **Commits:** 22 | **Files:** 56 changed (+5,773 / −55)

**What shipped:**
- Persistent JSON history storage with thread-safe reads/writes and FIFO eviction
- CLI `copy list`, `copy re-run`, `copy edit` commands with full filtering and preview
- GUI History tab with scrollable operations list, color-coded status badges, re-run/edit actions
- Auto-refresh on copy completion via callback wiring
- 9 mock-based headless GUI tests + 14 CLI history tests + 6 storage tests

**Key lessons:**
- CustomTkinter `CTkLabel.configure(text_color=None)` raises ValueError — must not pass `None` for color
- Tkinter widget polling with `after()` is adequate but not a proper event bus architecture
- Diff-based refresh (record hash comparison) is essential for smooth UI updates in scrollable lists

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state
