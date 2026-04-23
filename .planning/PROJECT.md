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

### Active

<!-- Current scope. Building toward these. -->

- [ ] [GUI-07] Mock-based unit tests for GUI widgets and threading logic (headless CI support)
- [ ] [GUI-08] Visual masking rules builder (table-based, instead of JSON/YAML)
- [ ] [GUI-09] Progress bars for long-running copy operations
- [ ] [GUI-10] Export/import environment configurations

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- [Web/Browser Application] — Excluded because the user specifically requested a native openable window directly tied to the CLI tool.
- [Background Service] — GUI is a foreground desktop app tied to interactive usage.

## Context

ELM-tool is a Python-based CLI + GUI hybrid utilizing `click`, `pandas`, `sqlalchemy`, `Faker`, and `customtkinter`. The v1.0 GUI provides a visual layer on top of the established CLI logic for environment management and data copying with real-time execution observability. The core CLI remains fully functional and unaffected when invoked with arguments.

Shipped v1.0 with ~865 LOC of GUI code across 3 modules (`app.py`, `environment_manager.py`, `operations_panel.py`). All existing core tests pass (345 tests).

## Constraints

- **Language/Toolkit**: Python-native GUI (CustomTkinter). Already shipped and working.
- **Compatibility**: Cross-platform (Windows, macOS, Linux) via CustomTkinter + tkinter.
- **Integration**: Must preserve existing `click` CLI headless functionality. Already verified.
- **Build System**: `hatchling` + PIP. `customtkinter>=5.2.0` added to dependencies.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Visual Form UI | Prioritize visual inputs for commands rather than a generic text command palette | ✓ Delivered in v1.0 |
| Dedicated Log Panel | Essential for streaming backend execution output directly into the UI | ✓ Delivered in v1.0 |
| CustomTkinter over PySide6 | Lighter dependency footprint, easier PIP install | ✓ Correct choice |
| Lazy GUI imports | Import GUI modules only when GUI is triggered, keeping CLI startup fast | ✓ Working well |
| Inside-method imports | Place GUI imports inside `_build_tabs` methods to prevent circular imports at module level | ✓ Working well |

---
*Last updated: 2026-04-24 after v1.0 milestone completion*

## Current State

**Version:** v1.0 shipped ✓
**Status:** GUI MVP complete. All 11 v1 requirements satisfied.
**Next focus:** v1.1 — testing, masking builder, progress bars

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
