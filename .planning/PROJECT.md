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

### Active

<!-- Current scope. Building toward these. -->

- [ ] [GUI-01] GUI opens automatically when the CLI entrypoint is executed with no arguments/parameters.
- [ ] [GUI-02] GUI presents visual forms and buttons for editing, adding, removing, and listing environments.
- [ ] [GUI-03] GUI allows executing data copy tasks between environments via visual controls.
- [ ] [GUI-04] GUI includes a dedicated, persistent log panel to stream the raw text output of underlying command executions.
- [ ] [GUI-05] GUI window stays open until the user actively closes it or terminates the application.
- [ ] [GUI-06] GUI is cross-platform, natively supporting Windows, Linux, and macOS.

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- [Web/Browser Application] — Excluded because the user specifically requested a native openable window directly tied to the CLI tool.
- [Background Service] — GUI is a foreground desktop app tied to interactive usage.

## Context

ELM-tool is currently a Python-based CLI utilizing `click`, `pandas`, `sqlalchemy`, and `Faker` for extracting, obfuscating, and loading data across different databases. The new addition needs to provide a visual layer on top of this established logic to simplify the workflow and provide better real-time execution observability without forcing the user to memorize command flags.

## Constraints

- **Language/Toolkit**: Since the existing core is Python (>= 3.7), the GUI framework should be Python-native (e.g., PyQt, Tkinter, CustomTkinter, or PySide) or bundleable with a Python backend.
- **Compatibility**: Requires seamless cross-platform support (Windows, macOS, Linux).
- **Integration**: Must align with the existing `click` CLI without breaking standard headless command usage.
- **Build System**: Must integrate gracefully with `hatchling` and PIP distribution methods.

## Key Decisions

<!-- Decisions that constrain future work. Add throughout project lifecycle. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Visual Form UI | Prioritize visual inputs for commands rather than a generic text command palette | — Pending |
| Dedicated Log Panel | Essential for streaming backend execution output directly into the UI | — Pending |

---
*Last updated: 2026-04-14 after initialization*

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
