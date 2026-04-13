# Requirements: ELM-tool GUI

**Defined:** 2026-04-14
**Core Value:** A visual interface that makes ELM-tool's environment management and copying features accessible and intuitive, bridging the gap between command-line power and GUI ease-of-use.

## v1 Requirements

### Integration and Launch

- [ ] **BOOT-01**: GUI automatically launches via `CustomTkinter` when the user invokes the main CLI entrypoint without any arguments.
- [ ] **BOOT-02**: GUI terminates completely when the user closes the root window.

### Environment Management

- [ ] **ENV-01**: User can view a list of all configured environments via the GUI.
- [ ] **ENV-02**: User can create a new environment via a visual form.
- [ ] **ENV-03**: User can edit an existing environment's credentials/details via a visual form.
- [ ] **ENV-04**: User can delete an existing environment via the GUI.

### Task Execution (Copy)

- [ ] **EXEC-01**: User can select a source environment and target environment from dropdowns to prepare a copy task.
- [ ] **EXEC-02**: User can trigger the copy/mask logic by clicking a "Run" or "Execute" button.
- [ ] **EXEC-03**: The execution of backend tasks must run in a background thread and not freeze the CustomTkinter UI.

### Observability

- [ ] **MON-01**: A persistent log panel exists at the bottom or side of the GUI.
- [ ] **MON-02**: Standard output (`stdout`) and standard error (`stderr`) from the core CLI execution are redirected to and rendered in the log panel in real-time.

## v2 Requirements

### Advanced Configuration

- **ADV-01**: User can configure fine-grained masking rules via a visual table builder instead of JSON/YAML.
- **ADV-02**: Visual progress bars deriving percentages from log outputs.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Web Browser Dashboard | Requires spinning up a server, violates the "native openable window" requirement. |
| PySide6/PyQt6 Backend | Introduces too much bloat and dependency overhead for a CLI tool wrapper. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BOOT-01 | Phase 1 | Pending |
| BOOT-02 | Phase 1 | Pending |
| ENV-01 | Phase 2 | Pending |
| ENV-02 | Phase 2 | Pending |
| ENV-03 | Phase 2 | Pending |
| ENV-04 | Phase 2 | Pending |
| EXEC-01 | Phase 3 | Pending |
| EXEC-02 | Phase 3 | Pending |
| EXEC-03 | Phase 3 | Pending |
| MON-01 | Phase 3 | Pending |
| MON-02 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-14*
*Last updated: 2026-04-14 after initial definition*
