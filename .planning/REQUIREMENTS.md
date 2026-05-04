# Requirements: ELM-tool GUI v1.1

**Defined:** 2026-05-04
**Core Value:** A visual interface that makes ELM-tool's environment management and copying features accessible and intuitive, bridging the gap between command-line power and GUI ease-of-use.

## v1.1 Requirements (Active)

### Storage & Persistence (STOR)

- [ ] **STOR-01**: Every copy operation (`db2db`, `db2file`, `file2db`) is automatically recorded with its full parameter set and outcome
- [ ] **STOR-02**: History records are stored in a lightweight JSON file alongside existing environment config
- [ ] **STOR-03**: Each history record has a unique auto-incrementing integer ID
- [ ] **STOR-04**: History records capture: operation type, source, target, query/table, mode, batch_size, parallel, timestamps (start/end), status (success/failure), record count, and error message if failed
- [ ] **STOR-05**: History storage is bounded (configurable max entries, default 100) with FIFO eviction

### CLI Commands (CLIC)

- [ ] **CLIC-01**: User can list previous copy operations via `elm copy list` showing: ID, timestamp, operation type, source → target, table, status
- [ ] **CLIC-02**: User can re-run a previous copy operation via `elm copy re-run <id>` using all original parameters
- [ ] **CLIC-03**: User can edit and re-run a previous copy operation via `elm copy edit <id>` with optional flags to override any original parameter
- [ ] **CLIC-04**: `elm copy list` supports optional filters: `--status`, `--source`, `--target`, `--limit`

### GUI History Panel (GUIR)

- [ ] **GUIR-01**: GUI displays a new "History" tab with a scrollable list of previous copy operations
- [ ] **GUIR-02**: Each history entry shows: ID, timestamp, operation type, source → target, table, status (color-coded)
- [ ] **GUIR-03**: Each history entry has a "Re-run" button that executes the operation again with original parameters
- [ ] **GUIR-04**: Each history entry has an "Edit & Re-run" button that pre-fills the Operations tab form with that entry's parameters
- [ ] **GUIR-05**: History list automatically refreshes after a new copy operation completes

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full undo/rollback | Too complex for v1.1; requires transaction log |
| History export to CSV/JSON | Nice-to-have, defer to v1.2 |
| Cross-machine sync | Requires server/cloud storage, out of scope |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STOR-01 | Phase 4 | Pending |
| STOR-02 | Phase 4 | Pending |
| STOR-03 | Phase 4 | Pending |
| STOR-04 | Phase 4 | Pending |
| STOR-05 | Phase 4 | Pending |
| CLIC-01 | Phase 5 | Pending |
| CLIC-02 | Phase 5 | Pending |
| CLIC-03 | Phase 5 | Pending |
| CLIC-04 | Phase 5 | Pending |
| GUIR-01 | Phase 6 | Pending |
| GUIR-02 | Phase 6 | Pending |
| GUIR-03 | Phase 6 | Pending |
| GUIR-04 | Phase 6 | Pending |
| GUIR-05 | Phase 6 | Pending |

**Coverage:**
- v1.1 requirements: 14 total
- Mapped to phases: 0 (pending roadmap creation)
- Unmapped: 14 ⚠️

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-05-04 after milestone start*
