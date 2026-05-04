# Phase 02: Environment Management Visuals - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Visual forms and list views for creating, editing, and listing database environments.

</domain>

<decisions>
## Implementation Decisions

### UI Layout Strategy
- **D-01:** Split Pane: List of environments on the left, a details/edit form on the right.

### Sensitive Data Handling
- **D-02:** Always Masked: Passwords are write-only. If you need to change it, you type a new one; you can never see the existing one.

### Connection Verification Flow
- **D-03:** On Demand: User explicitly clicks a "Test Connection" button on the form.

### the agent's Discretion
- Validation error message formats and precise widget arrangements (e.g., padding, colors, specific input placements inside the split pane).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Implementation Reference
- `elm/api.py` — Programmatic interface providing `list_environments`, `create_environment`, `update_environment`, etc.
- `elm/gui/app.py` — The main CTkTabview where the Environment Manager needs to be injected.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `elm/gui/app.py`: Sets up `CTkTabview` with a "🌐  Environments" tab. The new UI will replace the placeholder label.
- `elm/api.py`: Existing functions ready to be connected (e.g., `test_environment()`, `create_environment()`, `update_environment()`).

### Integration Points
- Add the split-pane layout to the "🌐  Environments" tab inside `_build_tabs` of `ELMApp`.

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard CustomTkinter approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-environment-management-visuals*
*Context gathered: 2026-04-14*
