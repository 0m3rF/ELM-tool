# Phase 01: foundation-gui-bootstrap - Context

**Gathered:** 2026-04-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Integrate CustomTkinter into the CLI entrypoint without breaking existing headless functionality. Includes intercepting empty arguments in `elm.py` to trigger GUI launch, setting up the root window, basic tabbed layout, and cleanly terminating the process on close.

</domain>

<decisions>
## Implementation Decisions

### Tab Layout Style
- **D-01:** Use a top `CTkTabview` for the overall navigation structure.
- **D-02:** Use simple Unicode symbols + Text for the navigation item style.

### the agent's Discretion
- GUI Module Loading: Lazy-load CustomTkinter `gui` module only when triggered via empty arguments (prevents slowing down the core CLI).
- Visual Theme: Default to 'system' appearance and 'dark-blue' theme.
- App Window Defaults: Default window size of 800x600, positioned center-screen, and resizable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Implementation Reference
- `elm/elm.py` — The main CLI entrypoint where the intervention needs to happen.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `elm/elm.py`: Contains existing Click CLI group setup. No GUI code exists yet.

### Integration Points
- Intercept logic in `elm/elm.py` right before `cli()` using `if len(sys.argv) == 1:`

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

*Phase: 01-foundation-gui-bootstrap*
*Context gathered: 2026-04-14*
