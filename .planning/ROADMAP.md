# Roadmap: ELM-tool GUI

**Granularity:** Standard

## Phase 1: Foundation & GUI Bootstrap
**Goal:** Integrate CustomTkinter into the CLI entrypoint without breaking existing headless functionality.
- [ ] Intercept empty arguments in `elm.py` to trigger GUI launch.
- [ ] Set up CustomTkinter root window and basic tabbed layout (Environments, Operations).
- [ ] Connect window closed protocol to cleanly terminate the process.
**Covers:** BOOT-01, BOOT-02

## Phase 2: Environment Management Visuals
**Goal:** Create forms to list, add, edit, and delete environments perfectly mirroring CLI capabilities.
- [ ] Build environment listing UI (e.g., dropdowns or listbox).
- [ ] Build form fields (Host, Port, User, Pass, DB, etc.) for adding/editing environments.
- [ ] Connect form buttons to core `elm/core/environment.py` operations.
**Covers:** ENV-01, ENV-02, ENV-03, ENV-04

## Phase 3: Execution Engine & Log Streaming
**Goal:** Build the execution thread and connect `sys.stdout` streaming to an observable log panel.
- [ ] Build the Copy Operation visual form (Source, Target, Execution Buttons).
- [ ] Implement `threading` and `queue.Queue` to run `elm/core` commands without blocking the main event loop.
- [ ] Implement the persistent log panel (`CTkTextbox`) and consume the queue to display logs natively in real-time.
**Covers:** EXEC-01, EXEC-02, EXEC-03, MON-01, MON-02
