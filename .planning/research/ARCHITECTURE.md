# Architecture Research

## Component Boundaries
1. **Entrypoint Hook**: `elm.py` checks `sys.argv`. If no arguments are provided, it imports and launches the GUI module. Otherwise, it delegates to `click` normally.
2. **GUI Controller**: Manages the life cycle of the CustomTkinter root window.
3. **Execution Thread**: When a user clicks "Run" on a form, the input is serialized into CLI arguments or core engine kwargs, and executed in a background `Thread` or `Process`.
4. **Log Queue**: A thread-safe queue (`queue.Queue`) passing log lines from the Execution Thread back to the main GUI thread via Tkinter's polling (`after()`).

## Data Flow
User clicks "Execute" -> GUI reads form states -> Controller formats kwargs -> Dispatches to `WorkerThread` -> Worker runs `elm/core/` logic -> Logs pushed to `Queue` -> GUI periodically consumes `Queue` and writes to `Log Textbox`.

## Build Order
1. CLI Entrypoint intercepter and GUI package structure.
2. Main GUI window structure and grid layout (Tabs/Frames).
3. Forms for "Environment Management" and "Copy" panels.
4. Execution binding and Threading/Queue/Stdout redirection setup for logs.
