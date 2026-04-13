# Pitfalls Research

## Pitfalls & Prevention Strategies

1. **Blocking the Main Loop**
   - *Warning Sign*: The GUI freezes and says "Not Responding" on Windows while copying data.
   - *Prevention*: Never run `click.Context.invoke` or `pandas` data processing directly in the `command` callback of a Tkinter button. Always wrap in a Thread.
   - *Phase*: Phase 2/3 (Execution Engine).

2. **Print Statement Capture**
   - *Warning Sign*: `click.echo` or `logger.info` outputs to the background terminal, but the GUI log panel remains empty.
   - *Prevention*: Because the GUI runs in the same process natively, simply redirecting `sys.stdout` to a custom Stream object that writes to the GUI's queue is the most robust way to capture all core logs without rewriting the tool's logging stack.
   - *Phase*: Phase 2/3 (Execution Engine).

3. **Packaging Size Bloat**
   - *Warning Sign*: PIP install of the tool jumps from 20MB to 200MB.
   - *Prevention*: Ensure CustomTkinter and PIL are the only major non-OS dependencies added. Avoid WebKit engines.
   - *Phase*: Phase 1 (Setup).
