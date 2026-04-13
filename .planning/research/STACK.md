# Stack Research

## Objective
Identify the standard stack for a native cross-platform GUI that integrates tightly with a Python CLI data tool (`click`, `pandas`, `sqlalchemy`).

## Recommended Stack: CustomTkinter

1. **CustomTkinter** (Confidence: High)
   - **Rationale**: Minimal footprint compared to PySide6/PyQt6. Uses the built-in Tkinter framework but modernizes the design with dark modes, rounded corners, and cleaner API. No large C++ dependencies, which prevents inflating the `hatchling` distribution size. Very suitable for mapping straightforward visual forms to CLI arguments.
   - **Version**: >= 5.2.0

2. **Subprocess / Threading** (Confidence: High)
   - **Rationale**: To capture standard output from the core logic (or invoke the CLI natively in-process) without blocking the GUI event loop, `concurrent.futures` or basic `threading` paired with `queue.Queue` is standard practice.

## What NOT To Use
- **PyQt6 / PySide6**: Excellent but way too heavy for a typical CLI wrapper tool. Inflates package size significantly (often 100MB+ overhead).
- **Flet**: Relies on a Flutter engine backend. Great for web, but for a true "run anywhere the CLI runs" lightweight native feel, CustomTkinter is simpler since it depends only on Python standard libraries augmented with minimal drawing wrappers.
