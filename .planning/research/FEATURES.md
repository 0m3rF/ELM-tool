# Features Research

## Objective
Identify table stakes vs differentiating features for a CLI GUI wrapper handling data pipelines.

## Table Stakes
- **CLI Arg Mapping**: Forms, dropdowns, and checkboxes that perfectly reflect the `click` options and arguments for environment management and copy tasks.
- **Log Panel**: A `Textbox` or `ScrolledText` widget streaming real-time stdout/stderr from executing tasks.
- **Non-blocking Execution**: The UI must remain responsive while a long-running copy or mask task executes.
- **Persistence**: Ability to remember the last used environment or settings between launches.

## Differentiators
- **Dry Run Toggles**: Allowing the user to easily click "Test Connection" or "Dry Run" before executing destructive loads/masks without memorizing flags.

## Anti-Features (Do Not Build)
- **Web Dashboard**: Do not spin up a Flask/FastAPI server to host a UI. Keep it as an interactive desktop window.
