---
phase: 1
plan: 1
type: foundation
wave: 1
depends_on: []
files_modified:
  - elm/gui/__init__.py
  - elm/gui/app.py
  - elm/elm.py
  - pyproject.toml
  - requirements.txt
autonomous: true
requirements: [BOOT-01, BOOT-02]
---

<objective>
Bootstrap the CustomTkinter GUI foundation into the existing ELM-tool CLI. When the user invokes the CLI entrypoint with zero arguments, the application launches a CustomTkinter window with a tabbed layout (Environments, Operations) using `CTkTabview`, Unicode icons for tab labels, and clean process termination on window close. Existing headless CLI functionality remains completely unaffected.
</objective>

<tasks>

### Task 1: Create GUI package structure

<type>create</type>
<files>elm/gui/__init__.py</files>

<read_first>
- elm/__init__.py (understand existing package structure, version, exports)
- elm/elm.py (see how cli() is structured and where GUI trigger will integrate)
</read_first>

<action>
Create the `elm/gui/` directory and `elm/gui/__init__.py` with:

```python
"""
ELM Tool GUI - CustomTkinter-based graphical interface.

This package provides the GUI layer that launches when the CLI
is invoked without arguments.
"""
```

This is a minimal marker file that establishes the `elm.gui` package namespace. No imports needed here — the app module is lazy-loaded from `elm.py` only when GUI is triggered.
</action>

<acceptance_criteria>
- File `elm/gui/__init__.py` exists
- File contains docstring with `CustomTkinter` keyword
- `python -c "import elm.gui"` executes without error
</acceptance_criteria>

---

### Task 2: Create the main GUI application module

<type>create</type>
<files>elm/gui/app.py</files>

<read_first>
- elm/elm.py (understand the CLI entrypoint pattern)
- elm/elm_utils/variables.py (understand APP_NAME and constants available)
- elm/__init__.py (understand package version `__version__`)
</read_first>

<action>
Create `elm/gui/app.py` with the following implementation:

```python
"""
Main GUI application window for ELM Tool.

Uses CustomTkinter with CTkTabview for tabbed navigation.
Launched by elm.py when CLI is invoked without arguments.
"""

import customtkinter as ctk
import sys


class ELMApp(ctk.CTk):
    """Main ELM Tool GUI application window."""

    WINDOW_TITLE = "ELM Tool"
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(self.WINDOW_TITLE)
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.minsize(640, 480)

        # Center window on screen
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - self.WINDOW_WIDTH) // 2
        y = (screen_h - self.WINDOW_HEIGHT) // 2
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}+{x}+{y}")

        # Appearance: system theme, dark-blue accent
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("dark-blue")

        # Clean shutdown on window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Build the tabbed layout
        self._build_tabs()

    def _build_tabs(self):
        """Create the CTkTabview with Environments and Operations tabs."""
        self.tabview = ctk.CTkTabview(self, anchor="nw")
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs with Unicode icons
        self.tabview.add("🌐  Environments")
        self.tabview.add("⚙  Operations")

        # Set default active tab
        self.tabview.set("🌐  Environments")

        # Placeholder labels (will be replaced in Phase 2 and 3)
        env_label = ctk.CTkLabel(
            self.tabview.tab("🌐  Environments"),
            text="Environment management will appear here.",
            font=("", 14),
        )
        env_label.pack(expand=True)

        ops_label = ctk.CTkLabel(
            self.tabview.tab("⚙  Operations"),
            text="Copy operations will appear here.",
            font=("", 14),
        )
        ops_label.pack(expand=True)

    def _on_close(self):
        """Handle window close: destroy the window and exit the process."""
        self.destroy()
        sys.exit(0)


def launch():
    """Create and run the ELM GUI application. Blocks until window is closed."""
    app = ELMApp()
    app.mainloop()
```

Key implementation details:
- `ELMApp` subclasses `ctk.CTk` (the root window class)
- Window size: 800×600, minimum 640×480, centered on screen
- Appearance: `"system"` mode + `"dark-blue"` color theme (per D-02 agent discretion)
- `CTkTabview` with two tabs: `"🌐  Environments"` and `"⚙  Operations"` (per D-01 top tabview, D-02 Unicode + text)
- `WM_DELETE_WINDOW` protocol calls `self.destroy()` then `sys.exit(0)` for clean termination (BOOT-02)
- `launch()` function is the single entry point called from `elm.py`
</action>

<acceptance_criteria>
- File `elm/gui/app.py` exists
- File contains `class ELMApp(ctk.CTk):`
- File contains `ctk.CTkTabview` usage
- File contains `self.protocol("WM_DELETE_WINDOW", self._on_close)`
- File contains `def launch():` function
- File contains `"🌐  Environments"` and `"⚙  Operations"` tab names
- File contains `ctk.set_appearance_mode("system")`
- File contains `ctk.set_default_color_theme("dark-blue")`
- File contains `sys.exit(0)` in `_on_close`
- `self.geometry("800x600` present in file
</acceptance_criteria>

---

### Task 3: Modify CLI entrypoint to launch GUI on empty args

<type>modify</type>
<files>elm/elm.py</files>

<read_first>
- elm/elm.py (MUST read current full content before editing)
- elm/cli.py (understand the secondary entrypoint wrapping cli())
- elm-tool.py (understand the standalone script entrypoint)
</read_first>

<action>
Modify `elm/elm.py` to intercept the case where `sys.argv` has no subcommand arguments and launch the GUI instead of the CLI.

Add `import sys` at the top of the file (alongside existing `import click, os`).

Replace the `if __name__ == '__main__':` block (lines 34-36) with:

```python
def _should_launch_gui():
    """Return True if CLI was invoked with no arguments (GUI trigger)."""
    return len(sys.argv) == 1


if __name__ == '__main__':
    if _should_launch_gui():
        from elm.gui.app import launch
        launch()
    else:
        venv.create_and_activate_venv(variables.VENV_DIR)
        cli()
```

Also modify the entrypoint function referenced by `pyproject.toml` (`elm.elm:cli`). Since `pyproject.toml` points to `elm.elm:cli` as the console script, we need to wrap the CLI group to handle the no-args case. Add a new wrapper function above the `if __name__` block:

```python
def main():
    """Entrypoint wrapper: launches GUI if no args, else runs CLI."""
    if _should_launch_gui():
        from elm.gui.app import launch
        launch()
    else:
        cli()
```

The full modified `elm/elm.py` should be:

```python
import click, os, sys
from elm.elm_commands import environment, mask, copy, generate, config, sync
from elm.elm_utils import venv, variables
from elm.elm_utils.command_utils import AliasedGroup

def ensure_env_dir():
    """Ensure the environment directory exists."""
    if not os.path.exists(variables.ENVS_FILE):
        os.makedirs(os.path.dirname(variables.ENVS_FILE), exist_ok=True)

@click.group(cls=AliasedGroup)
@click.help_option('-h', '--help')
def cli():
    """Extract, Load and Mask Tool for Database Operations"""
    pass

cli.add_command(environment.environment)
cli.add_command(copy.copy)
cli.add_command(mask.mask)
cli.add_command(generate.generate)
cli.add_command(config.config)
cli.add_command(sync.sync)

# Define aliases for main commands
ALIASES = {
    'env': environment.environment,
    'cpy': copy.copy,
    'msk': mask.mask,
    'gen': generate.generate,
    'cfg': config.config,
    'syn': sync.sync,
}

def _should_launch_gui():
    """Return True if CLI was invoked with no arguments (GUI trigger)."""
    return len(sys.argv) == 1

def main():
    """Entrypoint wrapper: launches GUI if no args, else runs CLI."""
    if _should_launch_gui():
        from elm.gui.app import launch
        launch()
    else:
        cli()

if __name__ == '__main__':
    venv.create_and_activate_venv(variables.VENV_DIR)
    main()
```

Key: The `from elm.gui.app import launch` import is inside the `if` block (lazy-load, per agent discretion D: GUI Module Loading). CustomTkinter is only imported when the GUI is actually needed, keeping CLI startup fast.
</action>

<acceptance_criteria>
- `elm/elm.py` contains `import sys` (or `import click, os, sys`)
- `elm/elm.py` contains `def _should_launch_gui():`
- `elm/elm.py` contains `len(sys.argv) == 1`
- `elm/elm.py` contains `from elm.gui.app import launch` inside a conditional block
- `elm/elm.py` contains `def main():` function
- Running `python -c "from elm.elm import cli; print('ok')"` still works (cli import not broken)
- Running `python -c "from elm.elm import main; print('ok')"` works (main function importable)
</acceptance_criteria>

---

### Task 4: Update pyproject.toml console script entrypoint

<type>modify</type>
<files>pyproject.toml</files>

<read_first>
- pyproject.toml (MUST read current content — particularly [project.scripts] and dependencies)
</read_first>

<action>
Make two changes to `pyproject.toml`:

1. **Update console script** — Change `elm-tool = "elm.elm:cli"` to `elm-tool = "elm.elm:main"` so that the installed `elm-tool` command routes through the `main()` wrapper which handles GUI-vs-CLI dispatch.

2. **Add `customtkinter` dependency** — Add `"customtkinter>=5.2.0"` to the `dependencies` list. Place it after the existing `"click>=8.0.0"` entry.

After changes, the relevant sections should be:

```toml
dependencies = [
    "click>=8.0.0",
    "customtkinter>=5.2.0",
    "cryptography>=3.4.7",
    "sqlalchemy>=2.0.0",
    "pandas>=2.0.0",
    "platformdirs>=3.0.0",
    "configparser>=5.0.0",
    "pyyaml>=6.0",
    "Faker>=37.4.2"
]

[project.scripts]
elm-tool = "elm.elm:main"
```
</action>

<acceptance_criteria>
- `pyproject.toml` contains `elm-tool = "elm.elm:main"`
- `pyproject.toml` contains `"customtkinter>=5.2.0"` in dependencies list
- `pyproject.toml` does NOT contain `elm-tool = "elm.elm:cli"`
</acceptance_criteria>

---

### Task 5: Update requirements.txt

<type>modify</type>
<files>requirements.txt</files>

<read_first>
- requirements.txt (read current content)
</read_first>

<action>
Add `customtkinter>=5.2.0` as a new line in `requirements.txt`.

The file should contain (existing content preserved):
```
click>=8.0.0
cryptography>=3.4.7
sqlalchemy>=2.0.0
pandas>=2.0.0
platformdirs>=3.0.0
configparser>=5.0.0
pyyaml>=6.0
Faker>=37.4.2
customtkinter>=5.2.0
```
</action>

<acceptance_criteria>
- `requirements.txt` contains `customtkinter>=5.2.0`
- All previously existing entries still present
</acceptance_criteria>

</tasks>

<verification>
## Verification Steps

1. **Import check (no GUI launch):**
   ```bash
   python -c "from elm.elm import cli, main, _should_launch_gui; print('imports ok')"
   ```
   Expected: prints `imports ok` with exit code 0.

2. **GUI package importable:**
   ```bash
   python -c "import elm.gui; print('gui package ok')"
   ```
   Expected: prints `gui package ok` with exit code 0.

3. **CLI still works with arguments:**
   ```bash
   python -m elm.elm --help
   ```
   Expected: prints CLI help text (not launch GUI).

4. **CustomTkinter importable (after install):**
   ```bash
   pip install customtkinter>=5.2.0
   python -c "import customtkinter; print(customtkinter.__version__)"
   ```
   Expected: prints version ≥ 5.2.0.

5. **GUI launches on zero args (manual verification):**
   ```bash
   python -m elm.elm
   ```
   Expected: CustomTkinter window opens (800×600, centered), with two tabs: "🌐 Environments" and "⚙ Operations". Closing the window exits the process cleanly (exit code 0).

6. **pyproject.toml entrypoint correct:**
   ```bash
   grep "elm-tool" pyproject.toml
   ```
   Expected: `elm-tool = "elm.elm:main"` (not `:cli`).
</verification>

<must_haves>
- [ ] GUI launches when `elm-tool` or `python elm/elm.py` is invoked with zero arguments (BOOT-01)
- [ ] CustomTkinter window has two tabs via `CTkTabview`: Environments and Operations (D-01)
- [ ] Tabs use Unicode symbols + text labels (D-02)
- [ ] Window closes cleanly with `sys.exit(0)` on close button (BOOT-02)
- [ ] Existing CLI functionality is completely unaffected when arguments are provided
- [ ] CustomTkinter is lazy-loaded only when GUI is triggered (agent discretion)
</must_haves>

## PLANNING COMPLETE
