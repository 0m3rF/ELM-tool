# Repository Guidelines

## Project Structure & Architecture

ELM Tool is a Python 3.9+ database utility with three interfaces. Keep interface code thin: `elm/elm.py` and `elm/elm_commands/` provide the Click CLI, `elm/gui/` contains the CustomTkinter app, and `elm/api.py` exposes the Python API. Put business logic in `elm/core/` and shared helpers in `elm/elm_utils/`. Packaged images live in `elm/assets/`; database integration helpers live in `database_setup/`.

Tests mirror these layers under `tests/core/`, `tests/cli/`, `tests/api/`, `tests/utils/`, and `tests/integration/`. Treat `dist/`, `htmlcov/`, and coverage files as generated output.

## Build, Test, and Development Commands

```powershell
python -m pip install -e .                 # Editable install
python -m pip install -r requirements.txt  # Test and common DB dependencies
python -m elm                              # Launch the GUI
python -m elm environment list             # Run a CLI command
pytest tests -m "not integration"          # Tests without a live database
pytest tests/core/test_copy.py -v           # One test module
pytest tests -n auto                        # Full suite in parallel
python -m build                             # Build wheel and sdist
```

`pytest.ini` generates terminal, HTML, and XML coverage reports and sets a 30-second timeout. Install `build` and `pytest-timeout` if needed.

## Coding Style & Naming Conventions

Use four-space indentation and the existing PEP 8-oriented style. Name modules, functions, fixtures, and variables with `snake_case`; classes with `PascalCase`; constants with `UPPER_SNAKE_CASE`. Preserve nearby type hints and docstrings. No formatter or linter is configured, so avoid unrelated reformatting.

## Testing Guidelines

Name files `test_<module>.py`, classes `Test<Subject>`, and tests `test_<behavior>_<scenario>`. Mock dependencies at layer boundaries; reserve live database workflows for `integration` or `db_access`. Other markers are `file_io` and `serial`. Add regression tests for fixes and cover success and failure paths. No coverage threshold is configured, but coverage should not materially regress.

## Commits & Pull Requests

Recent history primarily uses imperative Conventional Commit subjects such as `feat(gui): ...`, `fix(gui): ...`, `test(06-01): ...`, and `docs: ...`; follow that pattern and keep commits focused. Pull requests should explain the behavior change, link relevant issues, list test commands and results, and include screenshots for GUI changes. Call out database requirements, configuration changes, or compatibility risks.

## Security & Agent Workflow

Never commit database credentials, encryption keys, or generated environment files. Avoid production databases during development. State assumptions before coding, prefer the smallest viable change, touch only relevant code, and verify each change with targeted tests before the broader suite.
