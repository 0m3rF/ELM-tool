# Codebase Structure

## Directory Layout

- `/elm` : The main application package.
  - `/core` : Core business logic scripts driving the main operations (`copy.py`, `masking.py`, `metadata_sync.py`, etc.).
  - `/elm_commands` : CLI command handlers connecting user inputs to the core engine.
  - `/elm_utils` : Helpers, shared utilities, and specialized handlers (e.g., `db_utils.py`, `encryption.py`, `random_data.py`).
  - `api.py` : Programmatic API layer for interactions beyond CLI scripts.
  - `cli.py` & `elm.py` : Entry points for the `click` application.
  - `models.py` : Data structures or ORM models.
- `/tests` : Pytest suites mirroring the application structure.
  - `/api` : API integration tests.
  - `/cli` : CLI endpoint tests.
  - `/core` : Core unit tests.
  - `/integration` : End-to-end and data flow tests.
  - `/utils` : Utilities tests.

## Naming Conventions
- Snake case for files and functions (`mask_algorithms.py`).
- Clear separation between logical modules (`_commands`, `_utils`, `core`).
