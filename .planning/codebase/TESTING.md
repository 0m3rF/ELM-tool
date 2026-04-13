# Testing

## Frameworks
- **Primary Framework**: `pytest`
- **Configuration**: Uses `pytest.ini` (`c:\Projeler\ELM-tool\pytest.ini`).

## Structure & Categories
- Test suites reside in `/tests`.
- Tests are heavily componentized:
  - `tests/api/`: testing the API layer.
  - `tests/cli/`: testing the Click application commands.
  - `tests/core/`: testing the logic in `elm/core`.
  - `tests/integration/`: end-to-end functionality involving database engines and full data cycles.

## Execution
Run tests simply using `pytest` command. Test fixtures and shared setup logic are organized inside `tests/conftest.py`.

## Data Mocks
Since it is an ETL tool, rigorous database mocks or lightweight in-memory databases (like SQLite) are used for integration contexts.
