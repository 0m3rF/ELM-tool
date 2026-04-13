# Code Conventions

## Code Style
- **Python**: PEP 8 standards are generally expected. Use snake_case for functions and variables, PascalCase for classes.
- **Imports**: standard library imports, third-party imports, and local application/library specific imports in separate blocks.

## Development Patterns
- **CLI Commands**: CLI entry points use decorators from `click`. Core logic is deliberately kept out of CLI command files and delegated to `core` or `elm_commands` scripts.
- **Error Handling**: Custom exceptions are likely defined (`exceptions.py` inside `core`). Exceptions should be bubbled up to the CLI layer where proper formatted error messages are presented to the user.

## Data Masking Patterns
- Re-usability is a key aspect, `elm/elm_utils/mask_algorithms.py` provides the basic transformations which are then hooked into the main pipeline via `elm/core/masking.py`.
