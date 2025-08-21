# ELM Tool Tests

This directory contains unit tests for the ELM Tool using the pytest framework.

# To-Do Tests

- environment create
- environment list
- environment show
- environment update
- environment delete
- environment test
- environment execute
- copy db2file
- copy file2db
- copy db2db
- mask add
- mask remove
- mask list
- mask test
- generate data
- generate save

## Running Tests

To run all tests:

```bash
pytest
```

To run tests for a specific module:

```bash
pytest tests/test_environment.py
pytest tests/test_copy.py
pytest tests/test_mask.py
pytest tests/test_generate.py
pytest tests/test_cli.py
```

To run a specific test:

```bash
pytest tests/test_environment.py::test_create_environment
```

## Test Structure

The tests are organized by main command and some functionality:

- `test_environment.py`: Tests for environment management commands
- `test_copy.py`: Tests for data copy commands
- `test_mask.py`: Tests for data masking commands
- `test_generate.py`: Tests for data generation commands
- `test_cli.py`: Tests for the CLI interface

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `temp_env_dir`: Creates a temporary directory for environment files
- `sample_dataframe`: Provides a sample DataFrame for testing
