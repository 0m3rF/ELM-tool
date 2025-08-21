# ELM Tool Test Suite

This directory contains the comprehensive test suite for the ELM Tool, organized by architectural layers to ensure proper separation of concerns and testing methodologies.

## Test Structure

The test suite is organized into the following directories, each with a specific focus and testing approach:

### 📁 `api/` - API Layer Tests
Tests for the programmatic API interface (`elm.api` module).

**Focus**: Test that API functions properly delegate to core modules and return expected data structures.

**Methodology**:
- Test API functions directly (e.g., `elm.api.create_environment()`)
- Mock core module dependencies using `@patch('elm.core.module.function')`
- Verify correct parameter passing to core modules
- Test return value transformation and error handling
- No CLI command testing or subprocess calls

**Example**:
```python
@patch('elm.core.environment.create_environment')
def test_create_environment(self, mock_core_create):
    mock_core_create.return_value = OperationResult(success=True, message="Success")
    result = api.create_environment(name="test", host="localhost", ...)
    assert result is True
    mock_core_create.assert_called_once_with(name="test", host="localhost", ...)
```

### 📁 `cli/` - Command Line Interface Tests
Tests for the CLI commands using Click's testing framework.

**Focus**: Test actual command-line interface behavior, argument parsing, output formatting, and user interaction.

**Methodology**:
- Use `click.testing.CliRunner` for isolated CLI testing
- Test command-line argument parsing and validation
- Verify CLI output formatting and error messages
- Test exit codes and error handling
- Mock core modules to focus on CLI behavior
- No direct API function calls

**Example**:
```python
@patch('elm.core.environment.create_environment')
def test_environment_create_success(self, mock_core_create, runner):
    mock_core_create.return_value = OperationResult(success=True, message="Created")
    result = runner.invoke(cli, ['environment', 'create', '--name', 'test', ...])
    assert result.exit_code == 0
    assert "Created" in result.output
```

### 📁 `core/` - Core Business Logic Tests
Tests for the core modules that contain the unified business logic.

**Focus**: Test business logic in isolation with comprehensive coverage of edge cases and error conditions.

**Methodology**:
- Test core module functions directly
- Mock external dependencies (file I/O, database connections, etc.)
- Focus on business logic validation and data transformation
- Test error handling and edge cases
- Use standard unit testing practices
- No CLI or API layer dependencies

**Example**:
```python
@patch('elm.core.masking.save_masking_definitions')
@patch('elm.core.masking.load_masking_definitions')
def test_add_mask_global(self, mock_load, mock_save):
    mock_load.return_value = {'global': {}, 'environments': {}}
    result = masking.add_mask(column="password", algorithm="star")
    assert result.success is True
```

### 📁 `utils/` - Utility Function Tests
Tests for utility modules and helper functions.

**Focus**: Test individual utility functions in isolation.

**Methodology**:
- Test utility functions directly
- Focus on specific functionality (masking algorithms, data validation, etc.)
- Test edge cases and error conditions
- Mock external dependencies as needed
- Standard unit testing practices

**Files**:
- `test_mask_algorithms.py` - Tests for masking algorithm functions
- `test_data_utils.py` - Tests for data manipulation utilities
- `test_random_data.py` - Tests for random data generation utilities
- `test_encryption.py` - Tests for encryption/decryption utilities

### 📁 `others/` - Integration and Miscellaneous Tests
Tests that don't fit into the above categories.

**Focus**: Integration tests, Docker tests, environment setup tests, etc.

## Testing Principles

### 1. **Layer Separation**
- **API tests** only test the API layer and mock core modules
- **CLI tests** only test the command-line interface and mock core modules
- **Core tests** only test business logic and mock external dependencies
- **No cross-contamination** between layers

### 2. **Proper Mocking**
- Mock at the **boundary** of the layer being tested
- API tests mock `elm.core.*` modules
- CLI tests mock `elm.core.*` modules
- Core tests mock external dependencies (files, databases, etc.)

### 3. **Test Focus**
- **API tests**: Parameter passing, return value transformation, error handling
- **CLI tests**: Argument parsing, output formatting, exit codes, user experience
- **Core tests**: Business logic, data validation, edge cases, error conditions
- **Utils tests**: Individual function behavior, algorithm correctness

### 4. **Naming Conventions**
- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>` (for grouping related tests)
- Test methods: `test_<functionality>_<scenario>`

### 5. **Test Data**
- Use fixtures for reusable test data
- Mock external dependencies consistently
- Use temporary files for file-based tests

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Tests by Layer
```bash
# API layer tests
pytest tests/api/

# CLI layer tests
pytest tests/cli/

# Core business logic tests
pytest tests/core/

# Utility function tests
pytest tests/utils/
```

### Run Specific Test Files
```bash
pytest tests/api/test_api.py
pytest tests/cli/test_cli.py
pytest tests/core/test_masking.py
```

### Run with Coverage
```bash
pytest tests/ --cov=elm --cov-report=html
```

## Test Configuration

Test configuration is managed in:
- `conftest.py` - Shared fixtures and configuration
- `pytest.ini` or `pyproject.toml` - Pytest settings

## Best Practices

1. **Write tests first** when adding new functionality
2. **Mock external dependencies** to ensure test isolation
3. **Test both success and failure scenarios**
4. **Use descriptive test names** that explain what is being tested
5. **Keep tests focused** on a single piece of functionality
6. **Maintain test independence** - tests should not depend on each other
7. **Update tests** when refactoring code to maintain coverage

## Continuous Integration

Tests are automatically run in CI/CD pipelines to ensure:
- All tests pass before merging
- Code coverage meets minimum thresholds
- No regressions are introduced
- Cross-platform compatibility

## Contributing

When adding new functionality:
1. Write tests for the appropriate layer(s)
2. Follow the established testing patterns
3. Ensure proper mocking and isolation
4. Update this README if adding new test categories

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
