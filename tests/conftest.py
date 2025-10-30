from elm.elm_utils import variables
import pytest
import pandas as pd
import datetime
from unittest.mock import MagicMock


def get_worker_id(request):
    """Get the worker ID for parallel test execution.

    Returns:
        str: Worker ID (e.g., 'gw0', 'gw1') or 'master' for non-parallel execution
    """
    if hasattr(request.config, 'workerinput'):
        return request.config.workerinput['workerid']
    return 'master'


@pytest.fixture
def worker_id(request):
    """Fixture that provides the worker ID for the current test."""
    return get_worker_id(request)


@pytest.fixture
def temp_env_dir():
    return variables.ENVS_FILE


@pytest.fixture
def unique_env_name(request):
    """Generate a unique environment name for parallel test execution.

    This fixture ensures that each test worker gets a unique environment name
    to avoid conflicts when running tests in parallel.
    """
    worker = get_worker_id(request)
    test_name = request.node.name
    # Create a unique name combining worker ID and test name
    return f"test_env_{worker}_{test_name}"[:50]  # Limit length


@pytest.fixture
def unique_table_name(request):
    """Generate a unique table name for parallel test execution.

    This fixture ensures that each test worker gets a unique table name
    to avoid conflicts when running tests in parallel.
    """
    worker = get_worker_id(request)
    test_name = request.node.name
    # Create a unique name combining worker ID and test name
    return f"test_table_{worker}_{test_name}"[:50]  # Limit length


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Ethan'],
        'email': [
            'alice@example.com',
            'bob@example.com',
            'charlie@example.com',
            'diana@example.com',
            'ethan@example.com'
        ],
        'password': ['secret123', 'password456', 'mypassword', '12345678', 'hunter2'],
        'balance': [123.45, 678.90, 234.56, 890.12, 345.67],
        'signup_date': [
            datetime(2022, 1, 10),
            datetime(2022, 3, 15),
            datetime(2022, 5, 20),
            datetime(2022, 7, 25),
            datetime(2022, 9, 30)
        ]
    })
