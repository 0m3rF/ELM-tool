from elm.elm_utils import variables
import pytest
import pandas as pd
from unittest.mock import MagicMock


@pytest.fixture
def temp_env_dir():
    return variables.ENVS_FILE


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
        'password': ['secret123', 'password456', 'mypassword']
    })


@pytest.fixture
def mock_masking_file():
    """Create a mock masking file for testing."""
    # Create a simple dictionary-like object that can be used as a mock
    class MockMaskingFile(dict):
        def __init__(self):
            super().__init__()
            self.definitions = {
                'global': {},
                'environments': {}
            }
            self.save = MagicMock(return_value=True)

        def update(self, new_definitions):
            self.definitions.update(new_definitions)

        def __getitem__(self, key):
            if key == 'definitions':
                return self.definitions
            elif key == 'update':
                return self.update
            elif key == 'save':
                return self.save
            else:
                return super().__getitem__(key)

    return MockMaskingFile()