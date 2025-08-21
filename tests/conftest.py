from elm.elm_utils import variables
import pytest
import pandas as pd
import datetime
from unittest.mock import MagicMock


@pytest.fixture
def temp_env_dir():
    return variables.ENVS_FILE

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
