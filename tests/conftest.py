"""
Pytest configuration file for ELM tool tests.
Contains shared fixtures and setup/teardown functions.
"""
import os
import shutil
import tempfile
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine

import elm
from elm.elm_utils import variables


@pytest.fixture(scope="function")
def temp_env_dir():
    """Create a temporary directory for environment files during tests."""
    original_envs_file = variables.ENVS_FILE
    temp_dir = tempfile.mkdtemp()
    temp_envs_file = os.path.join(temp_dir, "environments.ini")
    
    # Patch the ENVS_FILE variable
    with patch.object(variables, 'ENVS_FILE', temp_envs_file):
        yield temp_envs_file
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def mock_db_connection():
    """Mock SQLAlchemy database connection."""
    with patch('sqlalchemy.create_engine') as mock_create_engine:
        # Create a mock engine and connection
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_result = MagicMock()
        
        # Configure the mocks
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value = mock_result
        mock_result.fetchall.return_value = [(1,)]
        
        yield mock_create_engine


@pytest.fixture(scope="function")
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 
                 'alice@example.com', 'charlie@example.com'],
        'password': ['password123', 'secret456', 'secure789', 'mypass321', 'letmein555'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    })


@pytest.fixture(scope="function")
def mock_file_operations():
    """Mock file read/write operations."""
    with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
         patch('pandas.DataFrame.to_json') as mock_to_json, \
         patch('pandas.read_csv') as mock_read_csv, \
         patch('pandas.read_json') as mock_read_json:
        
        # Configure the mocks
        mock_read_csv.return_value = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Test User 1', 'Test User 2', 'Test User 3'],
            'email': ['test1@example.com', 'test2@example.com', 'test3@example.com']
        })
        
        mock_read_json.return_value = mock_read_csv.return_value
        
        yield {
            'to_csv': mock_to_csv,
            'to_json': mock_to_json,
            'read_csv': mock_read_csv,
            'read_json': mock_read_json
        }


@pytest.fixture(scope="function")
def mock_masking_file():
    """Mock masking definitions file operations."""
    with patch('elm.elm_commands.mask.load_masking_definitions') as mock_load, \
         patch('elm.elm_commands.mask.save_masking_definitions') as mock_save:
        
        # Default masking definitions
        mock_definitions = {
            'global': {},
            'environments': {}
        }
        
        mock_load.return_value = mock_definitions
        
        def update_definitions(new_defs):
            nonlocal mock_definitions
            mock_definitions = new_defs
            mock_load.return_value = mock_definitions
        
        yield {
            'load': mock_load,
            'save': mock_save,
            'update': update_definitions,
            'definitions': mock_definitions
        }
