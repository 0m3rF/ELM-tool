"""
Tests for the data copy functionality of the ELM tool.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

import elm
import pandas as pd


def test_copy_db_to_file(temp_env_dir, mock_db_connection, sample_dataframe, mock_file_operations):
    """Test copying data from database to file."""
    # Create a test environment
    elm.create_environment(
        name="test-pg",
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    # Mock the execute_query function to return a sample DataFrame
    with patch('elm.api.execute_query') as mock_execute_query:
        mock_execute_query.return_value = sample_dataframe

        # Copy data from database to file
        result = elm.copy_db_to_file(
            source_env="test-pg",
            query="SELECT * FROM test_table",
            file_path="tests/test_output.csv",
            file_format="csv"
        )

        # Verify the result
        assert result["success"] is True
        assert "Successfully copied 5 records" in result["message"]
        assert result["record_count"] == 5

        # Verify the mocks were called correctly
        mock_execute_query.assert_called_once()
        mock_file_operations['to_csv'].assert_called_once()


def test_copy_db_to_file_with_masking(temp_env_dir, mock_db_connection, sample_dataframe,
                                     mock_file_operations, mock_masking_file):
    """Test copying data from database to file with masking applied."""
    # Create a test environment
    elm.create_environment(
        name="test-pg",
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    # Set up masking definitions
    mock_masking_file['update']({
        'global': {
            'password': {
                'algorithm': 'star',
                'params': {}
            }
        },
        'environments': {}
    })

    # Mock the execute_query function to return a sample DataFrame
    with patch('elm.api.execute_query') as mock_execute_query, \
         patch('elm.api.apply_masking') as mock_apply_masking:

        mock_execute_query.return_value = sample_dataframe

        # Mock the apply_masking function to return a modified DataFrame
        masked_df = sample_dataframe.copy()
        masked_df['password'] = '*****'
        mock_apply_masking.return_value = masked_df

        # Copy data from database to file with masking
        result = elm.copy_db_to_file(
            source_env="test-pg",
            query="SELECT * FROM test_table",
            file_path="tests/test_output.csv",
            file_format="csv",
            apply_masks=True
        )

        # Verify the result
        assert result["success"] is True
        assert "Successfully copied 5 records" in result["message"]

        # Verify the mocks were called correctly
        mock_execute_query.assert_called_once()
        mock_apply_masking.assert_called_once()
        mock_file_operations['to_csv'].assert_called_once()


def test_copy_file_to_db(temp_env_dir, mock_db_connection, mock_file_operations):
    """Test copying data from file to database."""
    # Create a test environment
    elm.create_environment(
        name="test-pg",
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    # Mock the write_to_db function
    with patch('elm.api.write_to_db') as mock_write_to_db:
        # Copy data from file to database
        result = elm.copy_file_to_db(
            file_path="test_input.csv",
            target_env="test-pg",
            table="test_table",
            file_format="csv",
            mode="APPEND"
        )

        # Verify the result
        assert result["success"] is True
        assert "Successfully copied" in result["message"]

        # Verify the mocks were called correctly
        mock_file_operations['read_csv'].assert_called_once()
        mock_write_to_db.assert_called_once()


def test_copy_file_to_db_with_validation(temp_env_dir, mock_db_connection, mock_file_operations):
    """Test copying data from file to database with table validation."""
    # Create a test environment
    elm.create_environment(
        name="test-pg",
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    # Mock the check_table_exists and write_to_db functions
    with patch('elm.api.check_table_exists') as mock_check_table, \
         patch('elm.api.write_to_db') as mock_write_to_db:

        # Test when table exists
        mock_check_table.return_value = True

        result = elm.copy_file_to_db(
            file_path="test_input.csv",
            target_env="test-pg",
            table="test_table",
            file_format="csv",
            mode="APPEND",
            validate_target=True
        )

        # Verify the result
        assert result["success"] is True

        # Test when table doesn't exist
        mock_check_table.return_value = False

        result = elm.copy_file_to_db(
            file_path="test_input.csv",
            target_env="test-pg",
            table="nonexistent_table",
            file_format="csv",
            mode="APPEND",
            validate_target=True
        )

        # Verify the result
        assert result["success"] is False
        assert "does not exist" in result["message"]


def test_copy_db_to_db(temp_env_dir, mock_db_connection, sample_dataframe):
    """Test copying data from database to database."""
    # Create source and target environments
    elm.create_environment(
        name="source-pg",
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    elm.create_environment(
        name="target-pg",
        host="localhost",
        port=5433,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    # Mock the execute_query and write_to_db functions
    with patch('elm.api.execute_query') as mock_execute_query, \
         patch('elm.api.write_to_db') as mock_write_to_db:

        mock_execute_query.return_value = sample_dataframe

        # Copy data from database to database
        result = elm.copy_db_to_db(
            source_env="source-pg",
            target_env="target-pg",
            query="SELECT * FROM source_table",
            table="target_table",
            mode="APPEND"
        )

        # Verify the result
        assert result["success"] is True
        assert "Successfully copied 5 records" in result["message"]
        assert result["record_count"] == 5

        # Verify the mocks were called correctly
        mock_execute_query.assert_called_once()
        mock_write_to_db.assert_called_once()


def test_copy_db_to_db_with_masking(temp_env_dir, mock_db_connection, sample_dataframe, mock_masking_file):
    """Test copying data from database to database with masking applied."""
    # Create source and target environments
    elm.create_environment(
        name="source-pg",
        host="localhost",
        port=5432,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    elm.create_environment(
        name="target-pg",
        host="localhost",
        port=5433,
        user="postgres",
        password="password",
        service="postgres",
        db_type="postgres"
    )

    # Set up masking definitions
    mock_masking_file['update']({
        'global': {
            'password': {
                'algorithm': 'star',
                'params': {}
            }
        },
        'environments': {
            'source-pg': {
                'email': {
                    'algorithm': 'star_length',
                    'params': {'length': 4}
                }
            }
        }
    })

    # Mock the execute_query, apply_masking, and write_to_db functions
    with patch('elm.api.execute_query') as mock_execute_query, \
         patch('elm.api.apply_masking') as mock_apply_masking, \
         patch('elm.api.write_to_db') as mock_write_to_db:

        mock_execute_query.return_value = sample_dataframe

        # Mock the apply_masking function to return a modified DataFrame
        masked_df = sample_dataframe.copy()
        masked_df['password'] = '*****'
        masked_df['email'] = ['john****', 'jane****', 'bob*****', 'alic****', 'char****']
        mock_apply_masking.return_value = masked_df

        # Copy data from database to database with masking
        result = elm.copy_db_to_db(
            source_env="source-pg",
            target_env="target-pg",
            query="SELECT * FROM source_table",
            table="target_table",
            mode="APPEND",
            apply_masks=True
        )

        # Verify the result
        assert result["success"] is True
        assert "Successfully copied 5 records" in result["message"]

        # Verify the mocks were called correctly
        mock_execute_query.assert_called_once()
        mock_apply_masking.assert_called_once()
        mock_write_to_db.assert_called_once()
