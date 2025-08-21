"""
Tests for the data copy functionality of the ELM tool.
"""
import pytest

import elm
import pandas as pd

TEST_SOURCE_ENV_NAME = "test-pg"
TEST_TARGET_ENV_NAME = "test-mysql"
TEST_PG_ENV_NAME = "test-pg"
TEST_MYSQL_ENV_NAME = "test-mysql"
TEST_MSSQL_ENV_NAME = "test-mssql"
TEST_ORACLE_ENV_NAME = "test-oracle"


def test_copy_db_to_file(temp_env_dir):
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



def test_copy_file_to_db(temp_env_dir):
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

def test_copy_file_to_db_with_validation(temp_env_dir):
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


        # Test when table exists

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


def test_copy_db_to_db(temp_env_dir):
    """Test copying data from database to database."""
    # Check if the source and target environments exist

    source_env = elm.get_environment(TEST_SOURCE_ENV_NAME)

    if(source_env is None):
        pytest.fail(f"Failed to get source environment {TEST_SOURCE_ENV_NAME}")
        return

    target_env = elm.get_environment(TEST_TARGET_ENV_NAME)
    
    if(target_env is None):
        pytest.fail(f"Failed to get target environment {TEST_TARGET_ENV_NAME}")
        return


        mock_execute_query.return_value 

        # Copy data from database to database
        result = elm.copy_db_to_db(
            source_env=source_env["name"],
            target_env=target_env["name"],
            query="SELECT * FROM categories",
            table="categories",
            mode="APPEND"
        )

        # Verify the result
        assert result["success"] is True
        assert "Successfully copied 5 records" in result["message"]
        assert result["record_count"] == 5

        # Verify the mocks were called correctly
        mock_execute_query.assert_called_once()
        mock_write_to_db.assert_called_once()