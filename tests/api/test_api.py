"""
Tests for the API module - Refactored for proper layer separation.

These tests focus on testing the API layer functions directly,
ensuring they properly delegate to core modules and return
expected data structures. All external dependencies are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from elm import api
from elm.core.types import OperationResult


class TestEnvironmentAPI:
    """Test environment-related API functions."""

    @patch('elm.core.environment.create_environment')
    def test_create_environment(self, mock_core_create):
        """Test API create_environment delegates to core module."""
        # Mock successful creation
        mock_core_create.return_value = OperationResult(
            success=True,
            message="Environment created successfully"
        )

        result = api.create_environment(
            name="test-env",
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            service="postgres",
            db_type="postgres"
        )

        assert result is True
        mock_core_create.assert_called_once_with(
            name="test-env",
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            service="postgres",
            db_type="postgres",
            encrypt=False,
            encryption_key=None,
            overwrite=False,
            connection_type=None
        )

    @patch('elm.core.environment.list_environments')
    def test_list_environments(self, mock_core_list):
        """Test API list_environments delegates to core module."""
        mock_environments = [
            {"name": "env1", "type": "postgres", "host": "localhost"},
            {"name": "env2", "type": "mysql", "host": "localhost"}
        ]
        mock_core_list.return_value = OperationResult(
            success=True,
            message="Found 2 environments",
            data=mock_environments,
            record_count=2
        )

        result = api.list_environments()

        assert len(result) == 2
        assert result[0]["name"] == "env1"
        mock_core_list.assert_called_once_with(show_all=False)

    @patch('elm.core.environment.get_environment')
    def test_get_environment(self, mock_core_get):
        """Test API get_environment delegates to core module."""
        mock_env_data = {
            "name": "test-env",
            "type": "postgres",
            "host": "localhost"
        }
        mock_core_get.return_value = OperationResult(
            success=True,
            message="Environment retrieved successfully",
            data=mock_env_data
        )

        result = api.get_environment("test-env")

        assert result is not None
        assert result["name"] == "test-env"
        mock_core_get.assert_called_once_with(name="test-env", encryption_key=None)

    @patch('elm.core.environment.delete_environment')
    def test_delete_environment(self, mock_core_delete):
        """Test API delete_environment delegates to core module."""
        mock_core_delete.return_value = OperationResult(
            success=True,
            message="Environment deleted successfully"
        )

        result = api.delete_environment("test-env")

        assert result is True
        mock_core_delete.assert_called_once_with(name="test-env")

    @patch('elm.core.environment.update_environment')
    def test_update_environment(self, mock_core_update):
        """Test API update_environment delegates to core module."""
        mock_core_update.return_value = OperationResult(
            success=True,
            message="Environment updated successfully"
        )

        result = api.update_environment(
            name="test-env",
            host="new-host",
            port=5433
        )

        assert result is True
        mock_core_update.assert_called_once_with(
            name="test-env",
            host="new-host",
            port=5433,
            user=None,
            password=None,
            service=None,
            db_type=None,
            encrypt=None,
            encryption_key=None
        )

    @patch('elm.core.environment.test_environment')
    def test_test_environment(self, mock_core_test):
        """Test API test_environment delegates to core module."""
        mock_core_test.return_value = OperationResult(
            success=True,
            message="Successfully connected to test-env"
        )

        result = api.test_environment("test-env")

        assert result["success"] is True
        assert "Successfully connected" in result["message"]
        mock_core_test.assert_called_once_with(name="test-env", encryption_key=None)

    @patch('elm.core.environment.execute_sql')
    def test_execute_sql(self, mock_core_execute):
        """Test API execute_sql delegates to core module."""
        mock_data = [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]
        mock_core_execute.return_value = OperationResult(
            success=True,
            message="Query executed successfully",
            data=mock_data,
            record_count=2
        )

        result = api.execute_sql("test-env", "SELECT * FROM test_table")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_core_execute.assert_called_once_with(
            environment="test-env",
            query="SELECT * FROM test_table",
            encryption_key=None,
            params=None
        )


class TestCopyAPI:
    """Test copy-related API functions."""

    @patch('elm.core.copy.copy_db_to_file')
    def test_copy_db_to_file(self, mock_core_copy):
        """Test API copy_db_to_file delegates to core module."""
        mock_core_copy.return_value = OperationResult(
            success=True,
            message='Successfully copied 100 records to test_output.csv',
            record_count=100
        )

        result = api.copy_db_to_file(
            source_env="test-env",
            query="SELECT * FROM test_table",
            file_path="test_output.csv",
            file_format="csv"
        )

        assert result["success"] is True
        assert result["record_count"] == 100
        mock_core_copy.assert_called_once_with(
            source_env="test-env",
            query="SELECT * FROM test_table",
            file_path="test_output.csv",
            file_format="csv",
            mode="REPLACE",
            batch_size=None,
            parallel_workers=1,
            source_encryption_key=None,
            apply_masks=True,
            verbose_batch_logs=True
        )

    @patch('elm.core.copy.copy_file_to_db')
    def test_copy_file_to_db(self, mock_core_copy):
        """Test API copy_file_to_db delegates to core module."""
        mock_core_copy.return_value = OperationResult(
            success=True,
            message='Successfully copied 50 records to test_table',
            record_count=50
        )

        result = api.copy_file_to_db(
            file_path="test_input.csv",
            target_env="test-env",
            table="test_table",
            file_format="csv"
        )

        assert result["success"] is True
        assert result["record_count"] == 50

    @patch('elm.core.copy.copy_db_to_db')
    def test_copy_db_to_db(self, mock_core_copy):
        """Test API copy_db_to_db delegates to core module."""
        mock_core_copy.return_value = OperationResult(
            success=True,
            message='Successfully copied 75 records to target_table',
            record_count=75
        )

        result = api.copy_db_to_db(
            source_env="source-env",
            target_env="target-env",
            query="SELECT * FROM source_table",
            table="target_table"
        )

        assert result["success"] is True
        assert result["record_count"] == 75


class TestMaskingAPI:
    """Test masking-related API functions."""

    @patch('elm.core.masking.add_mask')
    def test_add_mask(self, mock_core_add):
        """Test API add_mask delegates to core module."""
        mock_core_add.return_value = OperationResult(
            success=True,
            message="Masking rule added successfully"
        )

        result = api.add_mask(
            column="password",
            algorithm="star"
        )

        assert result is True
        mock_core_add.assert_called_once_with(
            column="password",
            algorithm="star",
            environment=None,
            length=None,
            params=None
        )

    @patch('elm.core.masking.remove_mask')
    def test_remove_mask(self, mock_core_remove):
        """Test API remove_mask delegates to core module."""
        mock_core_remove.return_value = OperationResult(
            success=True,
            message="Masking rule removed successfully"
        )

        result = api.remove_mask(column="password")

        assert result is True
        mock_core_remove.assert_called_once_with(
            column="password",
            environment=None
        )

    @patch('elm.core.masking.list_masks')
    def test_list_masks(self, mock_core_list):
        """Test API list_masks delegates to core module."""
        mock_masks = {
            'global': {
                'password': {'algorithm': 'star', 'params': {}}
            }
        }
        mock_core_list.return_value = OperationResult(
            success=True,
            message="Masking rules retrieved successfully",
            data=mock_masks
        )

        result = api.list_masks()

        assert 'global' in result
        assert 'password' in result['global']
        mock_core_list.assert_called_once_with(environment=None)

    @patch('elm.core.masking.test_mask')
    def test_test_mask(self, mock_core_test):
        """Test API test_mask delegates to core module."""
        mock_core_test.return_value = OperationResult(
            success=True,
            message="Mask test completed successfully",
            data={
                'original': 'secret123',
                'masked': '*****',
                'scope': 'global'
            }
        )

        result = api.test_mask(column="password", value="secret123")

        assert result['original'] == 'secret123'
        assert result['masked'] == '*****'
        mock_core_test.assert_called_once_with(
            column="password",
            value="secret123",
            environment=None
        )


class TestGenerationAPI:
    """Test generation-related API functions."""

    @patch('elm.core.generation.generate_data')
    def test_generate_data(self, mock_core_generate):
        """Test API generate_data delegates to core module."""
        mock_data = [
            {'id': 1, 'name': 'Test 1'},
            {'id': 2, 'name': 'Test 2'}
        ]
        mock_core_generate.return_value = OperationResult(
            success=True,
            message="Data generated successfully",
            data=mock_data,
            record_count=2
        )

        result = api.generate_data(
            num_records=2,
            columns=["id", "name"]
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        mock_core_generate.assert_called_once_with(
            num_records=2,
            columns=["id", "name"],
            environment=None,
            table=None,
            string_length=10,
            pattern=None,
            min_number=0,
            max_number=100,
            decimal_places=2,
            start_date=None,
            end_date=None,
            date_format="%Y-%m-%d"
        )

    @patch('elm.core.generation.generate_and_save')
    def test_generate_and_save(self, mock_core_generate_save):
        """Test API generate_and_save delegates to core module."""
        mock_core_generate_save.return_value = OperationResult(
            success=True,
            message='Successfully wrote 3 records to test_output.csv',
            record_count=3
        )

        result = api.generate_and_save(
            num_records=3,
            columns=["id", "name"],
            output="test_output.csv",
            format="csv"
        )

        assert result['success'] is True
        assert result['record_count'] == 3
        mock_core_generate_save.assert_called_once()
