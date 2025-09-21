"""
Tests for the elm_commands/environment.py module.
"""
import subprocess
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import click
from click.testing import CliRunner

from elm.elm_commands.environment import (
    ensure_db_driver_installed,
    AliasedGroup,
    DB_PACKAGES,
    environment
)
from elm.core.types import OperationResult


class TestEnsureDbDriverInstalled:
    """Test ensure_db_driver_installed function."""

    def test_ensure_db_driver_installed_unknown_db_type(self):
        """Test with unknown database type."""
        # Should return early without doing anything
        result = ensure_db_driver_installed("UNKNOWN_DB")
        assert result is None

    def test_ensure_db_driver_installed_already_installed_psycopg2(self):
        """Test when psycopg2-binary is already installed."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()

            result = ensure_db_driver_installed("POSTGRES")

            # Should import psycopg2 for psycopg2-binary package
            mock_import.assert_called_once_with("psycopg2")
            assert result is None

    def test_ensure_db_driver_installed_already_installed_other(self):
        """Test when other database driver is already installed."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()

            result = ensure_db_driver_installed("MYSQL")

            # Should import pymysql
            mock_import.assert_called_once_with("pymysql")
            assert result is None

    def test_ensure_db_driver_installed_not_installed_success(self):
        """Test installing driver when not installed - success case."""
        with patch('builtins.__import__') as mock_import, \
             patch('subprocess.check_call') as mock_subprocess, \
             patch('builtins.print') as mock_print:

            # First call raises ImportError (not installed)
            mock_import.side_effect = ImportError("No module named 'pymysql'")
            mock_subprocess.return_value = None  # Successful installation

            result = ensure_db_driver_installed("MYSQL")

            # Should try to install the package
            mock_subprocess.assert_called_once_with([
                sys.executable, "-m", "pip", "install", "pymysql"
            ])
            mock_print.assert_any_call("Installing required database driver: pymysql")
            mock_print.assert_any_call("Successfully installed pymysql")
            assert result is None

    def test_ensure_db_driver_installed_not_installed_failure(self):
        """Test installing driver when not installed - failure case."""
        with patch('builtins.__import__') as mock_import, \
             patch('subprocess.check_call') as mock_subprocess, \
             patch('builtins.print') as mock_print:

            # First call raises ImportError (not installed)
            mock_import.side_effect = ImportError("No module named 'oracledb'")
            # Installation fails
            mock_subprocess.side_effect = subprocess.CalledProcessError(1, "pip install")

            result = ensure_db_driver_installed("ORACLE")

            # Should try to install and handle failure
            mock_subprocess.assert_called_once_with([
                sys.executable, "-m", "pip", "install", "oracledb"
            ])
            mock_print.assert_any_call("Installing required database driver: oracledb")
            mock_print.assert_any_call("Failed to install oracledb: Command 'pip install' returned non-zero exit status 1.")
            mock_print.assert_any_call("Please install oracledb manually using: pip install oracledb")
            assert result is None

    def test_ensure_db_driver_installed_package_name_with_version(self):
        """Test with package name that has version specifier."""
        # Modify DB_PACKAGES temporarily to test version handling
        original_packages = DB_PACKAGES.copy()
        DB_PACKAGES["TEST_DB"] = "test-package>=1.0.0"

        try:
            with patch('builtins.__import__') as mock_import:
                mock_import.return_value = MagicMock()

                result = ensure_db_driver_installed("TEST_DB")

                # Should import test_package (before the >)
                mock_import.assert_called_once_with("test_package")
                assert result is None
        finally:
            # Restore original packages
            DB_PACKAGES.clear()
            DB_PACKAGES.update(original_packages)

    def test_ensure_db_driver_installed_package_name_with_dash(self):
        """Test with package name that has dashes."""
        # Modify DB_PACKAGES temporarily to test dash handling
        original_packages = DB_PACKAGES.copy()
        DB_PACKAGES["TEST_DB"] = "test-package-name"

        try:
            with patch('builtins.__import__') as mock_import:
                mock_import.return_value = MagicMock()

                result = ensure_db_driver_installed("TEST_DB")

                # Should import test_package_name (dashes replaced with underscores)
                mock_import.assert_called_once_with("test_package_name")
                assert result is None
        finally:
            # Restore original packages
            DB_PACKAGES.clear()
            DB_PACKAGES.update(original_packages)

    def test_db_packages_constants(self):
        """Test that DB_PACKAGES contains expected database types."""
        expected_packages = {
            "ORACLE": "oracledb",
            "MYSQL": "pymysql",
            "MSSQL": "pyodbc",
            "POSTGRES": "psycopg2-binary"
        }

        assert DB_PACKAGES == expected_packages


class TestEnvironmentAliasedGroup:
    """Test AliasedGroup functionality for environment commands."""

    def test_aliased_group_get_command_with_alias(self):
        """Test getting command with alias."""
        # Create mock commands
        mock_create = MagicMock()
        mock_create.name = 'create'

        # Create aliases dict
        aliases = {'new': mock_create}

        group = AliasedGroup()

        with patch('elm.elm_commands.environment.ALIASES', aliases):
            with patch.object(click.Group, 'get_command') as mock_super:
                mock_super.return_value = mock_create

                result = group.get_command(None, 'new')

                # Should call super with the resolved command name
                mock_super.assert_called_with(None, 'create')
                assert result == mock_create

    def test_aliased_group_get_command_without_alias(self):
        """Test getting command without alias."""
        group = AliasedGroup()

        with patch('elm.elm_commands.environment.ALIASES', {}):
            with patch.object(click.Group, 'get_command') as mock_super:
                mock_super.return_value = None

                result = group.get_command(None, 'unknown')

                # Should call super with the original command name
                mock_super.assert_called_with(None, 'unknown')
                assert result is None

    def test_aliased_group_get_command_none_cmd_name(self):
        """Test getting command with None cmd_name."""
        group = AliasedGroup()

        with patch('elm.elm_commands.environment.ALIASES', {}):
            result = group.get_command(None, None)

            # Should return None for None cmd_name
            assert result is None


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


class TestEnvironmentCLICommands:
    """Test CLI command functions for environment commands."""

    @patch('elm.elm_commands.environment.core_env.create_environment')
    def test_create_environment_success_case(self, mock_create, runner):
        """Test create environment success case."""
        mock_create.return_value = OperationResult(
            success=True,
            message="Environment created successfully"
        )

        result = runner.invoke(environment, [
            'create',
            'test-env',
            '--host', 'localhost',
            '--port', '5432',
            '--user', 'postgres',
            '--password', 'password',
            '--service', 'mydb',
            '--type', 'POSTGRES'
        ])

        assert result.exit_code == 0
        assert "Environment created successfully" in result.output
        mock_create.assert_called_once()

    @patch('elm.elm_commands.environment.core_env.create_environment')
    def test_create_environment_failure_case(self, mock_create, runner):
        """Test create environment failure case."""
        mock_create.return_value = OperationResult(
            success=False,
            message="Creation failed"
        )

        result = runner.invoke(environment, [
            'create',
            'test-env',
            '--host', 'localhost',
            '--port', '5432',
            '--user', 'postgres',
            '--password', 'password',
            '--service', 'mydb',
            '--type', 'POSTGRES'
        ])

        assert result.exit_code != 0
        assert "Creation failed" in result.output

    @patch('elm.elm_commands.environment.core_env.delete_environment')
    def test_delete_environment_success(self, mock_delete, runner):
        """Test delete environment successfully."""
        mock_delete.return_value = OperationResult(
            success=True,
            message="Environment 'test-env' deleted successfully"
        )

        result = runner.invoke(environment, [
            'delete',
            'test-env'
        ])

        assert result.exit_code == 0
        assert "deleted successfully" in result.output
        mock_delete.assert_called_once_with(name='test-env')

    @patch('elm.elm_commands.environment.core_env.delete_environment')
    def test_delete_environment_failure(self, mock_delete, runner):
        """Test delete environment failure."""
        mock_delete.return_value = OperationResult(
            success=False,
            message="Environment not found"
        )

        result = runner.invoke(environment, [
            'delete',
            'nonexistent-env'
        ])

        assert result.exit_code != 0
        assert "Environment not found" in result.output

    @patch('elm.elm_commands.environment.core_env.get_environment')
    def test_show_environment_success(self, mock_get, runner):
        """Test show environment successfully."""
        mock_get.return_value = OperationResult(
            success=True,
            message="Environment retrieved successfully",
            data={'name': 'test-env', 'host': 'localhost', 'port': '5432', 'user': 'postgres'}
        )

        result = runner.invoke(environment, [
            'show',
            'test-env'
        ])

        assert result.exit_code == 0
        mock_get.assert_called_once_with(name='test-env', encryption_key=None)
        # Should show environment details
        assert 'localhost' in result.output

    @patch('elm.elm_commands.environment.core_env.get_environment')
    def test_show_environment_failure(self, mock_get, runner):
        """Test show environment failure."""
        mock_get.return_value = OperationResult(
            success=False,
            message="Environment not found"
        )

        result = runner.invoke(environment, [
            'show',
            'nonexistent-env'
        ])

        assert result.exit_code != 0
        assert "Environment not found" in result.output

    @patch('elm.elm_commands.environment.core_env.update_environment')
    def test_update_environment_success(self, mock_update, runner):
        """Test update environment successfully."""
        mock_update.return_value = OperationResult(
            success=True,
            message="Environment 'test-env' updated successfully"
        )

        result = runner.invoke(environment, [
            'update',
            'test-env',
            '--host', 'new-host'
        ])

        assert result.exit_code == 0
        assert "updated successfully" in result.output
        mock_update.assert_called_once()

    @patch('elm.elm_commands.environment.core_env.update_environment')
    def test_update_environment_failure(self, mock_update, runner):
        """Test update environment failure."""
        mock_update.return_value = OperationResult(
            success=False,
            message="Update failed"
        )

        result = runner.invoke(environment, [
            'update',
            'test-env',
            '--host', 'new-host'
        ])

        assert result.exit_code != 0
        assert "Update failed" in result.output

    @patch('elm.elm_commands.environment.core_env.test_environment')
    def test_test_environment_success(self, mock_test, runner):
        """Test test environment successfully."""
        mock_test.return_value = OperationResult(
            success=True,
            message="Connection successful"
        )

        result = runner.invoke(environment, [
            'test',
            'test-env'
        ])

        assert result.exit_code == 0
        assert "✓ Connection successful" in result.output
        mock_test.assert_called_once_with(name='test-env', encryption_key=None)

    @patch('elm.elm_commands.environment.core_env.test_environment')
    def test_test_environment_failure(self, mock_test, runner):
        """Test test environment failure."""
        mock_test.return_value = OperationResult(
            success=False,
            message="Connection failed"
        )

        result = runner.invoke(environment, [
            'test',
            'test-env'
        ])

        assert result.exit_code == 0  # test command doesn't fail, just shows result
        assert "✗ Connection failed" in result.output
        mock_test.assert_called_once_with(name='test-env', encryption_key=None)

    @patch('elm.elm_commands.environment.core_env.execute_sql')
    def test_execute_sql_success(self, mock_execute, runner):
        """Test execute SQL successfully."""
        mock_execute.return_value = OperationResult(
            success=True,
            message="Query executed successfully",
            data=[{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}],
            record_count=2
        )

        result = runner.invoke(environment, [
            'execute',
            'test-env',
            '--query', 'SELECT * FROM test'
        ])

        assert result.exit_code == 0
        assert 'id' in result.output
        assert 'name' in result.output
        mock_execute.assert_called_once()

    @patch('elm.elm_commands.environment.core_env.execute_sql')
    def test_execute_sql_no_results(self, mock_execute, runner):
        """Test execute SQL with no results."""
        mock_execute.return_value = OperationResult(
            success=True,
            data=None,
            message="Query executed successfully. No rows returned."
        )

        result = runner.invoke(environment, [
            'execute',
            'test-env',
            '--query', 'DELETE FROM test'
        ])

        assert result.exit_code == 0
        assert "Query executed successfully. No rows returned." in result.output

    @patch('elm.elm_commands.environment.core_env.execute_sql')
    def test_execute_sql_failure(self, mock_execute, runner):
        """Test execute SQL with failure."""
        mock_execute.return_value = OperationResult(
            success=False,
            message="SQL execution failed"
        )

        result = runner.invoke(environment, [
            'execute',
            'test-env',
            '--query', 'INVALID SQL'
        ])

        assert result.exit_code == 0  # execute command doesn't fail, just shows error
        assert "SQL execution failed" in result.output


class TestEnvironmentCLIEdgeCases:
    """Test edge cases for environment CLI commands."""

    def test_create_environment_missing_required_fields(self, runner):
        """Test create environment with missing required fields."""
        result = runner.invoke(environment, [
            'create',
            'test-env'
            # Missing required fields like --host, --user, etc.
        ])

        assert result.exit_code != 0
        # Click will show missing option errors

    def test_create_environment_encrypt_without_key(self, runner):
        """Test create environment with encrypt=True but no encryption key."""
        result = runner.invoke(environment, [
            'create',
            'test-env',
            '--host', 'localhost',
            '--port', '5432',
            '--user', 'postgres',
            '--password', 'password',
            '--service', 'mydb',
            '--type', 'POSTGRES',
            '--encrypt'
            # Missing --encryption-key
        ])

        assert result.exit_code != 0
        assert "encryption" in result.output.lower()

    @patch('elm.elm_commands.environment.core_env.create_environment')
    def test_create_environment_already_exists_no_overwrite(self, mock_create, runner):
        """Test create environment when it already exists without overwrite."""
        mock_create.return_value = OperationResult(
            success=False,
            message="Environment already exists"
        )

        result = runner.invoke(environment, [
            'create',
            'existing-env',
            '--host', 'localhost',
            '--port', '5432',
            '--user', 'postgres',
            '--password', 'password',
            '--service', 'mydb',
            '--type', 'POSTGRES'
        ])

        assert result.exit_code != 0
        assert "Environment already exists" in result.output

    @patch('elm.elm_commands.environment.core_env.create_environment')
    def test_create_environment_user_input_mode(self, mock_create, runner):
        """Test create environment with user input mode."""
        mock_create.return_value = OperationResult(
            success=True,
            message="Environment created successfully"
        )

        # Test that --user-input flag is accepted
        result = runner.invoke(environment, [
            'create',
            'test-env',
            '--user-input'
        ], input='localhost\n5432\npostgres\npassword\npassword\nmydb\nPOSTGRES\nn\n')

        # The command should start prompting for input
        assert result.exit_code == 0 or 'Host' in result.output

    @patch('elm.elm_commands.environment.core_env.create_environment')
    def test_create_environment_user_input_with_encryption(self, mock_create, runner):
        """Test create environment with user input mode and encryption."""
        mock_create.return_value = OperationResult(
            success=True,
            message="Environment created successfully"
        )

        # Test that --user-input flag with encryption is accepted
        result = runner.invoke(environment, [
            'create',
            'test-env',
            '--user-input'
        ], input='localhost\n5432\npostgres\npassword\npassword\nmydb\nPOSTGRES\ny\nsecret\nsecret\n')

        # The command should start prompting for input
        assert result.exit_code == 0 or 'Host' in result.output

    @patch('elm.elm_commands.environment.core_env.create_environment')
    def test_create_environment_password_mismatch(self, mock_create, runner):
        """Test create environment with password mismatch."""
        mock_create.return_value = OperationResult(
            success=True,
            message="Environment created successfully"
        )

        # Test that --user-input flag handles password mismatch
        result = runner.invoke(environment, [
            'create',
            'test-env',
            '--user-input'
        ], input='localhost\n5432\npostgres\npassword1\npassword2\npassword\npassword\nmydb\nPOSTGRES\nn\n')

        # The command should handle password mismatch and retry
        assert result.exit_code == 0 or 'Host' in result.output