"""
Tests for config CLI commands.

This module tests the configuration management CLI commands.
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from elm.elm_commands.config import config, show, set, reset, paths
from elm.core.types import OperationResult


class TestConfigCLICommands:
    """Test config CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_config_help(self):
        """Test config command help."""
        result = self.runner.invoke(config, ['--help'])
        assert result.exit_code == 0
        assert 'Configuration management commands' in result.output
        assert 'show' in result.output
        assert 'set' in result.output
        assert 'reset' in result.output
        assert 'paths' in result.output

    def test_show_command_success(self):
        """Test show command with successful result."""
        mock_result = OperationResult(
            success=True,
            message="Config retrieved successfully",
            data={
                'config': {
                    'ELM_TOOL_HOME': '/test/home',
                    'VENV_NAME': 'test_venv',
                    'APP_NAME': 'ELMtool'
                },
                'paths': {
                    'Config File': '/test/home/config.json',
                    'ELM Tool Home': '/test/home',
                    'Environments File': '/test/home/environments.ini'
                }
            }
        )
        
        with patch('elm.core.config.show_config_info', return_value=mock_result):
            result = self.runner.invoke(show)
            
            assert result.exit_code == 0
            assert 'ELM Tool Configuration:' in result.output
            assert 'Configuration Values:' in result.output
            assert 'File Paths:' in result.output
            assert 'ELM_TOOL_HOME: /test/home' in result.output
            assert 'VENV_NAME: test_venv' in result.output

    def test_show_command_failure(self):
        """Test show command with failure result."""
        mock_result = OperationResult(
            success=False,
            message="Failed to retrieve config"
        )

        with patch('elm.core.config.show_config_info', return_value=mock_result):
            result = self.runner.invoke(show)

            assert result.exit_code != 0
            assert 'Failed to retrieve config' in result.output

    def test_set_command_success(self):
        """Test set command with successful result."""
        mock_result = OperationResult(
            success=True,
            message="Configuration updated successfully"
        )
        
        with patch('elm.core.config.set_config', return_value=mock_result):
            result = self.runner.invoke(set, ['ELM_TOOL_HOME', '/new/path'])
            
            assert result.exit_code == 0
            assert 'Configuration updated successfully' in result.output
            assert 'You may need to restart the tool' in result.output

    def test_set_command_failure(self):
        """Test set command with failure result."""
        mock_result = OperationResult(
            success=False,
            message="Failed to update configuration"
        )

        with patch('elm.core.config.set_config', return_value=mock_result):
            result = self.runner.invoke(set, ['ELM_TOOL_HOME', '/new/path'])

            assert result.exit_code != 0
            assert 'Failed to update configuration' in result.output

    def test_set_command_invalid_key_continue(self):
        """Test set command with invalid key but user continues."""
        mock_result = OperationResult(
            success=True,
            message="Configuration updated successfully"
        )
        
        with patch('elm.core.config.set_config', return_value=mock_result):
            with patch('click.confirm', return_value=True):
                result = self.runner.invoke(set, ['INVALID_KEY', 'test_value'])
                
                assert result.exit_code == 0
                assert 'Warning:' in result.output
                assert 'INVALID_KEY' in result.output
                assert 'is not a standard configuration key' in result.output

    def test_set_command_invalid_key_abort(self):
        """Test set command with invalid key and user aborts."""
        with patch('click.confirm', return_value=False):
            result = self.runner.invoke(set, ['INVALID_KEY', 'test_value'])
            
            assert result.exit_code == 0
            assert 'Warning:' in result.output

    def test_reset_command_success(self):
        """Test reset command with successful result."""
        mock_result = OperationResult(
            success=True,
            message="Configuration reset successfully"
        )

        with patch('elm.core.config.reset_config', return_value=mock_result):
            result = self.runner.invoke(reset, input='y\n')

            assert result.exit_code == 0
            assert 'Configuration reset successfully' in result.output

    def test_reset_command_failure(self):
        """Test reset command with failure result."""
        mock_result = OperationResult(
            success=False,
            message="Failed to reset configuration"
        )

        with patch('elm.core.config.reset_config', return_value=mock_result):
            result = self.runner.invoke(reset, input='y\n')

            assert result.exit_code != 0
            assert 'Failed to reset configuration' in result.output

    def test_reset_command_abort(self):
        """Test reset command when user aborts."""
        result = self.runner.invoke(reset, input='n\n')

        assert result.exit_code != 0
        # Should exit without doing anything

    def test_paths_command_success(self):
        """Test paths command with successful result."""
        mock_result = OperationResult(
            success=True,
            message="Paths retrieved successfully",
            data={
                'paths': {
                    'Config File': '/test/home/config.json',
                    'ELM Tool Home': '/test/home',
                    'Environments File': '/test/home/environments.ini',
                    'Masking File': '/test/home/masking.json'
                }
            }
        )
        
        with patch('elm.core.config.show_config_info', return_value=mock_result):
            with patch('os.path.exists') as mock_exists:
                # Mock some paths as existing, some as not
                mock_exists.side_effect = lambda path: path.endswith('config.json')
                
                result = self.runner.invoke(paths)
                
                assert result.exit_code == 0
                assert 'ELM Tool File Paths:' in result.output
                assert '✓' in result.output  # For existing files
                assert '✗' in result.output  # For non-existing files
                assert '✓ = exists, ✗ = does not exist' in result.output

    def test_paths_command_failure(self):
        """Test paths command with failure result."""
        mock_result = OperationResult(
            success=False,
            message="Failed to retrieve paths"
        )

        with patch('elm.core.config.show_config_info', return_value=mock_result):
            result = self.runner.invoke(paths)

            assert result.exit_code != 0
            assert 'Failed to retrieve paths' in result.output

    def test_config_aliases(self):
        """Test config command aliases."""
        # Test info alias for show
        result = self.runner.invoke(config, ['info', '--help'])
        assert result.exit_code == 0
        assert 'Show current configuration' in result.output
        
        # Test update alias for set
        result = self.runner.invoke(config, ['update', '--help'])
        assert result.exit_code == 0
        assert 'Set a configuration value' in result.output
        
        # Test dirs alias for paths
        result = self.runner.invoke(config, ['dirs', '--help'])
        assert result.exit_code == 0
        assert 'Show important file and directory paths' in result.output
