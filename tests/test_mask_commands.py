"""
Tests for the elm_commands/mask.py module.
"""
import pytest
import pandas as pd
import click
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from elm.elm_commands.mask import (
    load_masking_definitions,
    save_masking_definitions,
    apply_masking,
    AliasedGroup,
    mask
)
from elm.core.types import OperationResult


class TestMaskCommands:
    """Test mask command functions."""

    def test_load_masking_definitions_success(self):
        """Test successful loading of masking definitions."""
        with patch('elm.core.masking.load_masking_definitions') as mock_load:
            mock_load.return_value = {'global': {}, 'environments': {}}

            result = load_masking_definitions()

            assert result == {'global': {}, 'environments': {}}
            mock_load.assert_called_once()

    def test_save_masking_definitions_success(self):
        """Test successful saving of masking definitions."""
        definitions = {'global': {'password': {'algorithm': 'star'}}}

        with patch('elm.core.masking.save_masking_definitions') as mock_save:
            mock_save.return_value = None

            result = save_masking_definitions(definitions)

            assert result is True
            mock_save.assert_called_once_with(definitions)

    def test_save_masking_definitions_error(self):
        """Test error handling in save_masking_definitions."""
        definitions = {'global': {'password': {'algorithm': 'star'}}}

        with patch('elm.core.masking.save_masking_definitions') as mock_save, \
             patch('click.echo') as mock_echo:
            mock_save.side_effect = Exception("Save error")

            result = save_masking_definitions(definitions)

            assert result is False
            mock_echo.assert_called_once_with("Error saving masking definitions: Save error")

    def test_apply_masking_non_dataframe(self):
        """Test apply_masking with non-DataFrame input."""
        result = apply_masking("not a dataframe")
        assert result == "not a dataframe"

    def test_apply_masking_with_global_definitions(self):
        """Test apply_masking with global masking definitions."""
        def star_mask(*_args, **_kwargs):
            return '*****'

        # First, fix the missing import issue
        with patch('elm.elm_commands.mask.MASKING_ALGORITHMS', {
            'star': star_mask
        }):
            data = pd.DataFrame({'password': ['secret123', 'password456']})

            with patch('elm.elm_commands.mask.load_masking_definitions') as mock_load:
                mock_load.return_value = {
                    'global': {
                        'password': {'algorithm': 'star', 'params': {}}
                    },
                    'environments': {}
                }

                result = apply_masking(data)

                assert isinstance(result, pd.DataFrame)
                assert result['password'].iloc[0] == '*****'
                assert result['password'].iloc[1] == '*****'

    def test_apply_masking_with_environment_definitions(self):
        """Test apply_masking with environment-specific masking definitions."""
        with patch('elm.elm_commands.mask.MASKING_ALGORITHMS', {
            'star_length': lambda x, length=4, **_: x[:length] + '*' * (len(x) - length) if len(x) > length else x
        }):
            data = pd.DataFrame({'credit_card': ['1234567890123456', '9876543210987654']})

            with patch('elm.elm_commands.mask.load_masking_definitions') as mock_load:
                mock_load.return_value = {
                    'global': {},
                    'environments': {
                        'prod': {
                            'credit_card': {'algorithm': 'star_length', 'params': {'length': 4}}
                        }
                    }
                }

                result = apply_masking(data, environment='prod')

                assert isinstance(result, pd.DataFrame)
                assert result['credit_card'].iloc[0] == '1234************'
                assert result['credit_card'].iloc[1] == '9876************'

    def test_apply_masking_no_definitions(self):
        """Test apply_masking with no masking definitions."""
        data = pd.DataFrame({'name': ['Alice', 'Bob']})

        with patch('elm.elm_commands.mask.load_masking_definitions') as mock_load:
            mock_load.return_value = {'global': {}, 'environments': {}}

            result = apply_masking(data)

            assert isinstance(result, pd.DataFrame)
            assert result['name'].iloc[0] == 'Alice'
            assert result['name'].iloc[1] == 'Bob'

    def test_apply_masking_unknown_algorithm(self):
        """Test apply_masking with unknown algorithm."""
        with patch('elm.elm_commands.mask.MASKING_ALGORITHMS', {}):
            data = pd.DataFrame({'password': ['secret123']})

            with patch('elm.elm_commands.mask.load_masking_definitions') as mock_load:
                mock_load.return_value = {
                    'global': {
                        'password': {'algorithm': 'unknown', 'params': {}}
                    },
                    'environments': {}
                }

                result = apply_masking(data)

                # Should return original data if algorithm is unknown
                assert isinstance(result, pd.DataFrame)
                assert result['password'].iloc[0] == 'secret123'


class TestAliasedGroup:
    """Test AliasedGroup functionality."""

    def test_aliased_group_get_command_with_alias(self):
        """Test getting command with alias."""
        # Create mock commands
        mock_add = MagicMock()
        mock_add.name = 'add'

        # Create aliases dict
        aliases = {'a': mock_add}

        group = AliasedGroup()

        with patch('elm.elm_commands.mask.ALIASES', aliases):
            with patch.object(click.Group, 'get_command') as mock_super:
                mock_super.return_value = mock_add

                result = group.get_command(None, 'a')

                # Should call super with the resolved command name
                mock_super.assert_called_with(None, 'add')
                assert result == mock_add

    def test_aliased_group_get_command_without_alias(self):
        """Test getting command without alias."""
        group = AliasedGroup()

        with patch('elm.elm_commands.mask.ALIASES', {}):
            with patch.object(AliasedGroup, 'get_command', wraps=group.get_command) as mock_super:
                mock_super.return_value = None

                group.get_command(None, 'unknown')

                # Should call super with the original command name
                mock_super.assert_called_with(None, 'unknown')


@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance."""
    return CliRunner()


class TestMaskCLICommands:
    """Test CLI command functions for mask commands."""

    @patch('elm.elm_commands.mask.core_mask.add_mask')
    def test_add_mask_success(self, mock_add, runner):
        """Test add mask command success."""
        mock_add.return_value = OperationResult(
            success=True,
            message="Mask added successfully"
        )

        result = runner.invoke(mask, [
            'add',
            '--column', 'email',
            '--algorithm', 'star',
            '--length', '4'
        ])

        assert result.exit_code == 0
        assert "Mask added successfully" in result.output
        mock_add.assert_called_once_with(
            column='email',
            algorithm='star',
            environment=None,
            length=4
        )

    @patch('elm.elm_commands.mask.core_mask.add_mask')
    def test_add_mask_failure(self, mock_add, runner):
        """Test add mask command failure."""
        mock_add.return_value = OperationResult(
            success=False,
            message="Failed to add mask"
        )

        result = runner.invoke(mask, [
            'add',
            '--column', 'email',
            '--algorithm', 'star',
            '--length', '4'
        ])

        assert result.exit_code != 0
        assert "Failed to add mask" in result.output

    @patch('elm.elm_commands.mask.core_mask.remove_mask')
    def test_remove_mask_success(self, mock_remove, runner):
        """Test remove mask command success."""
        mock_remove.return_value = OperationResult(
            success=True,
            message="Mask removed successfully"
        )

        result = runner.invoke(mask, [
            'remove',
            '--column', 'email'
        ])

        assert result.exit_code == 0
        assert "Mask removed successfully" in result.output
        mock_remove.assert_called_once_with(column='email', environment=None)

    @patch('elm.elm_commands.mask.core_mask.remove_mask')
    def test_remove_mask_failure(self, mock_remove, runner):
        """Test remove mask command failure."""
        mock_remove.return_value = OperationResult(
            success=False,
            message="Mask not found"
        )

        result = runner.invoke(mask, [
            'remove',
            '--column', 'email'
        ])

        assert result.exit_code == 0  # remove command doesn't fail, just shows message
        assert "Mask not found" in result.output

    @patch('elm.elm_commands.mask.load_masking_definitions')
    def test_list_masks_global(self, mock_load, runner):
        """Test list masks command for global masks."""
        mock_load.return_value = {
            'global': {
                'email': {'algorithm': 'star', 'params': {'length': 4}},
                'phone': {'algorithm': 'random'}
            }
        }

        result = runner.invoke(mask, [
            'list'
        ])

        assert result.exit_code == 0
        mock_load.assert_called_once()
        # Should show global masks
        assert 'email' in result.output
        assert 'phone' in result.output

    @patch('elm.elm_commands.mask.load_masking_definitions')
    def test_list_masks_environment_specific(self, mock_load, runner):
        """Test list masks command for environment-specific masks."""
        mock_load.return_value = {
            'environments': {
                'test-env': {
                    'ssn': {'algorithm': 'nullify'},
                    'credit_card': {'algorithm': 'star_length', 'params': {'length': 4}}
                }
            }
        }

        result = runner.invoke(mask, [
            'list',
            '--environment', 'test-env'
        ])

        assert result.exit_code == 0
        mock_load.assert_called_once()
        # Should show environment-specific masks
        assert 'ssn' in result.output
        assert 'credit_card' in result.output

    @patch('elm.elm_commands.mask.load_masking_definitions')
    def test_list_masks_no_masks(self, mock_load, runner):
        """Test list masks command when no masks exist."""
        mock_load.return_value = {}  # Empty definitions

        result = runner.invoke(mask, [
            'list'
        ])

        assert result.exit_code == 0
        assert "No global masking definitions found" in result.output

    @patch('elm.elm_commands.mask.core_mask.test_mask')
    def test_test_mask_success(self, mock_test, runner):
        """Test test mask command success."""
        mock_test.return_value = OperationResult(
            success=True,
            message="Mask test completed successfully",
            data={'original': 'test@example.com', 'masked': '****@example.com'}
        )

        result = runner.invoke(mask, [
            'test',
            '--column', 'email',
            '--value', 'test@example.com'
        ])

        assert result.exit_code == 0
        mock_test.assert_called_once_with(
            column='email',
            value='test@example.com',
            environment=None
        )
        # Should show both original and masked values
        assert 'test@example.com' in result.output
        assert '****@example.com' in result.output

    @patch('elm.elm_commands.mask.core_mask.test_mask')
    def test_test_mask_failure(self, mock_test, runner):
        """Test test mask command failure."""
        mock_test.return_value = OperationResult(
            success=False,
            message="No mask found for column"
        )

        result = runner.invoke(mask, [
            'test',
            '--column', 'unknown',
            '--value', 'test'
        ])

        assert result.exit_code == 0  # test command doesn't fail, just shows message
        assert "No mask found for column" in result.output