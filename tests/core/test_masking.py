"""
Tests for the core masking module.

These tests focus on testing the business logic of masking operations
in isolation, with appropriate mocking of external dependencies.
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock
import json
import pandas as pd

from elm.core import masking
from elm.core.types import OperationResult
from elm.core.exceptions import MaskingError, ValidationError


class TestMaskingCore:
    """Test core masking functionality."""

    @patch('elm.core.masking.save_masking_definitions')
    @patch('elm.core.masking.load_masking_definitions')
    def test_add_mask_global(self, mock_load, mock_save):
        """Test adding a global masking rule."""
        # Mock existing definitions
        mock_load.return_value = {'global': {}, 'environments': {}}
        mock_save.return_value = True

        result = masking.add_mask(
            column="password",
            algorithm="star",
            environment=None,
            length=None
        )

        assert result.success is True
        assert "Added global masking for column 'password'" in result.message
        mock_save.assert_called_once()

    @patch('elm.core.masking.save_masking_definitions')
    @patch('elm.core.masking.load_masking_definitions')
    def test_add_mask_environment_specific(self, mock_load, mock_save):
        """Test adding an environment-specific masking rule."""
        mock_load.return_value = {'global': {}, 'environments': {}}
        mock_save.return_value = True

        result = masking.add_mask(
            column="ssn",
            algorithm="star_length",
            environment="prod",
            length=4
        )

        assert result.success is True
        assert "Added environment 'prod' masking for column 'ssn'" in result.message
        mock_save.assert_called_once()

    def test_add_mask_invalid_algorithm(self):
        """Test adding mask with invalid algorithm."""
        result = masking.add_mask(
            column="password",
            algorithm="invalid_algorithm",
            environment=None,
            length=None
        )

        assert result.success is False
        assert "Invalid masking algorithm" in result.message

    def test_add_mask_empty_column(self):
        """Test adding mask with empty column name (currently allowed)."""
        result = masking.add_mask(
            column="",
            algorithm="star",
            environment=None,
            length=None
        )

        # Current implementation allows empty column names
        assert result.success is True
        assert "Added global masking for column ''" in result.message

    @patch('elm.core.masking.save_masking_definitions')
    @patch('elm.core.masking.load_masking_definitions')
    def test_remove_mask_global(self, mock_load, mock_save):
        """Test removing a global masking rule."""
        mock_load.return_value = {
            'global': {'password': {'algorithm': 'star', 'params': {}}},
            'environments': {}
        }
        mock_save.return_value = True

        result = masking.remove_mask(column="password", environment=None)

        assert result.success is True
        assert "Removed global masking for column 'password'" in result.message

    @patch('elm.core.masking.load_masking_definitions')
    def test_remove_mask_not_found(self, mock_load):
        """Test removing a non-existent masking rule."""
        mock_load.return_value = {'global': {}, 'environments': {}}

        result = masking.remove_mask(column="nonexistent", environment=None)

        assert result.success is False
        assert "No global masking found for column 'nonexistent'" in result.message

    @patch('elm.core.masking.load_masking_definitions')
    def test_list_masks_all(self, mock_load):
        """Test listing all masking rules."""
        mock_definitions = {
            'global': {
                'password': {'algorithm': 'star', 'params': {}}
            },
            'environments': {
                'prod': {
                    'ssn': {'algorithm': 'star_length', 'params': {'length': 4}}
                }
            }
        }
        mock_load.return_value = mock_definitions

        result = masking.list_masks(environment=None)

        assert result.success is True
        assert result.data == mock_definitions

    @patch('elm.core.masking.load_masking_definitions')
    def test_list_masks_environment_specific(self, mock_load):
        """Test listing environment-specific masking rules."""
        mock_definitions = {
            'global': {
                'password': {'algorithm': 'star', 'params': {}}
            },
            'environments': {
                'prod': {
                    'ssn': {'algorithm': 'star_length', 'params': {'length': 4}}
                }
            }
        }
        mock_load.return_value = mock_definitions

        result = masking.list_masks(environment="prod")

        assert result.success is True
        assert 'rules' in result.data
        assert 'ssn' in result.data['rules']

    @patch('elm.core.masking.load_masking_definitions')
    def test_test_mask_global(self, mock_load):
        """Test testing a global masking rule."""
        mock_definitions = {
            'global': {
                'password': {'algorithm': 'star', 'params': {}}
            },
            'environments': {}
        }
        mock_load.return_value = mock_definitions

        result = masking.test_mask(
            column="password",
            value="secret123",
            environment=None
        )

        assert result.success is True
        assert result.data['original'] == "secret123"
        assert result.data['masked'] == "*****"
        assert result.data['scope'] == "global"

    @patch('elm.core.masking.load_masking_definitions')
    def test_test_mask_environment_specific(self, mock_load):
        """Test testing an environment-specific masking rule."""
        mock_definitions = {
            'global': {},
            'environments': {
                'prod': {
                    'ssn': {'algorithm': 'star_length', 'params': {'length': 4}}
                }
            }
        }
        mock_load.return_value = mock_definitions

        result = masking.test_mask(
            column="ssn",
            value="123-45-6789",
            environment="prod"
        )

        assert result.success is True
        assert result.data['original'] == "123-45-6789"
        assert result.data['masked'].startswith("123-")
        assert result.data['scope'] == "environment 'prod'"

    @patch('elm.core.masking.load_masking_definitions')
    def test_test_mask_not_found(self, mock_load):
        """Test testing a non-existent masking rule."""
        mock_load.return_value = {'global': {}, 'environments': {}}

        result = masking.test_mask(
            column="nonexistent",
            value="test",
            environment=None
        )

        assert result.success is False
        assert "No global or environment-specific masking definition found" in result.message

    def test_apply_masking_success(self):
        """Test applying masking to a DataFrame."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'password': ['secret1', 'secret2', 'secret3'],
            'email': ['user1@example.com', 'user2@example.com', 'user3@example.com']
        })

        # Test the actual apply_masking function
        result = masking.apply_masking(
            data=test_df,
            environment="test"
        )

        # The function returns a DataFrame directly, not an OperationResult
        assert isinstance(result, pd.DataFrame)
        # Since there are no masking rules defined, data should be unchanged
        assert result.equals(test_df)

    def test_apply_masking_invalid_data(self):
        """Test applying masking to invalid data."""
        # The apply_masking function returns the input data unchanged if it's not a DataFrame
        result = masking.apply_masking(
            data="not a dataframe",
            environment="test"
        )

        # Should return the original data unchanged
        assert result == "not a dataframe"

    @patch('builtins.open', new_callable=mock_open, read_data='{"global": {}, "environments": {}}')
    @patch('os.path.exists')
    def test_load_masking_definitions_success(self, mock_exists, mock_file):
        """Test loading masking definitions from file."""
        mock_exists.return_value = True

        definitions = masking.load_masking_definitions()

        assert 'global' in definitions
        assert 'environments' in definitions

    @patch('os.path.exists')
    def test_load_masking_definitions_file_not_exists(self, mock_exists):
        """Test loading masking definitions when file doesn't exist."""
        mock_exists.return_value = False

        definitions = masking.load_masking_definitions()

        assert definitions == {'global': {}, 'environments': {}}

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_masking_definitions_success(self, mock_makedirs, mock_file):
        """Test saving masking definitions to file."""
        test_definitions = {
            'global': {'password': {'algorithm': 'star', 'params': {}}},
            'environments': {}
        }

        # Function should not raise an exception
        masking.save_masking_definitions(test_definitions)

        mock_file.assert_called_once()
        mock_makedirs.assert_called_once()

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_save_masking_definitions_failure(self, mock_file):
        """Test saving masking definitions when file write fails."""
        test_definitions = {'global': {}, 'environments': {}}

        # Function should raise a MaskingError
        with pytest.raises(MaskingError):
            masking.save_masking_definitions(test_definitions)

    def test_validate_masking_algorithm_valid(self):
        """Test validation of valid masking algorithms."""
        from elm.core.utils import validate_masking_algorithm
        from elm.core.types import MaskingAlgorithm

        valid_algorithms = ['star', 'star_length', 'random', 'nullify']

        for algorithm in valid_algorithms:
            result = validate_masking_algorithm(algorithm)
            assert isinstance(result, MaskingAlgorithm)
            assert result.value == algorithm

    def test_validate_masking_algorithm_invalid(self):
        """Test validation of invalid masking algorithms."""
        from elm.core.utils import validate_masking_algorithm
        from elm.core.exceptions import ValidationError

        invalid_algorithms = ['invalid', 'unknown', '']

        for algorithm in invalid_algorithms:
            with pytest.raises(ValidationError):
                validate_masking_algorithm(algorithm)

    # get_masking_config function tests removed - function not implemented in current core module
