"""
Tests for the core utils module.
"""
import pytest
import os
import configparser
from unittest.mock import patch, MagicMock, mock_open

from elm.core import utils
from elm.core.types import DatabaseType, WriteMode, FileFormat, MaskingAlgorithm, OperationResult
from elm.core.exceptions import ValidationError, ELMError


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_database_type_valid(self):
        """Test validating valid database types."""
        assert utils.validate_database_type('ORACLE') == DatabaseType.ORACLE
        assert utils.validate_database_type('oracle') == DatabaseType.ORACLE
        assert utils.validate_database_type('POSTGRES') == DatabaseType.POSTGRES
        assert utils.validate_database_type('postgres') == DatabaseType.POSTGRES
        assert utils.validate_database_type('MYSQL') == DatabaseType.MYSQL
        assert utils.validate_database_type('mysql') == DatabaseType.MYSQL
        assert utils.validate_database_type('MSSQL') == DatabaseType.MSSQL
        assert utils.validate_database_type('mssql') == DatabaseType.MSSQL

    def test_validate_database_type_invalid(self):
        """Test validating invalid database types."""
        with pytest.raises(ValidationError) as exc_info:
            utils.validate_database_type('INVALID')
        
        assert "Invalid database type 'INVALID'" in str(exc_info.value)
        assert "Valid types:" in str(exc_info.value)

    def test_validate_write_mode_valid(self):
        """Test validating valid write modes."""
        assert utils.validate_write_mode('OVERWRITE') == WriteMode.REPLACE
        assert utils.validate_write_mode('REPLACE') == WriteMode.REPLACE
        assert utils.validate_write_mode('replace') == WriteMode.REPLACE
        assert utils.validate_write_mode('APPEND') == WriteMode.APPEND
        assert utils.validate_write_mode('append') == WriteMode.APPEND
        assert utils.validate_write_mode('FAIL') == WriteMode.FAIL
        assert utils.validate_write_mode('fail') == WriteMode.FAIL

    def test_validate_write_mode_invalid(self):
        """Test validating invalid write modes."""
        with pytest.raises(ValidationError) as exc_info:
            utils.validate_write_mode('INVALID')
        
        assert "Invalid write mode 'INVALID'" in str(exc_info.value)
        assert "Valid modes:" in str(exc_info.value)

    def test_validate_file_format_valid(self):
        """Test validating valid file formats."""
        assert utils.validate_file_format('CSV') == FileFormat.CSV
        assert utils.validate_file_format('csv') == FileFormat.CSV
        assert utils.validate_file_format('JSON') == FileFormat.JSON
        assert utils.validate_file_format('json') == FileFormat.JSON

    def test_validate_file_format_invalid(self):
        """Test validating invalid file formats."""
        with pytest.raises(ValidationError) as exc_info:
            utils.validate_file_format('INVALID')
        
        assert "Invalid file format 'INVALID'" in str(exc_info.value)
        assert "Valid formats:" in str(exc_info.value)

    def test_validate_masking_algorithm_valid(self):
        """Test validating valid masking algorithms."""
        assert utils.validate_masking_algorithm('STAR') == MaskingAlgorithm.STAR
        assert utils.validate_masking_algorithm('star') == MaskingAlgorithm.STAR
        assert utils.validate_masking_algorithm('STAR_LENGTH') == MaskingAlgorithm.STAR_LENGTH
        assert utils.validate_masking_algorithm('star_length') == MaskingAlgorithm.STAR_LENGTH
        assert utils.validate_masking_algorithm('RANDOM') == MaskingAlgorithm.RANDOM
        assert utils.validate_masking_algorithm('random') == MaskingAlgorithm.RANDOM
        assert utils.validate_masking_algorithm('NULLIFY') == MaskingAlgorithm.NULLIFY
        assert utils.validate_masking_algorithm('nullify') == MaskingAlgorithm.NULLIFY

    def test_validate_masking_algorithm_invalid(self):
        """Test validating invalid masking algorithms."""
        with pytest.raises(ValidationError) as exc_info:
            utils.validate_masking_algorithm('INVALID')
        
        assert "Invalid masking algorithm 'INVALID'" in str(exc_info.value)
        assert "Valid algorithms:" in str(exc_info.value)


class TestStringParsing:
    """Test string parsing utilities."""

    def test_parse_columns_string_valid(self):
        """Test parsing valid column strings."""
        assert utils.parse_columns_string('col1,col2,col3') == ['col1', 'col2', 'col3']
        assert utils.parse_columns_string('col1, col2, col3') == ['col1', 'col2', 'col3']
        assert utils.parse_columns_string('  col1  ,  col2  ,  col3  ') == ['col1', 'col2', 'col3']
        assert utils.parse_columns_string('single_column') == ['single_column']

    def test_parse_columns_string_empty(self):
        """Test parsing empty column strings."""
        assert utils.parse_columns_string('') == []
        assert utils.parse_columns_string('   ') == []
        assert utils.parse_columns_string(',,,') == []

    def test_parse_columns_string_with_empty_columns(self):
        """Test parsing column strings with empty columns."""
        assert utils.parse_columns_string('col1,,col3') == ['col1', 'col3']
        assert utils.parse_columns_string('col1, , col3') == ['col1', 'col3']


class TestFileOperations:
    """Test file operation utilities."""

    @patch('os.makedirs')
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    def test_ensure_directory_exists(self, mock_abspath, mock_dirname, mock_makedirs):
        """Test ensuring directory exists."""
        mock_abspath.return_value = '/abs/path/to/file.txt'
        mock_dirname.return_value = '/abs/path/to'

        utils.ensure_directory_exists('/path/to/file.txt')

        mock_abspath.assert_called_once_with('/path/to/file.txt')
        mock_dirname.assert_called_once_with('/abs/path/to/file.txt')
        mock_makedirs.assert_called_once_with('/abs/path/to', exist_ok=True)

    @patch('os.makedirs')
    @patch('os.path.dirname')
    @patch('os.path.abspath')
    def test_ensure_directory_exists_no_directory(self, mock_abspath, mock_dirname, mock_makedirs):
        """Test ensuring directory exists when no directory."""
        mock_abspath.return_value = '/file.txt'
        mock_dirname.return_value = ''

        utils.ensure_directory_exists('file.txt')

        mock_abspath.assert_called_once_with('file.txt')
        mock_dirname.assert_called_once_with('/file.txt')
        mock_makedirs.assert_not_called()


class TestConfigOperations:
    """Test configuration operation utilities."""

    @patch('os.path.exists')
    @patch('configparser.ConfigParser.read')
    def test_load_environment_config_exists(self, mock_read, mock_exists):
        """Test loading environment config when file exists."""
        mock_exists.return_value = True
        
        config = utils.load_environment_config()
        
        assert isinstance(config, configparser.ConfigParser)
        mock_exists.assert_called_once()
        mock_read.assert_called_once()

    @patch('os.path.exists')
    def test_load_environment_config_not_exists(self, mock_exists):
        """Test loading environment config when file doesn't exist."""
        mock_exists.return_value = False
        
        config = utils.load_environment_config()
        
        assert isinstance(config, configparser.ConfigParser)
        mock_exists.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    @patch('os.path.dirname')
    def test_save_environment_config(self, mock_dirname, mock_makedirs, mock_file):
        """Test saving environment config."""
        mock_dirname.return_value = '/path/to'
        mock_config = MagicMock()
        
        utils.save_environment_config(mock_config)
        
        mock_dirname.assert_called_once()
        mock_makedirs.assert_called_once_with('/path/to', exist_ok=True)
        mock_file.assert_called_once()
        mock_config.write.assert_called_once()


class TestResultCreation:
    """Test operation result creation utilities."""

    def test_create_success_result_basic(self):
        """Test creating basic success result."""
        result = utils.create_success_result("Success message")
        
        assert result.success is True
        assert result.message == "Success message"
        assert result.data is None
        assert result.record_count is None

    def test_create_success_result_with_data(self):
        """Test creating success result with data."""
        data = {'key': 'value'}
        result = utils.create_success_result("Success message", data=data, record_count=5)
        
        assert result.success is True
        assert result.message == "Success message"
        assert result.data == data
        assert result.record_count == 5

    def test_create_error_result_basic(self):
        """Test creating basic error result."""
        result = utils.create_error_result("Error message")
        
        assert result.success is False
        assert result.message == "Error message"
        assert result.error_details is None

    def test_create_error_result_with_details(self):
        """Test creating error result with details."""
        result = utils.create_error_result("Error message", error_details="Detailed error")
        
        assert result.success is False
        assert result.message == "Error message"
        assert result.error_details == "Detailed error"


class TestExceptionHandling:
    """Test exception handling utilities."""

    def test_handle_exception_elm_error(self):
        """Test handling ELM-specific errors."""
        elm_error = ELMError("ELM error message", "Error details")
        
        result = utils.handle_exception(elm_error, "test operation")
        
        assert result.success is False
        assert result.message == "ELM error message"
        assert result.error_details == "Error details"

    def test_handle_exception_generic_error(self):
        """Test handling generic errors."""
        generic_error = ValueError("Generic error message")
        
        result = utils.handle_exception(generic_error, "test operation")
        
        assert result.success is False
        assert "Error during test operation: Generic error message" in result.message
        assert result.error_details == "ValueError"


class TestConversionUtilities:
    """Test conversion utility functions."""

    def test_convert_sqlalchemy_mode(self):
        """Test converting WriteMode to SQLAlchemy mode."""
        assert utils.convert_sqlalchemy_mode(WriteMode.APPEND) == 'append'
        assert utils.convert_sqlalchemy_mode(WriteMode.REPLACE) == 'replace'
        assert utils.convert_sqlalchemy_mode(WriteMode.FAIL) == 'fail'


class TestParameterValidation:
    """Test parameter validation utilities."""

    def test_validate_required_params_all_present(self):
        """Test validating required parameters when all are present."""
        params = {'param1': 'value1', 'param2': 'value2', 'param3': 'value3'}
        required = ['param1', 'param2']
        
        # Should not raise an exception
        utils.validate_required_params(params, required)

    def test_validate_required_params_missing(self):
        """Test validating required parameters when some are missing."""
        params = {'param1': 'value1'}
        required = ['param1', 'param2', 'param3']
        
        with pytest.raises(ValidationError) as exc_info:
            utils.validate_required_params(params, required)
        
        assert "Missing required parameters: param2, param3" in str(exc_info.value)

    def test_validate_required_params_none_values(self):
        """Test validating required parameters when some are None."""
        params = {'param1': 'value1', 'param2': None, 'param3': 'value3'}
        required = ['param1', 'param2']
        
        with pytest.raises(ValidationError) as exc_info:
            utils.validate_required_params(params, required)
        
        assert "Missing required parameters: param2" in str(exc_info.value)


class TestBooleanNormalization:
    """Test boolean normalization utilities."""

    def test_normalize_boolean_param_bool_values(self):
        """Test normalizing actual boolean values."""
        assert utils.normalize_boolean_param(True, 'test_param') is True
        assert utils.normalize_boolean_param(False, 'test_param') is False

    def test_normalize_boolean_param_string_true_values(self):
        """Test normalizing string values that represent True."""
        assert utils.normalize_boolean_param('true', 'test_param') is True
        assert utils.normalize_boolean_param('TRUE', 'test_param') is True
        assert utils.normalize_boolean_param('True', 'test_param') is True
        assert utils.normalize_boolean_param('1', 'test_param') is True
        assert utils.normalize_boolean_param('yes', 'test_param') is True
        assert utils.normalize_boolean_param('YES', 'test_param') is True
        assert utils.normalize_boolean_param('on', 'test_param') is True
        assert utils.normalize_boolean_param('ON', 'test_param') is True

    def test_normalize_boolean_param_string_false_values(self):
        """Test normalizing string values that represent False."""
        assert utils.normalize_boolean_param('false', 'test_param') is False
        assert utils.normalize_boolean_param('FALSE', 'test_param') is False
        assert utils.normalize_boolean_param('False', 'test_param') is False
        assert utils.normalize_boolean_param('0', 'test_param') is False
        assert utils.normalize_boolean_param('no', 'test_param') is False
        assert utils.normalize_boolean_param('NO', 'test_param') is False
        assert utils.normalize_boolean_param('off', 'test_param') is False
        assert utils.normalize_boolean_param('OFF', 'test_param') is False

    def test_normalize_boolean_param_invalid_values(self):
        """Test normalizing invalid boolean values."""
        with pytest.raises(ValidationError) as exc_info:
            utils.normalize_boolean_param('invalid', 'test_param')
        
        assert "Invalid boolean value for test_param: invalid" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            utils.normalize_boolean_param(123, 'test_param')
        
        assert "Invalid boolean value for test_param: 123" in str(exc_info.value)
