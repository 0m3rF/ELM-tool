"""Tests for core exceptions."""

import pytest
from elm.core.exceptions import (
    ELMError,
    EnvironmentError,
    CopyError,
    MaskingError,
    GenerationError,
    ValidationError,
    DatabaseError,
    EncryptionError,
    FileError
)


class TestELMError:
    """Test cases for ELMError base class."""

    def test_elm_error_basic(self):
        """Test basic ELMError creation."""
        error = ELMError("Test error message")
        assert error.message == "Test error message"
        assert error.details is None
        assert str(error) == "Test error message"

    def test_elm_error_with_details(self):
        """Test ELMError with details."""
        error = ELMError("Test error message", "Additional details")
        assert error.message == "Test error message"
        assert error.details == "Additional details"

    def test_elm_error_to_dict_without_details(self):
        """Test to_dict method without details."""
        error = ELMError("Test error message")
        result = error.to_dict()
        
        assert result == {
            'error': 'ELMError',
            'message': 'Test error message'
        }
        assert 'details' not in result

    def test_elm_error_to_dict_with_details(self):
        """Test to_dict method with details."""
        error = ELMError("Test error message", "Additional details")
        result = error.to_dict()
        
        assert result == {
            'error': 'ELMError',
            'message': 'Test error message',
            'details': 'Additional details'
        }

    def test_elm_error_inheritance(self):
        """Test that ELMError inherits from Exception."""
        error = ELMError("Test error")
        assert isinstance(error, Exception)


class TestSpecificErrors:
    """Test cases for specific error types."""

    def test_environment_error(self):
        """Test EnvironmentError."""
        error = EnvironmentError("Environment not found", "Check configuration")
        assert error.message == "Environment not found"
        assert error.details == "Check configuration"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'EnvironmentError'

    def test_copy_error(self):
        """Test CopyError."""
        error = CopyError("Copy operation failed")
        assert error.message == "Copy operation failed"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'CopyError'

    def test_masking_error(self):
        """Test MaskingError."""
        error = MaskingError("Invalid masking algorithm")
        assert error.message == "Invalid masking algorithm"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'MaskingError'

    def test_generation_error(self):
        """Test GenerationError."""
        error = GenerationError("Data generation failed")
        assert error.message == "Data generation failed"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'GenerationError'

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid parameter")
        assert error.message == "Invalid parameter"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'ValidationError'

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Database connection failed")
        assert error.message == "Database connection failed"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'DatabaseError'

    def test_encryption_error(self):
        """Test EncryptionError."""
        error = EncryptionError("Decryption failed")
        assert error.message == "Decryption failed"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'EncryptionError'

    def test_file_error(self):
        """Test FileError."""
        error = FileError("File not found")
        assert error.message == "File not found"
        assert isinstance(error, ELMError)
        
        result = error.to_dict()
        assert result['error'] == 'FileError'


class TestErrorRaising:
    """Test that errors can be raised and caught properly."""

    def test_raise_elm_error(self):
        """Test raising ELMError."""
        with pytest.raises(ELMError) as exc_info:
            raise ELMError("Test error")
        
        assert exc_info.value.message == "Test error"

    def test_raise_environment_error(self):
        """Test raising EnvironmentError."""
        with pytest.raises(EnvironmentError) as exc_info:
            raise EnvironmentError("Environment error")
        
        assert exc_info.value.message == "Environment error"

    def test_catch_as_elm_error(self):
        """Test catching specific error as ELMError."""
        with pytest.raises(ELMError):
            raise CopyError("Copy failed")

    def test_catch_as_exception(self):
        """Test catching ELMError as Exception."""
        with pytest.raises(Exception):
            raise ELMError("Test error")


class TestErrorDetails:
    """Test error details functionality."""

    def test_all_errors_support_details(self):
        """Test that all error types support details parameter."""
        error_classes = [
            EnvironmentError,
            CopyError,
            MaskingError,
            GenerationError,
            ValidationError,
            DatabaseError,
            EncryptionError,
            FileError
        ]
        
        for error_class in error_classes:
            error = error_class("Test message", "Test details")
            assert error.details == "Test details"
            
            result = error.to_dict()
            assert 'details' in result
            assert result['details'] == "Test details"

