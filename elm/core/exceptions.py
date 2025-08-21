"""
ELM Tool Core Exceptions

Custom exception classes for the ELM Tool core module.
These exceptions provide consistent error handling across CLI and API interfaces.
"""


class ELMError(Exception):
    """Base exception class for all ELM Tool errors."""
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert exception to dictionary format for API responses."""
        result = {
            'error': self.__class__.__name__,
            'message': self.message
        }
        if self.details:
            result['details'] = self.details
        return result


class EnvironmentError(ELMError):
    """Exception raised for environment management errors."""
    pass


class CopyError(ELMError):
    """Exception raised for data copy operation errors."""
    pass


class MaskingError(ELMError):
    """Exception raised for data masking errors."""
    pass


class GenerationError(ELMError):
    """Exception raised for data generation errors."""
    pass


class ValidationError(ELMError):
    """Exception raised for parameter validation errors."""
    pass


class DatabaseError(ELMError):
    """Exception raised for database operation errors."""
    pass


class EncryptionError(ELMError):
    """Exception raised for encryption/decryption errors."""
    pass


class FileError(ELMError):
    """Exception raised for file operation errors."""
    pass
