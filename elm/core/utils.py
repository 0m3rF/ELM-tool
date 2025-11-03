"""
ELM Tool Core Utilities

Shared utility functions used across the core module.
These utilities provide common functionality for validation, conversion, and error handling.
"""

import os
import sys
import configparser
from typing import List, Dict, Any, Optional
from elm.core.types import DatabaseType, WriteMode, FileFormat, MaskingAlgorithm, OperationResult
from elm.core.exceptions import ValidationError, ELMError
from elm.elm_utils import variables


def safe_print(message: str) -> None:
    """
    Safely print messages with Unicode characters (including emojis) to console.

    This function handles encoding issues on Windows where the default console
    encoding (cp1252/charmap) cannot handle Unicode emojis and special characters.

    Args:
        message: Message to print (can contain Unicode characters and emojis)
    """
    try:
        # Try to print normally first
        print(message)
    except UnicodeEncodeError:
        # If that fails, encode to UTF-8 and write to stdout buffer
        try:
            # Write to stdout buffer with UTF-8 encoding
            sys.stdout.buffer.write((message + '\n').encode('utf-8'))
            sys.stdout.buffer.flush()
        except Exception:
            # Last resort: replace problematic characters
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            print(safe_message)


def validate_database_type(db_type: str) -> DatabaseType:
    """Validate and convert database type string to enum."""
    try:
        return DatabaseType(db_type.upper())
    except ValueError:
        valid_types = [t.value for t in DatabaseType]
        raise ValidationError(f"Invalid database type '{db_type}'. Valid types: {valid_types}")


def validate_write_mode(mode: str) -> WriteMode:
    """Validate and convert write mode string to enum."""
    # Handle CLI naming inconsistencies
    mode_mapping = {
        'OVERWRITE': WriteMode.REPLACE,
        'REPLACE': WriteMode.REPLACE,
        'APPEND': WriteMode.APPEND,
        'FAIL': WriteMode.FAIL
    }
    
    mode_upper = mode.upper()
    if mode_upper in mode_mapping:
        return mode_mapping[mode_upper]
    
    valid_modes = list(mode_mapping.keys())
    raise ValidationError(f"Invalid write mode '{mode}'. Valid modes: {valid_modes}")


def validate_file_format(file_format: str) -> FileFormat:
    """Validate and convert file format string to enum."""
    try:
        return FileFormat(file_format.lower())
    except ValueError:
        valid_formats = [f.value for f in FileFormat]
        raise ValidationError(f"Invalid file format '{file_format}'. Valid formats: {valid_formats}")


def validate_masking_algorithm(algorithm: str) -> MaskingAlgorithm:
    """Validate and convert masking algorithm string to enum."""
    try:
        return MaskingAlgorithm(algorithm.lower())
    except ValueError:
        valid_algorithms = [a.value for a in MaskingAlgorithm]
        raise ValidationError(f"Invalid masking algorithm '{algorithm}'. Valid algorithms: {valid_algorithms}")


def parse_columns_string(columns_str: str) -> List[str]:
    """Parse comma-separated column string into list."""
    if not columns_str:
        return []
    return [col.strip() for col in columns_str.split(',') if col.strip()]


def ensure_directory_exists(file_path: str) -> None:
    """Ensure the directory for a file path exists."""
    directory = os.path.dirname(os.path.abspath(file_path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def load_environment_config() -> configparser.ConfigParser:
    """Load environment configuration from file with file locking for parallel safety."""
    from elm.elm_utils.file_lock import file_lock

    config = configparser.ConfigParser()
    if os.path.exists(variables.ENVS_FILE):
        try:
            with file_lock(variables.ENVS_FILE, timeout=10.0):
                config.read(variables.ENVS_FILE)
        except TimeoutError:
            # If we can't get the lock, try reading without it
            # This is safe for reads in most cases
            config.read(variables.ENVS_FILE)
    return config


def save_environment_config(config: configparser.ConfigParser) -> None:
    """Save environment configuration to file with file locking for parallel safety."""
    from elm.elm_utils.file_lock import file_lock

    # Ensure directory exists
    os.makedirs(os.path.dirname(variables.ENVS_FILE), exist_ok=True)

    with file_lock(variables.ENVS_FILE, timeout=10.0):
        with open(variables.ENVS_FILE, 'w') as f:
            config.write(f)


def create_success_result(message: str, data: Any = None, record_count: int = None) -> OperationResult:
    """Create a successful operation result."""
    return OperationResult(
        success=True,
        message=message,
        data=data,
        record_count=record_count
    )


def create_error_result(message: str, error_details: str = None) -> OperationResult:
    """Create an error operation result."""
    return OperationResult(
        success=False,
        message=message,
        error_details=error_details
    )


def handle_exception(e: Exception, operation: str) -> OperationResult:
    """Handle exceptions and convert to operation result."""
    if isinstance(e, ELMError):
        return create_error_result(e.message, e.details)
    else:
        return create_error_result(
            f"Error during {operation}: {str(e)}",
            str(type(e).__name__)
        )


def convert_sqlalchemy_mode(mode: WriteMode) -> str:
    """Convert WriteMode enum to SQLAlchemy if_exists parameter."""
    mode_mapping = {
        WriteMode.APPEND: 'append',
        WriteMode.REPLACE: 'replace',
        WriteMode.FAIL: 'fail'
    }
    return mode_mapping[mode]


def validate_required_params(params: Dict[str, Any], required: List[str]) -> None:
    """Validate that required parameters are present and not None."""
    missing = []
    for param in required:
        if param not in params or params[param] is None:
            missing.append(param)
    
    if missing:
        raise ValidationError(f"Missing required parameters: {', '.join(missing)}")


def normalize_boolean_param(value: Any, param_name: str) -> bool:
    """Normalize various boolean representations to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
    
    raise ValidationError(f"Invalid boolean value for {param_name}: {value}")
