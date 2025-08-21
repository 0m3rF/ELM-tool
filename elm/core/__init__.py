"""
ELM Tool Core Module

This module contains the core business logic for the ELM Tool, providing
unified implementations that are used by both CLI commands and API functions.

The core module ensures consistency between different interfaces and eliminates
code duplication by centralizing business logic.

Modules:
    environment: Environment management operations
    copy: Data copy operations (db2file, file2db, db2db)
    masking: Data masking operations
    generation: Data generation operations
    exceptions: Custom exception classes
    types: Type definitions and data models
"""

from elm.core.exceptions import (
    ELMError,
    EnvironmentError,
    CopyError,
    MaskingError,
    GenerationError,
    ValidationError
)

from elm.core.types import (
    EnvironmentConfig,
    CopyConfig,
    MaskingConfig,
    GenerationConfig,
    OperationResult
)

__all__ = [
    'ELMError',
    'EnvironmentError', 
    'CopyError',
    'MaskingError',
    'GenerationError',
    'ValidationError',
    'EnvironmentConfig',
    'CopyConfig',
    'MaskingConfig',
    'GenerationConfig',
    'OperationResult'
]
