"""
ELM Tool Core Type Definitions

Type definitions and data models for the ELM Tool core module.
These types ensure consistency across CLI and API interfaces.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class DatabaseType(Enum):
    """Supported database types."""
    ORACLE = "ORACLE"
    POSTGRES = "POSTGRES"
    MYSQL = "MYSQL"
    MSSQL = "MSSQL"


class WriteMode(Enum):
    """Data write modes."""
    APPEND = "APPEND"
    REPLACE = "REPLACE"
    FAIL = "FAIL"


class FileFormat(Enum):
    """Supported file formats."""
    CSV = "csv"
    JSON = "json"


class MaskingAlgorithm(Enum):
    """Supported masking algorithms."""
    STAR = "star"
    STAR_LENGTH = "star_length"
    RANDOM = "random"
    NULLIFY = "nullify"


@dataclass
class EnvironmentConfig:
    """Configuration for database environments."""
    name: str
    host: str
    port: int
    user: str
    password: str
    service: str
    db_type: DatabaseType
    is_encrypted: bool = False
    encryption_key: Optional[str] = None


@dataclass
class CopyConfig:
    """Configuration for data copy operations."""
    source_env: Optional[str] = None
    target_env: Optional[str] = None
    query: Optional[str] = None
    table: Optional[str] = None
    file_path: Optional[str] = None
    file_format: FileFormat = FileFormat.CSV
    mode: WriteMode = WriteMode.APPEND
    batch_size: Optional[int] = None
    parallel_workers: int = 1
    apply_masking: bool = True
    validate_target: bool = False
    create_if_not_exists: bool = False
    source_encryption_key: Optional[str] = None
    target_encryption_key: Optional[str] = None


@dataclass
class MaskingConfig:
    """Configuration for data masking operations."""
    column: str
    algorithm: MaskingAlgorithm
    environment: Optional[str] = None
    length: Optional[int] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class GenerationConfig:
    """Configuration for data generation operations."""
    num_records: int = 10
    columns: Optional[List[str]] = None
    environment: Optional[str] = None
    table: Optional[str] = None
    output_file: Optional[str] = None
    file_format: FileFormat = FileFormat.CSV
    write_to_db: bool = False
    mode: WriteMode = WriteMode.APPEND
    string_length: int = 10
    pattern: Optional[Dict[str, str]] = None
    min_number: float = 0
    max_number: float = 100
    decimal_places: int = 2
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date_format: str = '%Y-%m-%d'


@dataclass
class OperationResult:
    """Standard result format for all operations."""
    success: bool
    message: str
    data: Optional[Any] = None
    record_count: Optional[int] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result = {
            'success': self.success,
            'message': self.message
        }
        if self.data is not None:
            result['data'] = self.data
        if self.record_count is not None:
            result['record_count'] = self.record_count
        if self.error_details is not None:
            result['error_details'] = self.error_details
        return result
