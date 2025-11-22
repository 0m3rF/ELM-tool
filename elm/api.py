"""
ELM Tool API - Programmatic interface for the ELM Tool

This module provides functions for programmatically using the ELM Tool
without going through the command-line interface.

All API functions now use the unified core modules to ensure consistency
with CLI commands and eliminate code duplication.
"""

import pandas as pd
from typing import Dict, List, Union, Optional, Any

# Import core modules for unified business logic
from elm.core import environment as core_env
from elm.core import copy as core_copy
from elm.core import masking as core_mask
from elm.core import generation as core_gen
from elm.core import config as core_config
from elm.core.types import OperationResult

# Environment Management Functions

def create_environment(
    name: str,
    host: str,
    port: int,
    user: str,
    password: str,
    service: str,
    db_type: str,
    encrypt: bool = False,
    encryption_key: Optional[str] = None,
    overwrite: bool = False,
    connection_type: Optional[str] = None
) -> bool:
    """
    Create a new database environment.

    Args:
        name: Environment name
        host: Database host
        port: Database port
        user: Database username
        password: Database password
        service: Database service name (or SID for Oracle)
        db_type: Database type (ORACLE, MYSQL, MSSQL, POSTGRES)
        encrypt: Whether to encrypt the environment
        encryption_key: Encryption key (required if encrypt=True)
        overwrite: Whether to overwrite if environment already exists
        connection_type: Oracle connection type ('service_name' or 'sid'). Defaults to 'service_name'

    Returns:
        bool: True if successful, False otherwise
    """
    result = core_env.create_environment(
        name=name,
        host=host,
        port=port,
        user=user,
        password=password,
        service=service,
        db_type=db_type,
        encrypt=encrypt,
        encryption_key=encryption_key,
        overwrite=overwrite,
        connection_type=connection_type
    )
    return result.success

def list_environments(show_all: bool = False) -> List[Dict[str, Any]]:
    """
    List all environments.

    Args:
        show_all: Whether to show all details (passwords will be masked)

    Returns:
        List of environment dictionaries
    """
    result = core_env.list_environments(show_all=show_all)
    return result.data if result.success else []

def get_environment(name: str, encryption_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get details of a specific environment.

    Args:
        name: Environment name
        encryption_key: Encryption key for encrypted environments

    Returns:
        Environment details dictionary or None if not found
    """
    result = core_env.get_environment(name=name, encryption_key=encryption_key)
    return result.data if result.success else None

def delete_environment(name: str) -> bool:
    """
    Delete an environment.

    Args:
        name: Environment name

    Returns:
        bool: True if successful, False otherwise
    """
    result = core_env.delete_environment(name=name)
    return result.success

def test_environment(name: str, encryption_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Test database connection for an environment.

    Args:
        name: Environment name
        encryption_key: Encryption key for encrypted environments

    Returns:
        Dictionary with test results
    """
    result = core_env.test_environment(name=name, encryption_key=encryption_key)
    return result.to_dict()

def execute_sql(
    environment: str,
    query: str,
    encryption_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Execute SQL query on an environment.

    Args:
        environment: Environment name
        query: SQL query to execute
        encryption_key: Encryption key for encrypted environments
        params: Query parameters

    Returns:
        DataFrame with query results
    """
    result = core_env.execute_sql(
        environment=environment,
        query=query,
        encryption_key=encryption_key,
        params=params
    )
    if result.success and result.data:
        return pd.DataFrame(result.data)
    else:
        return pd.DataFrame()

def update_environment(
    name: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    service: Optional[str] = None,
    db_type: Optional[str] = None,
    encrypt: Optional[bool] = None,
    encryption_key: Optional[str] = None
) -> bool:
    """
    Update an existing environment.

    Args:
        name: Environment name
        host: New database host
        port: New database port
        user: New database username
        password: New database password
        service: New database service name
        db_type: New database type
        encrypt: Whether to encrypt the environment
        encryption_key: Encryption key (required if encrypt=True)

    Returns:
        bool: True if successful, False otherwise
    """
    result = core_env.update_environment(
        name=name,
        host=host,
        port=port,
        user=user,
        password=password,
        service=service,
        db_type=db_type,
        encrypt=encrypt,
        encryption_key=encryption_key
    )
    return result.success

# Data Copy Functions

def copy_db_to_file(
    source_env: str,
    query: str,
    file_path: str,
    file_format: str = 'csv',
    mode: str = 'REPLACE',
    batch_size: Optional[int] = None,
    parallel_workers: int = 1,
    source_encryption_key: Optional[str] = None,
    apply_masks: bool = True,
    verbose_batch_logs: bool = True,
) -> Dict[str, Any]:
    """
    Copy data from database to file.

    Args:
        source_env: Source environment name
        query: SQL query to execute
        file_path: Output file path
        file_format: Output file format (csv, json)
        mode: Write mode (REPLACE, APPEND)
        batch_size: Batch size for processing large datasets
        parallel_workers: Number of parallel workers
        source_encryption_key: Encryption key for source environment
        apply_masks: Whether to apply masking rules
        verbose_batch_logs: Whether to log per-batch timings (overall summary is always logged)

    Returns:
        Dictionary with operation results
    """
    result = core_copy.copy_db_to_file(
        source_env=source_env,
        query=query,
        file_path=file_path,
        file_format=file_format,
        mode=mode,
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        source_encryption_key=source_encryption_key,
        apply_masks=apply_masks,
        verbose_batch_logs=verbose_batch_logs,
    )
    return result.to_dict()

def copy_file_to_db(
    file_path: str,
    target_env: str,
    table: str,
    file_format: str = 'csv',
    mode: str = 'APPEND',
    batch_size: Optional[int] = 1000,
    parallel_workers: int = 1,
    target_encryption_key: Optional[str] = None,
    validate_target: bool = False,
    create_if_not_exists: bool = False,
    apply_masks: bool = True,
    verbose_batch_logs: bool = True,
) -> Dict[str, Any]:
    """
    Copy data from file to database.

    Args:
        file_path: Input file path
        target_env: Target environment name
        table: Target table name
        file_format: Input file format (csv, json)
        mode: Write mode (APPEND, REPLACE, FAIL)
        batch_size: Batch size for writing
        parallel_workers: Number of parallel workers
        target_encryption_key: Encryption key for target environment
        validate_target: Whether to validate target table
        create_if_not_exists: Whether to create target table if it doesn't exist
        apply_masks: Whether to apply masking rules
        verbose_batch_logs: Whether to log per-batch timings (overall summary is always logged)

    Returns:
        Dictionary with operation results
    """
    result = core_copy.copy_file_to_db(
        file_path=file_path,
        target_env=target_env,
        table=table,
        file_format=file_format,
        mode=mode,
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        target_encryption_key=target_encryption_key,
        validate_target=validate_target,
        create_if_not_exists=create_if_not_exists,
        apply_masks=apply_masks,
        verbose_batch_logs=verbose_batch_logs,
    )
    return result.to_dict()

def copy_db_to_db(
    source_env: str,
    target_env: str,
    query: str,
    table: str,
    mode: str = 'APPEND',
    batch_size: Optional[int] = 1000,
    parallel_workers: int = 1,
    source_encryption_key: Optional[str] = None,
    target_encryption_key: Optional[str] = None,
    validate_target: bool = False,
    create_if_not_exists: bool = False,
    apply_masks: bool = True,
    verbose_batch_logs: bool = True,
) -> Dict[str, Any]:
    """
    Copy data from database to database.

    Args:
        source_env: Source environment name
        target_env: Target environment name
        query: SQL query to execute on source
        table: Target table name
        mode: Write mode (APPEND, REPLACE, FAIL)
        batch_size: Batch size for writing
        parallel_workers: Number of parallel workers
        source_encryption_key: Encryption key for source environment
        target_encryption_key: Encryption key for target environment
        validate_target: Whether to validate target table
        create_if_not_exists: Whether to create target table if it doesn't exist
        apply_masks: Whether to apply masking rules
        verbose_batch_logs: Whether to print per-batch timing logs (in addition to summary)

    Returns:
        Dictionary with operation results
    """
    result = core_copy.copy_db_to_db(
        source_env=source_env,
        target_env=target_env,
        query=query,
        table=table,
        mode=mode,
        batch_size=batch_size,
        parallel_workers=parallel_workers,
        source_encryption_key=source_encryption_key,
        target_encryption_key=target_encryption_key,
        validate_target=validate_target,
        create_if_not_exists=create_if_not_exists,
        apply_masks=apply_masks,
        verbose_batch_logs=verbose_batch_logs,
    )
    return result.to_dict()

# Data Masking Functions

def add_mask(
    column: str,
    algorithm: str,
    environment: Optional[str] = None,
    length: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Add a masking rule for a column.

    Args:
        column: Column name
        algorithm: Masking algorithm (star, star_length, random, nullify)
        environment: Environment name (None for global)
        length: Length parameter for algorithms that need it
        params: Additional algorithm parameters

    Returns:
        bool: True if successful, False otherwise
    """
    result = core_mask.add_mask(
        column=column,
        algorithm=algorithm,
        environment=environment,
        length=length,
        params=params
    )
    return result.success

def remove_mask(column: str, environment: Optional[str] = None) -> bool:
    """
    Remove a masking rule for a column.

    Args:
        column: Column name
        environment: Environment name (None for global)

    Returns:
        bool: True if successful, False otherwise
    """
    result = core_mask.remove_mask(column=column, environment=environment)
    return result.success

def list_masks(environment: Optional[str] = None) -> Dict[str, Any]:
    """
    List masking rules.

    Args:
        environment: Environment name (None for all)

    Returns:
        Dictionary with masking rules
    """
    result = core_mask.list_masks(environment=environment)
    return result.data if result.success else {}

def test_mask(
    column: str,
    value: str,
    environment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test a masking rule on a value.

    Args:
        column: Column name
        value: Value to mask
        environment: Environment name

    Returns:
        Dictionary with original and masked values
    """
    result = core_mask.test_mask(column=column, value=value, environment=environment)
    return result.data if result.success else {}

# Data Generation Functions

def generate_data(
    num_records: int = 10,
    columns: Optional[List[str]] = None,
    environment: Optional[str] = None,
    table: Optional[str] = None,
    string_length: int = 10,
    pattern: Optional[Dict[str, str]] = None,
    min_number: float = 0,
    max_number: float = 100,
    decimal_places: int = 2,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_format: str = '%Y-%m-%d'
) -> pd.DataFrame:
    """
    Generate random data for testing.

    Args:
        num_records: Number of records to generate
        columns: List of column names
        environment: Environment name to get table schema from
        table: Table name to get schema from
        string_length: Default length for string values
        pattern: Dictionary of column patterns
        min_number: Minimum value for numeric columns
        max_number: Maximum value for numeric columns
        decimal_places: Number of decimal places for numeric columns
        start_date: Start date for date columns
        end_date: End date for date columns
        date_format: Date format for date columns

    Returns:
        DataFrame with generated data
    """
    result = core_gen.generate_data(
        num_records=num_records,
        columns=columns,
        environment=environment,
        table=table,
        string_length=string_length,
        pattern=pattern,
        min_number=min_number,
        max_number=max_number,
        decimal_places=decimal_places,
        start_date=start_date,
        end_date=end_date,
        date_format=date_format
    )

    if result.success and result.data:
        return pd.DataFrame(result.data)
    else:
        return pd.DataFrame()

def generate_and_save(
    num_records: int = 10,
    columns: Optional[List[str]] = None,
    environment: Optional[str] = None,
    table: Optional[str] = None,
    output_file: Optional[str] = None,
    file_format: str = 'csv',
    write_to_db: bool = False,
    mode: str = 'APPEND',
    encryption_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate random data and save it to a file or database.

    Args:
        num_records: Number of records to generate
        columns: List of column names
        environment: Environment name to get table schema from
        table: Table name to get schema from
        output_file: Output file path
        file_format: Output file format (csv, json)
        write_to_db: Whether to write to database
        mode: Write mode (APPEND, REPLACE, FAIL)
        encryption_key: Encryption key for encrypted environments
        **kwargs: Additional parameters for generate_data

    Returns:
        Dictionary with operation results
    """
    result = core_gen.generate_and_save(
        num_records=num_records,
        columns=columns,
        environment=environment,
        table=table,
        output_file=output_file,
        file_format=file_format,
        write_to_db=write_to_db,
        mode=mode,
        encryption_key=encryption_key,
        **kwargs
    )
    return result.to_dict()


# Configuration Management Functions

def get_config() -> Dict[str, Any]:
    """
    Get current configuration.

    Returns:
        Dictionary with current configuration values
    """
    result = core_config.get_config()
    return result.data if result.success else {}


def set_config(key: str, value: Any) -> bool:
    """
    Set a configuration value.

    Args:
        key: Configuration key
        value: Configuration value

    Returns:
        bool: True if successful, False otherwise
    """
    result = core_config.set_config(key, value)
    return result.success


def reset_config() -> bool:
    """
    Reset configuration to defaults.

    Returns:
        bool: True if successful, False otherwise
    """
    result = core_config.reset_config()
    return result.success


def get_config_info() -> Dict[str, Any]:
    """
    Get configuration information including file paths.

    Returns:
        Dictionary with configuration and path information
    """
    result = core_config.show_config_info()
    return result.data if result.success else {}
