"""
ELM Tool Core Data Copy Operations

Unified data copy operations for both CLI and API interfaces.
This module provides consistent data copying functionality between databases and files.
"""

import os
import json
import pandas as pd
import concurrent.futures
from typing import Optional, List, Dict, Any, Union, Iterator
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError

from elm.core.types import CopyConfig, WriteMode, FileFormat, OperationResult
from elm.core.exceptions import CopyError, ValidationError, DatabaseError, FileError
from elm.core.utils import (
    validate_write_mode, validate_file_format, ensure_directory_exists,
    create_success_result, create_error_result, handle_exception, 
    validate_required_params, convert_sqlalchemy_mode
)
from elm.core.environment import get_connection_url
from elm.core.masking import apply_masking


def execute_query(
    connection_url: str,
    query: str,
    batch_size: Optional[int] = None,
    environment: Optional[str] = None,
    apply_masks: bool = True
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Execute a query and return the results.
    
    Args:
        connection_url: Database connection URL
        query: SQL query to execute
        batch_size: Batch size for chunked processing
        environment: Environment name for masking
        apply_masks: Whether to apply masking rules
    
    Returns:
        DataFrame or iterator of DataFrames (if batched)
    """
    try:
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            if batch_size:
                # Execute with batching
                result = pd.read_sql_query(query, connection, chunksize=batch_size)
                
                if not apply_masks:
                    return result  # Return iterator as-is
                
                # Create a generator that applies masking to each batch
                def masked_batches():
                    for batch in result:
                        yield apply_masking(batch, environment)
                
                return masked_batches()
            else:
                # Execute without batching
                result = pd.read_sql_query(query, connection)
                
                # Apply masking if requested
                if apply_masks:
                    result = apply_masking(result, environment)
                
                return result
    except SQLAlchemyError as e:
        raise DatabaseError(f"Database error: {str(e)}")
    except Exception as e:
        raise CopyError(f"Error executing query: {str(e)}")


def write_to_file(
    data: pd.DataFrame, 
    file_path: str, 
    file_format: FileFormat = FileFormat.CSV,
    mode: str = 'w'
) -> None:
    """
    Write data to a file in the specified format.
    
    Args:
        data: DataFrame to write
        file_path: Output file path
        file_format: File format (CSV or JSON)
        mode: Write mode ('w' for write, 'a' for append)
    """
    try:
        # Create directory if it doesn't exist
        ensure_directory_exists(file_path)
        
        if file_format == FileFormat.CSV:
            # For CSV, handle append mode specially
            if mode == 'a' and os.path.exists(file_path):
                # Append without header
                data.to_csv(file_path, mode='a', header=False, index=False)
            else:
                # Write with header
                data.to_csv(file_path, index=False)
        elif file_format == FileFormat.JSON:
            if mode == 'a' and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                # Append to existing JSON
                try:
                    with open(file_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Convert DataFrame to list of dicts
                    new_records = data.to_dict('records')
                    
                    # Append new records
                    if isinstance(existing_data, list):
                        existing_data.extend(new_records)
                    else:
                        existing_data = [existing_data] + new_records
                    
                    # Write back
                    with open(file_path, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                except Exception as e:
                    raise FileError(f"Error appending to JSON: {str(e)}")
            else:
                # Write new JSON file
                data.to_json(file_path, orient='records', indent=2)
        else:
            raise ValidationError(f"Unsupported file format: {file_format}")
            
    except Exception as e:
        if isinstance(e, (FileError, ValidationError)):
            raise
        raise FileError(f"Error writing to file: {str(e)}")


def read_from_file(file_path: str, file_format: FileFormat = FileFormat.CSV) -> pd.DataFrame:
    """
    Read data from a file in the specified format.
    
    Args:
        file_path: Input file path
        file_format: File format (CSV or JSON)
    
    Returns:
        DataFrame with file contents
    """
    if not os.path.exists(file_path):
        raise FileError(f"File not found: {file_path}")
    
    try:
        if file_format == FileFormat.CSV:
            return pd.read_csv(file_path)
        elif file_format == FileFormat.JSON:
            return pd.read_json(file_path, orient='records')
        else:
            raise ValidationError(f"Unsupported file format: {file_format}")
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise FileError(f"Error reading file: {str(e)}")


def write_to_db(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND,
    batch_size: Optional[int] = None
) -> None:
    """
    Write data to a database table.
    
    Args:
        data: DataFrame to write
        connection_url: Database connection URL
        table_name: Target table name
        mode: Write mode (APPEND, REPLACE, FAIL)
        batch_size: Batch size for large datasets
    """
    try:
        engine = create_engine(connection_url)
        if_exists = convert_sqlalchemy_mode(mode)
        
        if batch_size and len(data) > batch_size:
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                current_if_exists = if_exists if i == 0 else 'append'
                batch.to_sql(table_name, engine, if_exists=current_if_exists, index=False)
        else:
            # Process all at once
            data.to_sql(table_name, engine, if_exists=if_exists, index=False)
            
    except SQLAlchemyError as e:
        raise DatabaseError(f"Database error: {str(e)}")
    except Exception as e:
        raise CopyError(f"Error writing to database: {str(e)}")


def check_table_exists(connection_url: str, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        connection_url: Database connection URL
        table_name: Table name to check
    
    Returns:
        True if table exists, False otherwise
    """
    try:
        engine = create_engine(connection_url)
        inspector = inspect(engine)
        return inspector.has_table(table_name)
    except SQLAlchemyError as e:
        raise DatabaseError(f"Database error while checking table existence: {str(e)}")
    except Exception as e:
        raise CopyError(f"Error checking table existence: {str(e)}")


def get_table_columns(connection_url: str, table_name: str) -> Optional[List[str]]:
    """
    Get the column names of a table.
    
    Args:
        connection_url: Database connection URL
        table_name: Table name
    
    Returns:
        List of column names or None if table doesn't exist
    """
    try:
        engine = create_engine(connection_url)
        inspector = inspect(engine)
        if not inspector.has_table(table_name):
            return None
        columns = inspector.get_columns(table_name)
        return [column['name'].lower() for column in columns]
    except SQLAlchemyError as e:
        raise DatabaseError(f"Database error while getting table columns: {str(e)}")
    except Exception as e:
        raise CopyError(f"Error getting table columns: {str(e)}")


def validate_target_table(
    source_data: pd.DataFrame,
    target_url: str,
    table_name: str,
    create_if_not_exists: bool = False
) -> None:
    """
    Validate that the target table exists and has all required columns.
    
    Args:
        source_data: Source DataFrame for column validation
        target_url: Target database connection URL
        table_name: Target table name
        create_if_not_exists: Whether to create table if it doesn't exist
    """
    # Check if table exists
    if not check_table_exists(target_url, table_name):
        if create_if_not_exists:
            # Create the table based on source data
            try:
                engine = create_engine(target_url)
                source_data.head(0).to_sql(table_name, engine, if_exists='fail', index=False)
                return
            except Exception as e:
                raise CopyError(f"Failed to create table {table_name}: {str(e)}")
        else:
            raise CopyError(f"Target table {table_name} does not exist. Use create_if_not_exists=True to create it.")
    
    # Check if all source columns exist in target
    source_columns = set(source_data.columns.str.lower())
    target_columns = set(get_table_columns(target_url, table_name) or [])
    
    if not target_columns:
        raise CopyError(f"Could not retrieve columns for target table {table_name}")
    
    missing_columns = source_columns - target_columns
    if missing_columns:
        raise CopyError(f"Target table {table_name} is missing columns: {', '.join(missing_columns)}")


def process_in_parallel(func, items: List[Any], max_workers: int) -> List[Any]:
    """
    Process items in parallel using the provided function.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        max_workers: Maximum number of parallel workers
    
    Returns:
        List of results
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Log error but continue processing other items
                results.append(None)
    return results


def copy_db_to_file(
    source_env: str,
    query: str,
    file_path: str,
    file_format: str = 'csv',
    mode: str = 'REPLACE',
    batch_size: Optional[int] = None,
    parallel_workers: int = 1,
    source_encryption_key: Optional[str] = None,
    apply_masks: bool = True
) -> OperationResult:
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

    Returns:
        OperationResult with operation status and details
    """
    try:
        validate_required_params(
            {'source_env': source_env, 'query': query, 'file_path': file_path},
            ['source_env', 'query', 'file_path']
        )

        # Validate and convert parameters
        format_enum = validate_file_format(file_format)
        mode_enum = validate_write_mode(mode)

        # Get connection URL
        connection_url = get_connection_url(source_env, source_encryption_key)

        # Set file mode
        file_mode = 'w' if mode_enum == WriteMode.REPLACE else 'a'

        # Execute query and write to file
        if batch_size:
            # Handle batched results
            result = execute_query(connection_url, query, batch_size, source_env, apply_masks)

            first_batch = True
            total_records = 0
            for chunk in result:
                current_mode = file_mode if first_batch else 'a'
                write_to_file(chunk, file_path, format_enum, current_mode)
                total_records += len(chunk)
                first_batch = False
        else:
            # Handle single result
            result = execute_query(connection_url, query, None, source_env, apply_masks)
            write_to_file(result, file_path, format_enum, file_mode)
            total_records = len(result)

        message = f"Successfully copied {total_records} records to {file_path}"
        if apply_masks:
            message += " (with masking applied)"

        return create_success_result(message, record_count=total_records)

    except Exception as e:
        return handle_exception(e, "database to file copy")


def copy_file_to_db(
    file_path: str,
    target_env: str,
    table: str,
    file_format: str = 'csv',
    mode: str = 'APPEND',
    batch_size: Optional[int] = None,
    parallel_workers: int = 1,
    target_encryption_key: Optional[str] = None,
    validate_target: bool = False,
    create_if_not_exists: bool = False,
    apply_masks: bool = True
) -> OperationResult:
    """
    Copy data from file to database.

    Args:
        file_path: Input file path
        target_env: Target environment name
        table: Target table name
        file_format: Input file format (csv, json)
        mode: Write mode (APPEND, REPLACE, FAIL)
        batch_size: Batch size for processing large datasets
        parallel_workers: Number of parallel workers
        target_encryption_key: Encryption key for target environment
        validate_target: Whether to validate target table
        create_if_not_exists: Whether to create target table if it doesn't exist
        apply_masks: Whether to apply masking rules

    Returns:
        OperationResult with operation status and details
    """
    try:
        validate_required_params(
            {'file_path': file_path, 'target_env': target_env, 'table': table},
            ['file_path', 'target_env', 'table']
        )

        # Validate and convert parameters
        format_enum = validate_file_format(file_format)
        mode_enum = validate_write_mode(mode)

        # Get connection URL
        connection_url = get_connection_url(target_env, target_encryption_key)

        # Read data from file
        data = read_from_file(file_path, format_enum)

        # Apply masking if needed
        if apply_masks:
            data = apply_masking(data, target_env)

        # Validate target table if requested
        if validate_target:
            validate_target_table(data, connection_url, table, create_if_not_exists)

        # Write to database
        write_to_db(data, connection_url, table, mode_enum, batch_size)

        message = f"Successfully copied {len(data)} records to table '{table}'"
        if apply_masks:
            message += " (with masking applied)"

        return create_success_result(message, record_count=len(data))

    except Exception as e:
        return handle_exception(e, "file to database copy")


def copy_db_to_db(
    source_env: str,
    target_env: str,
    query: str,
    table: str,
    mode: str = 'APPEND',
    batch_size: Optional[int] = None,
    parallel_workers: int = 1,
    source_encryption_key: Optional[str] = None,
    target_encryption_key: Optional[str] = None,
    validate_target: bool = False,
    create_if_not_exists: bool = False,
    apply_masks: bool = True
) -> OperationResult:
    """
    Copy data from database to database.

    Args:
        source_env: Source environment name
        target_env: Target environment name
        query: SQL query to execute on source
        table: Target table name
        mode: Write mode (APPEND, REPLACE, FAIL)
        batch_size: Batch size for processing large datasets
        parallel_workers: Number of parallel workers
        source_encryption_key: Encryption key for source environment
        target_encryption_key: Encryption key for target environment
        validate_target: Whether to validate target table
        create_if_not_exists: Whether to create target table if it doesn't exist
        apply_masks: Whether to apply masking rules

    Returns:
        OperationResult with operation status and details
    """
    try:
        validate_required_params(
            {'source_env': source_env, 'target_env': target_env, 'query': query, 'table': table},
            ['source_env', 'target_env', 'query', 'table']
        )

        # Validate and convert parameters
        mode_enum = validate_write_mode(mode)

        # Get connection URLs
        source_url = get_connection_url(source_env, source_encryption_key)
        target_url = get_connection_url(target_env, target_encryption_key)

        # For validation, get a sample of the data first
        if validate_target:
            sample_query = f"{query} LIMIT 1"
            try:
                sample_data = execute_query(source_url, sample_query, None, None, False)  # Don't mask validation data
                if len(sample_data) > 0:
                    validate_target_table(sample_data, target_url, table, create_if_not_exists)
            except Exception as e:
                raise CopyError(f"Error during validation: {str(e)}")

        # Execute query and write to target
        if batch_size:
            # Handle batched results
            result = execute_query(source_url, query, batch_size, target_env if apply_masks else None, apply_masks)

            first_batch = True
            total_records = 0
            for chunk in result:
                current_mode = mode_enum if first_batch else WriteMode.APPEND
                write_to_db(chunk, target_url, table, current_mode)
                total_records += len(chunk)
                first_batch = False
        else:
            # Handle single result
            result = execute_query(source_url, query, None, target_env if apply_masks else None, apply_masks)
            write_to_db(result, target_url, table, mode_enum)
            total_records = len(result)

        message = f"Successfully copied {total_records} records to table '{table}'"
        if apply_masks:
            message += " (with masking applied)"

        return create_success_result(message, record_count=total_records)

    except Exception as e:
        return handle_exception(e, "database to database copy")
