"""
ELM Tool Core Data Copy Operations

Unified data copy operations for both CLI and API interfaces.
This module provides consistent data copying functionality between databases and files.
"""

import os
import json
import time
from datetime import datetime
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
    validate_required_params, convert_sqlalchemy_mode, safe_print
)
from elm.core.environment import get_connection_url, _initialize_oracle_client, _handle_oracle_connection_error
from elm.core.masking import apply_masking
from elm.core.streaming import write_to_db_streaming


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
    def _create_batched_generator(eng):
        """Helper function to create a batched query generator."""
        def batched_query_generator():
            try:
                with eng.connect() as connection:
                    result = pd.read_sql_query(query, connection, chunksize=batch_size)
                    for batch in result:
                        if apply_masks:
                            yield apply_masking(batch, environment)
                        else:
                            yield batch
            except Exception as e:
                # Handle Oracle errors that occur during iteration
                if 'oracle' in connection_url.lower():
                    if _handle_oracle_connection_error(connection_url, e):
                        # Retry with a new engine
                        retry_engine = create_engine(connection_url)
                        with retry_engine.connect() as connection:
                            result = pd.read_sql_query(query, connection, chunksize=batch_size)
                            for batch in result:
                                if apply_masks:
                                    yield apply_masking(batch, environment)
                                else:
                                    yield batch
                    else:
                        # Re-raise the original error
                        if isinstance(e, SQLAlchemyError):
                            raise DatabaseError(f"Database error: {str(e)}")
                        else:
                            raise CopyError(f"Error executing query: {str(e)}")
                else:
                    # Not an Oracle connection, re-raise the error
                    if isinstance(e, SQLAlchemyError):
                        raise DatabaseError(f"Database error: {str(e)}")
                    else:
                        raise CopyError(f"Error executing query: {str(e)}")

        return batched_query_generator()

    try:
        engine = create_engine(connection_url)

        if batch_size:
            # For batched queries, return a generator that handles errors internally
            return _create_batched_generator(engine)
        else:
            # For non-batched queries, we can use a simple context manager
            with engine.connect() as connection:
                result = pd.read_sql_query(query, connection)

                # Apply masking if requested
                if apply_masks:
                    result = apply_masking(result, environment)

                return result
    except Exception as e:
        # Check if this is an Oracle connection and try to handle thin mode errors
        if 'oracle' in connection_url.lower():
            if _handle_oracle_connection_error(connection_url, e):
                # Retry the query after Oracle client initialization
                engine = create_engine(connection_url)

                if batch_size:
                    # For batched queries, return a generator
                    return _create_batched_generator(engine)
                else:
                    with engine.connect() as connection:
                        result = pd.read_sql_query(query, connection)
                        if apply_masks:
                            result = apply_masking(result, environment)
                        return result
            else:
                # Re-raise the original error
                if isinstance(e, SQLAlchemyError):
                    raise DatabaseError(f"Database error: {str(e)}")
                else:
                    raise CopyError(f"Error executing query: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            if isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database error: {str(e)}")
            else:
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
            # For CSV, handle append mode specially with UTF-8 encoding
            if mode == 'a' and os.path.exists(file_path):
                # Append without header
                data.to_csv(file_path, mode='a', header=False, index=False, encoding='utf-8')
            else:
                # Write with header
                data.to_csv(file_path, index=False, encoding='utf-8')
        elif file_format == FileFormat.JSON:
            if mode == 'a' and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                # Append to existing JSON with UTF-8 encoding
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)

                    # Convert DataFrame to list of dicts
                    new_records = data.to_dict('records')

                    # Append new records
                    if isinstance(existing_data, list):
                        existing_data.extend(new_records)
                    else:
                        existing_data = [existing_data] + new_records

                    # Write back with UTF-8 encoding and ensure_ascii=False to preserve Unicode
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    raise FileError(f"Error appending to JSON: {str(e)}")
            else:
                # Write new JSON file with UTF-8 encoding and ensure_ascii=False
                with open(file_path, 'w', encoding='utf-8') as f:
                    data.to_json(f, orient='records', indent=2, force_ascii=False)
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
            # Read CSV with UTF-8 encoding
            return pd.read_csv(file_path, encoding='utf-8')
        elif file_format == FileFormat.JSON:
            # Read JSON with UTF-8 encoding
            return pd.read_json(file_path, orient='records', encoding='utf-8')
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
    batch_size: Optional[int] = None,
    use_streaming: bool = True
) -> int:
    """
    Write data to a database table using optimized streaming methods.

    Args:
        data: DataFrame to write
        connection_url: Database connection URL
        table_name: Target table name
        mode: Write mode (APPEND, REPLACE, FAIL)
        batch_size: Batch size for large datasets
        use_streaming: Whether to use optimized streaming (default: True)

    Returns:
        Number of records written
    """
    # Use optimized streaming method
    return write_to_db_streaming(
        data=data,
        connection_url=connection_url,
        table_name=table_name,
        mode=mode,
        batch_size=batch_size,
        use_optimized=use_streaming
    )


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

        # For MSSQL, use a direct SQL query to avoid NVARCHAR(max) issues with old drivers
        if 'mssql' in connection_url.lower():
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = :table_name"),
                    {"table_name": table_name}
                )
                count = result.scalar()
                return count > 0
        else:
            # Use inspector for other databases
            inspector = inspect(engine)
            return inspector.has_table(table_name)
    except Exception as e:
        # Check if this is an Oracle connection and try to handle thin mode errors
        if 'oracle' in connection_url.lower():
            if _handle_oracle_connection_error(connection_url, e):
                # Retry after Oracle client initialization
                engine = create_engine(connection_url)
                inspector = inspect(engine)
                return inspector.has_table(table_name)
            else:
                # Re-raise the original error
                if isinstance(e, SQLAlchemyError):
                    raise DatabaseError(f"Database error while checking table existence: {str(e)}")
                else:
                    raise CopyError(f"Error checking table existence: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            if isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database error while checking table existence: {str(e)}")
            else:
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

        # For MSSQL, use a direct SQL query to avoid NVARCHAR(max) issues with old drivers
        if 'mssql' in connection_url.lower():
            from sqlalchemy import text

            # First check if table exists
            if not check_table_exists(connection_url, table_name):
                return None

            # Get columns using direct SQL
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = :table_name ORDER BY ORDINAL_POSITION"),
                    {"table_name": table_name}
                )
                columns = [row[0].lower() for row in result]
                return columns
        else:
            # Use inspector for other databases
            inspector = inspect(engine)
            if not inspector.has_table(table_name):
                return None
            columns = inspector.get_columns(table_name)
            return [column['name'].lower() for column in columns]
    except Exception as e:
        # Check if this is an Oracle connection and try to handle thin mode errors
        if 'oracle' in connection_url.lower():
            if _handle_oracle_connection_error(connection_url, e):
                # Retry after Oracle client initialization
                engine = create_engine(connection_url)
                inspector = inspect(engine)
                if not inspector.has_table(table_name):
                    return None
                columns = inspector.get_columns(table_name)
                return [column['name'].lower() for column in columns]
            else:
                # Re-raise the original error
                if isinstance(e, SQLAlchemyError):
                    raise DatabaseError(f"Database error while getting table columns: {str(e)}")
                else:
                    raise CopyError(f"Error getting table columns: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            if isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database error while getting table columns: {str(e)}")
            else:
                raise CopyError(f"Error getting table columns: {str(e)}")


def _get_oracle_dtype_mapping(source_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create Oracle-specific type mapping for DataFrame columns.

    Oracle requires special handling for FLOAT types to avoid binary precision errors.

    Args:
        source_data: Source DataFrame

    Returns:
        Dictionary mapping column names to SQLAlchemy types
    """
    from sqlalchemy.types import Float, Integer, String, DateTime, Boolean, Numeric
    from sqlalchemy.dialects import oracle

    dtype_mapping = {}

    for col in source_data.columns:
        dtype = source_data[col].dtype

        # Map pandas dtypes to Oracle-compatible SQLAlchemy types
        if dtype == 'float64' or dtype == 'float32':
            # Use Oracle FLOAT with binary_precision to avoid conversion errors
            dtype_mapping[col] = oracle.FLOAT(binary_precision=126)
        elif dtype == 'int64' or dtype == 'int32':
            dtype_mapping[col] = Integer()
        elif dtype == 'bool':
            dtype_mapping[col] = Integer()  # Oracle doesn't have native BOOLEAN
        elif dtype == 'datetime64[ns]':
            dtype_mapping[col] = DateTime()
        elif str(dtype).startswith('decimal'):
            dtype_mapping[col] = Numeric()
        else:
            # Default to VARCHAR for string and other types
            dtype_mapping[col] = String(4000)

    return dtype_mapping


def _create_mssql_table_direct(engine, table_name: str, source_data: pd.DataFrame) -> None:
    """
    Create MSSQL table using direct SQL to avoid NVARCHAR(max) issues with old drivers.

    Args:
        engine: SQLAlchemy engine
        table_name: Table name to create
        source_data: Source DataFrame for schema inference
    """
    from sqlalchemy import text

    # Map pandas dtypes to SQL Server types
    column_defs = []
    for col in source_data.columns:
        dtype = source_data[col].dtype

        if dtype == 'float64' or dtype == 'float32':
            sql_type = "FLOAT"
        elif dtype == 'int64' or dtype == 'int32':
            sql_type = "INT"
        elif dtype == 'bool':
            sql_type = "BIT"
        elif dtype == 'datetime64[ns]':
            sql_type = "DATETIME"
        else:
            # Default to NVARCHAR for string and other types
            sql_type = "NVARCHAR(4000)"

        column_defs.append(f"[{col}] {sql_type}")

    # Create table SQL
    create_sql = f"CREATE TABLE [{table_name}] ({', '.join(column_defs)})"

    # Execute the CREATE TABLE statement
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()


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

                # Use database-specific table creation methods
                if 'oracle' in target_url.lower():
                    # Use Oracle-specific type mapping
                    dtype_mapping = _get_oracle_dtype_mapping(source_data)
                    source_data.head(0).to_sql(table_name, engine, if_exists='fail', index=False, dtype=dtype_mapping)
                elif 'mssql' in target_url.lower():
                    # Use direct SQL for MSSQL to avoid NVARCHAR(max) issues with old drivers
                    _create_mssql_table_direct(engine, table_name, source_data)
                else:
                    # Use pandas to_sql for other databases
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
    """Process items in parallel using the provided function.

    This helper preserves the order of ``items`` in the returned list and
    propagates the first exception raised by any worker *after* all futures
    have completed. This makes it safe to use in batch-processing flows where
    per-item results (e.g., metrics) are required and failures must not be
    silently ignored.

    Args:
        func: Function to apply to each item.
        items: List of items to process.
        max_workers: Maximum number of parallel workers.

    Returns:
        List of results in the same order as ``items``.
    """
    if not items:
        return []

    results: List[Any] = [None] * len(items)
    first_exc: Optional[BaseException] = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(func, item): index for index, item in enumerate(items)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except BaseException as exc:  # noqa: BLE001 - propagate first failure
                if first_exc is None:
                    first_exc = exc

    if first_exc is not None:
        # Re-raise the first captured exception after allowing all
        # futures to complete, so that resources are cleaned up.
        raise first_exc

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
        safe_print(f"📂 Reading data from file: {file_path}")
        data = read_from_file(file_path, format_enum)
        safe_print(f"✓ Loaded {len(data):,} records from file")

        # Apply masking if needed
        if apply_masks:
            data = apply_masking(data, target_env)

        # Validate target table if requested
        if validate_target:
            validate_target_table(data, connection_url, table, create_if_not_exists)

        # Write to database
        safe_print(f"📊 Writing data to table '{table}'...")
        total_records = write_to_db(data, connection_url, table, mode_enum, batch_size)
        safe_print(f"✓ Successfully wrote {total_records:,} records")

        message = f"Successfully copied {total_records} records to table '{table}'"
        if apply_masks:
            message += " (with masking applied)"

        return create_success_result(message, record_count=total_records)

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

        # Handle table creation if needed (before validation or data copy)
        if create_if_not_exists or validate_target:
            # Create database-specific sample query
            if 'mssql' in source_url.lower():
                # MSSQL uses TOP instead of LIMIT
                # Need to inject TOP after SELECT
                if query.strip().upper().startswith('SELECT'):
                    # Find the position after SELECT
                    select_pos = query.upper().find('SELECT') + 6
                    sample_query = query[:select_pos] + ' TOP 1' + query[select_pos:]
                else:
                    sample_query = query  # If not a SELECT, use as-is
            elif 'oracle' in source_url.lower():
                # Oracle uses ROWNUM
                sample_query = f"SELECT * FROM ({query}) WHERE ROWNUM <= 1"
            else:
                # PostgreSQL, MySQL, SQLite use LIMIT
                sample_query = f"{query} LIMIT 1"

            try:
                sample_data = execute_query(source_url, sample_query, None, None, False)  # Don't mask validation data
                if len(sample_data) > 0:
                    validate_target_table(sample_data, target_url, table, create_if_not_exists)
            except Exception as e:
                raise CopyError(f"Error during validation: {str(e)}")

        # Execute query and write to target
        if batch_size:
            # Handle batched results with progress reporting
            result_iter = execute_query(
                source_url,
                query,
                batch_size,
                target_env if apply_masks else None,
                apply_masks,
            )

            total_records = 0
            batch_number = 0

            safe_print(
                f"📊 Starting batch copy (batch size: {batch_size}, "
                f"parallel_workers: {parallel_workers})..."
            )

            if not parallel_workers or parallel_workers <= 1:
                # Sequential path (no parallelism requested)
                for chunk in result_iter:
                    batch_number += 1
                    current_mode = mode_enum if batch_number == 1 else WriteMode.APPEND

                    start_ts = datetime.now()
                    start_perf = time.perf_counter()
                    records_written = write_to_db(chunk, target_url, table, mode=current_mode)
                    duration = time.perf_counter() - start_perf
                    end_ts = datetime.now()

                    total_records += records_written

                    safe_print(
                        f"✓ Batch {batch_number}: Copied {records_written:,} records "
                        f"| Total: {total_records:,} records "
                        f"| Duration: {duration:.2f}s "
                        f"| Start: {start_ts.isoformat(timespec='seconds')} "
                        f"| End: {end_ts.isoformat(timespec='seconds')}"
                    )
            else:
                # Parallel path for batches after the first one
                try:
                    first_chunk = next(result_iter)
                except StopIteration:
                    first_chunk = None

                if first_chunk is not None:
                    batch_number = 1
                    start_ts = datetime.now()
                    start_perf = time.perf_counter()
                    records_written = write_to_db(first_chunk, target_url, table, mode=mode_enum)
                    duration = time.perf_counter() - start_perf
                    end_ts = datetime.now()

                    total_records += records_written

                    safe_print(
                        f"✓ Batch {batch_number}: Copied {records_written:,} records "
                        f"| Total: {total_records:,} records "
                        f"| Duration: {duration:.2f}s "
                        f"| Start: {start_ts.isoformat(timespec='seconds')} "
                        f"| End: {end_ts.isoformat(timespec='seconds')}"
                    )

                pending_batches: List[Dict[str, Any]] = []

                def _write_single_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
                    """Worker function to write a single batch in APPEND mode.

                    Returns:
                        Dict with batch_number, records_written, start_ts,
                        end_ts and duration for logging/metrics.
                    """
                    batch_no = payload["batch_number"]
                    chunk_df = payload["data"]

                    start = datetime.now()
                    start_perf_local = time.perf_counter()
                    written = write_to_db(chunk_df, target_url, table, mode=WriteMode.APPEND)
                    duration_local = time.perf_counter() - start_perf_local
                    end = datetime.now()

                    return {
                        "batch_number": batch_no,
                        "records_written": written,
                        "start_ts": start,
                        "end_ts": end,
                        "duration": duration_local,
                    }

                def _flush_pending_batches(batches: List[Dict[str, Any]]) -> None:
                    nonlocal total_records
                    if not batches:
                        return

                    batch_results = process_in_parallel(
                        _write_single_batch,
                        batches,
                        max_workers=parallel_workers,
                    )

                    for batch_result in sorted(batch_results, key=lambda r: r["batch_number"]):
                        total_records += batch_result["records_written"]
                        safe_print(
                            f"✓ Batch {batch_result['batch_number']}: Copied {batch_result['records_written']:,} records "
                            f"| Total: {total_records:,} records "
                            f"| Duration: {batch_result['duration']:.2f}s "
                            f"| Start: {batch_result['start_ts'].isoformat(timespec='seconds')} "
                            f"| End: {batch_result['end_ts'].isoformat(timespec='seconds')}"
                        )

                for chunk in result_iter:
                    batch_number += 1
                    pending_batches.append(
                        {
                            "batch_number": batch_number,
                            "data": chunk,
                        }
                    )

                    if len(pending_batches) >= parallel_workers:
                        _flush_pending_batches(pending_batches)
                        pending_batches = []

                if pending_batches:
                    _flush_pending_batches(pending_batches)
        else:
            # Handle single result
            safe_print("📊 Copying data in single batch...")
            result = execute_query(
                source_url,
                query,
                None,
                target_env if apply_masks else None,
                apply_masks,
            )

            start_ts = datetime.now()
            start_perf = time.perf_counter()
            total_records = write_to_db(result, target_url, table, mode=mode_enum)
            duration = time.perf_counter() - start_perf
            end_ts = datetime.now()

            safe_print(
                f"✓ Copied {total_records:,} records "
                f"| Duration: {duration:.2f}s "
                f"| Start: {start_ts.isoformat(timespec='seconds')} "
                f"| End: {end_ts.isoformat(timespec='seconds')}"
            )

        message = f"Successfully copied {total_records} records to table '{table}'"
        if apply_masks:
            message += " (with masking applied)"

        return create_success_result(message, record_count=total_records)

    except Exception as e:
        return handle_exception(e, "database to database copy")
