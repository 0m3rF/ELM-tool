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
from elm.core.masking import apply_masking, _get_masking_definitions_cached
from elm.core.streaming import write_to_db_streaming, detect_database_type


def execute_query(
    connection_url: str,
    query: str,
    batch_size: Optional[int] = None,
    environment: Optional[str] = None,
    apply_masks: bool = True
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Execute a query and return the results.

    For Oracle URLs we proactively attempt to initialise the Oracle
    client in *thick* mode via :func:`_initialize_oracle_client` before
    creating any SQLAlchemy engines. This avoids the python-oracledb
    limitation where thick mode cannot be enabled after a thin
    connection has already been created (DPY-2019) and helps prevent
    DPY-3015 password verifier issues.

    Args:
        connection_url: Database connection URL
        query: SQL query to execute
        batch_size: Batch size for chunked processing
        environment: Environment name for masking
        apply_masks: Whether to apply masking rules

    Returns:
        DataFrame or iterator of DataFrames (if batched)
    """
    # For Oracle connections, try to activate thick mode *before* any
    # engine/connection is created so that subsequent operations use the
    # most compatible mode available. The helper handles and logs all
    # failures internally, so we intentionally ignore its return value
    # and proceed regardless.
    if 'oracle' in connection_url.lower():
        _initialize_oracle_client()
    # Pre-load masking definitions once per execute_query call when masking is
    # enabled. This avoids re-resolving the definitions for every batch while
    # still benefiting from the cached, mtime-aware loader in the masking
    # module. If masking is disabled, we never touch the masking layer here.
    definitions = None
    if apply_masks:
        definitions = _get_masking_definitions_cached()

    def _create_batched_generator(eng):
        """Helper function to create a batched query generator."""
        def batched_query_generator():
            try:
                with eng.connect() as connection:
                    result = pd.read_sql_query(query, connection, chunksize=batch_size)
                    for batch in result:
                        if apply_masks:
                            # Pass pre-loaded definitions into apply_masking so
                            # they are reused across all batches.
                            yield apply_masking(batch, environment, definitions=definitions)
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
                                    yield apply_masking(batch, environment, definitions=definitions)
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
                    result = apply_masking(result, environment, definitions=definitions)

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
                            result = apply_masking(result, environment, definitions=definitions)
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


def get_column_mapping(
    source_data: pd.DataFrame,
    target_url: str,
    table_name: str
) -> Optional[List[str]]:
    """
    Get the mapping of columns that exist in both source data and target table.

    This function returns the list of column names (in source DataFrame case)
    that should be used when copying data. Only columns that exist in BOTH
    the source query result set AND the target table schema are included.

    Args:
        source_data: Source DataFrame with column names from query result
        target_url: Target database connection URL
        table_name: Target table name

    Returns:
        List of column names to use for data copy, or None if table doesn't exist.
        Columns are returned in the order they appear in the source DataFrame.
    """
    # Check if table exists
    if not check_table_exists(target_url, table_name):
        return None

    # Get target table columns (lowercased)
    target_columns_lower = set(get_table_columns(target_url, table_name) or [])

    if not target_columns_lower:
        return None

    # Get source columns and find intersection (preserve source column order and case)
    # Map each source column to its lowercased version for comparison
    mapped_columns = []
    for col in source_data.columns:
        if col.lower() in target_columns_lower:
            mapped_columns.append(col)

    return mapped_columns if mapped_columns else None


def validate_target_table(
    source_data: pd.DataFrame,
    target_url: str,
    table_name: str,
    create_if_not_exists: bool = False
) -> Optional[List[str]]:
    """
    Validate that the target table exists and determine column mapping.

    When the target table exists, this function returns the list of columns
    that exist in BOTH the source data AND the target table. Extra columns
    in the source that don't exist in the target are silently excluded.

    Args:
        source_data: Source DataFrame for column validation
        target_url: Target database connection URL
        table_name: Target table name
        create_if_not_exists: Whether to create table if it doesn't exist

    Returns:
        List of column names to use for data copy (columns in both source and target),
        or None if table was just created (use all source columns).
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
                # Return None to indicate all source columns should be used
                return None
            except Exception as e:
                raise CopyError(f"Failed to create table {table_name}: {str(e)}")
        else:
            raise CopyError(f"Target table {table_name} does not exist. Use create_if_not_exists=True to create it.")

    # Get column mapping (columns that exist in both source and target)
    mapped_columns = get_column_mapping(source_data, target_url, table_name)

    if not mapped_columns:
        raise CopyError(f"No matching columns found between source data and target table {table_name}")

    # Log info about column filtering if any columns are excluded
    source_columns = set(source_data.columns.str.lower())
    target_columns = set(get_table_columns(target_url, table_name) or [])
    excluded_columns = source_columns - target_columns

    if excluded_columns:
        safe_print(
            f"ℹ️ Column mapping: {len(mapped_columns)} columns will be copied. "
            f"Excluded {len(excluded_columns)} source columns not in target: {', '.join(sorted(excluded_columns))}"
        )

    return mapped_columns


def filter_dataframe_columns(
    data: pd.DataFrame,
    column_mapping: Optional[List[str]]
) -> pd.DataFrame:
    """
    Filter a DataFrame to only include the columns in the mapping.

    Args:
        data: Source DataFrame to filter
        column_mapping: List of column names to keep. If None, return original DataFrame.

    Returns:
        Filtered DataFrame with only the mapped columns
    """
    if column_mapping is None:
        return data

    # Filter to only include columns that exist in the mapping
    # Use intersection to handle case where mapping has columns not in data
    cols_to_keep = [col for col in column_mapping if col in data.columns]

    if not cols_to_keep:
        return data

    return data[cols_to_keep]


def check_table_partitioned(connection_url: str, table_name: str) -> Dict[str, Any]:
    """
    Check if a table is partitioned and retrieve partition information.

    This function queries system catalogs to determine if the target table uses
    partitioning and, if so, retrieves information about partition columns and
    existing partitions.

    Args:
        connection_url: Database connection URL
        table_name: Table name to check

    Returns:
        Dictionary with partition information:
        - is_partitioned: bool - Whether the table is partitioned
        - partition_type: str or None - Type of partitioning (RANGE, LIST, HASH, etc.)
        - partition_columns: List[str] or None - Columns used for partitioning
        - partitions: List[Dict] or None - List of existing partitions with their bounds
        - database_type: str - Type of database
    """
    db_type = detect_database_type(connection_url)
    result = {
        'is_partitioned': False,
        'partition_type': None,
        'partition_columns': None,
        'partitions': None,
        'database_type': db_type
    }

    try:
        engine = create_engine(connection_url)

        if db_type == 'postgresql':
            result = _check_postgresql_partition(engine, table_name, result)
        elif db_type == 'oracle':
            result = _check_oracle_partition(engine, table_name, result)
        elif db_type == 'mysql':
            result = _check_mysql_partition(engine, table_name, result)
        elif db_type == 'mssql':
            result = _check_mssql_partition(engine, table_name, result)

    except Exception as e:
        # If partition check fails, log and return non-partitioned result
        safe_print(f"⚠ Could not check partition status for {table_name}: {str(e)}")

    return result


def _check_postgresql_partition(engine, table_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check partition information for PostgreSQL tables."""
    from sqlalchemy import text

    with engine.connect() as conn:
        # Check if table is partitioned (PostgreSQL 10+)
        partition_check = conn.execute(text("""
            SELECT
                c.relkind,
                p.partstrat
            FROM pg_class c
            LEFT JOIN pg_partitioned_table p ON c.oid = p.partrelid
            WHERE c.relname = :table_name
              AND c.relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        """), {"table_name": table_name})

        row = partition_check.fetchone()
        if row and row[0] == 'p':  # 'p' means partitioned table
            result['is_partitioned'] = True
            # Map partition strategy
            strat_map = {'r': 'RANGE', 'l': 'LIST', 'h': 'HASH'}
            result['partition_type'] = strat_map.get(row[1], 'UNKNOWN')

            # Get partition columns
            part_cols = conn.execute(text("""
                SELECT a.attname
                FROM pg_partitioned_table p
                JOIN pg_class c ON c.oid = p.partrelid
                JOIN pg_attribute a ON a.attrelid = c.oid
                WHERE c.relname = :table_name
                  AND a.attnum = ANY(p.partattrs)
                ORDER BY array_position(p.partattrs, a.attnum)
            """), {"table_name": table_name})
            result['partition_columns'] = [r[0] for r in part_cols.fetchall()]

            # Get existing partitions
            partitions = conn.execute(text("""
                SELECT
                    child.relname as partition_name,
                    pg_get_expr(child.relpartbound, child.oid) as partition_bound
                FROM pg_inherits
                JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
                JOIN pg_class child ON pg_inherits.inhrelid = child.oid
                WHERE parent.relname = :table_name
                ORDER BY child.relname
            """), {"table_name": table_name})
            result['partitions'] = [
                {'name': r[0], 'bound': r[1]} for r in partitions.fetchall()
            ]

    return result


def _check_oracle_partition(engine, table_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check partition information for Oracle tables."""
    from sqlalchemy import text

    with engine.connect() as conn:
        # Check if table is partitioned
        partition_check = conn.execute(text("""
            SELECT partitioned FROM user_tables WHERE table_name = UPPER(:table_name)
        """), {"table_name": table_name})

        row = partition_check.fetchone()
        if row and row[0] == 'YES':
            result['is_partitioned'] = True

            # Get partition type and columns
            part_info = conn.execute(text("""
                SELECT partitioning_type, column_name
                FROM user_part_tables t
                JOIN user_part_key_columns c ON t.table_name = c.name
                WHERE t.table_name = UPPER(:table_name)
                ORDER BY c.column_position
            """), {"table_name": table_name})

            rows = part_info.fetchall()
            if rows:
                result['partition_type'] = rows[0][0]
                result['partition_columns'] = [r[1] for r in rows]

            # Get existing partitions
            partitions = conn.execute(text("""
                SELECT partition_name, high_value
                FROM user_tab_partitions
                WHERE table_name = UPPER(:table_name)
                ORDER BY partition_position
            """), {"table_name": table_name})
            result['partitions'] = [
                {'name': r[0], 'bound': r[1]} for r in partitions.fetchall()
            ]

    return result


def _check_mysql_partition(engine, table_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check partition information for MySQL tables."""
    from sqlalchemy import text

    with engine.connect() as conn:
        # Get partition information from INFORMATION_SCHEMA
        partition_check = conn.execute(text("""
            SELECT PARTITION_METHOD, PARTITION_EXPRESSION
            FROM INFORMATION_SCHEMA.PARTITIONS
            WHERE TABLE_NAME = :table_name
              AND TABLE_SCHEMA = DATABASE()
              AND PARTITION_NAME IS NOT NULL
            LIMIT 1
        """), {"table_name": table_name})

        row = partition_check.fetchone()
        if row:
            result['is_partitioned'] = True
            result['partition_type'] = row[0]
            # Parse partition expression to get column(s)
            result['partition_columns'] = [row[1].strip('`').strip()] if row[1] else []

            # Get existing partitions
            partitions = conn.execute(text("""
                SELECT PARTITION_NAME, PARTITION_DESCRIPTION
                FROM INFORMATION_SCHEMA.PARTITIONS
                WHERE TABLE_NAME = :table_name
                  AND TABLE_SCHEMA = DATABASE()
                  AND PARTITION_NAME IS NOT NULL
                ORDER BY PARTITION_ORDINAL_POSITION
            """), {"table_name": table_name})
            result['partitions'] = [
                {'name': r[0], 'bound': r[1]} for r in partitions.fetchall()
            ]

    return result


def _check_mssql_partition(engine, table_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Check partition information for SQL Server tables."""
    from sqlalchemy import text

    with engine.connect() as conn:
        # Check if table uses a partition scheme
        partition_check = conn.execute(text("""
            SELECT
                ps.name as partition_scheme,
                pf.name as partition_function,
                c.name as column_name
            FROM sys.tables t
            JOIN sys.indexes i ON t.object_id = i.object_id AND i.index_id <= 1
            JOIN sys.partition_schemes ps ON i.data_space_id = ps.data_space_id
            JOIN sys.partition_functions pf ON ps.function_id = pf.function_id
            JOIN sys.index_columns ic ON i.object_id = ic.object_id
                AND i.index_id = ic.index_id AND ic.partition_ordinal > 0
            JOIN sys.columns c ON t.object_id = c.object_id AND ic.column_id = c.column_id
            WHERE t.name = :table_name
        """), {"table_name": table_name})

        row = partition_check.fetchone()
        if row:
            result['is_partitioned'] = True
            result['partition_type'] = 'RANGE'  # SQL Server primarily uses range partitioning
            result['partition_columns'] = [row[2]]

            # Get partition boundary values
            partitions = conn.execute(text("""
                SELECT
                    p.partition_number,
                    prv.value as boundary_value
                FROM sys.tables t
                JOIN sys.indexes i ON t.object_id = i.object_id AND i.index_id <= 1
                JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
                JOIN sys.partition_schemes ps ON i.data_space_id = ps.data_space_id
                JOIN sys.partition_functions pf ON ps.function_id = pf.function_id
                LEFT JOIN sys.partition_range_values prv
                    ON pf.function_id = prv.function_id AND p.partition_number = prv.boundary_id + 1
                WHERE t.name = :table_name
                ORDER BY p.partition_number
            """), {"table_name": table_name})
            result['partitions'] = [
                {'name': f"partition_{r[0]}", 'bound': str(r[1]) if r[1] is not None else None}
                for r in partitions.fetchall()
            ]

    return result


def perform_partition_maintenance(
    connection_url: str,
    table_name: str,
    partition_info: Dict[str, Any],
    source_data: pd.DataFrame
) -> bool:
    """
    Perform partition maintenance to ensure partitions exist for incoming data.

    This function analyzes the source data and creates any necessary partitions
    in the target table before data is copied. The maintenance operations are
    database-specific and handle different partitioning strategies.

    Args:
        connection_url: Target database connection URL
        table_name: Target table name
        partition_info: Partition information from check_table_partitioned()
        source_data: Source DataFrame to be copied

    Returns:
        True if maintenance was successful or not needed, False if it failed
    """
    if not partition_info.get('is_partitioned'):
        return True  # Nothing to do for non-partitioned tables

    db_type = partition_info.get('database_type', 'unknown')
    partition_columns = partition_info.get('partition_columns', [])
    partition_type = partition_info.get('partition_type', '')
    existing_partitions = partition_info.get('partitions', [])

    if not partition_columns:
        safe_print(f"⚠ Partitioned table {table_name} has no partition columns defined, skipping maintenance")
        return True

    # Check if partition columns exist in source data
    missing_cols = [col for col in partition_columns if col.lower() not in [c.lower() for c in source_data.columns]]
    if missing_cols:
        safe_print(f"⚠ Source data missing partition columns {missing_cols}, skipping partition maintenance")
        return True

    safe_print(f"🔧 Performing partition maintenance for {table_name} ({partition_type} on {partition_columns})")

    try:
        engine = create_engine(connection_url)

        if db_type == 'postgresql':
            return _maintain_postgresql_partitions(
                engine, table_name, partition_type, partition_columns,
                existing_partitions, source_data
            )
        elif db_type == 'oracle':
            return _maintain_oracle_partitions(
                engine, table_name, partition_type, partition_columns,
                existing_partitions, source_data
            )
        elif db_type == 'mysql':
            return _maintain_mysql_partitions(
                engine, table_name, partition_type, partition_columns,
                existing_partitions, source_data
            )
        elif db_type == 'mssql':
            # SQL Server partition maintenance requires more complex setup
            # and is typically done through partition switching, not dynamic creation
            safe_print(f"ℹ️ SQL Server partition maintenance is handled by the partition function")
            return True
        else:
            safe_print(f"⚠ Partition maintenance not implemented for {db_type}")
            return True

    except Exception as e:
        safe_print(f"⚠ Partition maintenance failed for {table_name}: {str(e)}")
        return False


def _maintain_postgresql_partitions(
    engine,
    table_name: str,
    partition_type: str,
    partition_columns: List[str],
    existing_partitions: List[Dict],
    source_data: pd.DataFrame
) -> bool:
    """Create missing PostgreSQL partitions for incoming data."""
    from sqlalchemy import text

    # Get the partition column from source data (case-insensitive match)
    part_col = None
    for col in source_data.columns:
        if col.lower() == partition_columns[0].lower():
            part_col = col
            break

    if part_col is None:
        return True

    existing_names = {p['name'] for p in existing_partitions}

    with engine.connect() as conn:
        if partition_type == 'RANGE':
            # For RANGE partitions, create partitions for date ranges in source data
            if pd.api.types.is_datetime64_any_dtype(source_data[part_col]):
                # Get unique months/years from source data
                dates = pd.to_datetime(source_data[part_col].dropna())
                if len(dates) > 0:
                    periods = dates.dt.to_period('M').unique()
                    for period in periods:
                        partition_name = f"{table_name}_{period.strftime('%Y_%m')}"
                        if partition_name not in existing_names:
                            start_date = period.start_time.strftime('%Y-%m-%d')
                            end_date = (period + 1).start_time.strftime('%Y-%m-%d')
                            try:
                                conn.execute(text(f"""
                                    CREATE TABLE IF NOT EXISTS {partition_name}
                                    PARTITION OF {table_name}
                                    FOR VALUES FROM ('{start_date}') TO ('{end_date}')
                                """))
                                conn.commit()
                                safe_print(f"  ✓ Created partition {partition_name}")
                            except Exception as e:
                                if 'already exists' not in str(e).lower():
                                    safe_print(f"  ⚠ Could not create partition {partition_name}: {e}")

        elif partition_type == 'LIST':
            # For LIST partitions, create partitions for unique values
            unique_values = source_data[part_col].dropna().unique()
            for value in unique_values:
                partition_name = f"{table_name}_{str(value).replace(' ', '_').replace('-', '_')[:50]}"
                if partition_name not in existing_names:
                    try:
                        if isinstance(value, str):
                            conn.execute(text(f"""
                                CREATE TABLE IF NOT EXISTS {partition_name}
                                PARTITION OF {table_name}
                                FOR VALUES IN ('{value}')
                            """))
                        else:
                            conn.execute(text(f"""
                                CREATE TABLE IF NOT EXISTS {partition_name}
                                PARTITION OF {table_name}
                                FOR VALUES IN ({value})
                            """))
                        conn.commit()
                        safe_print(f"  ✓ Created partition {partition_name}")
                    except Exception as e:
                        if 'already exists' not in str(e).lower():
                            safe_print(f"  ⚠ Could not create partition {partition_name}: {e}")

    return True


def _maintain_oracle_partitions(
    engine,
    table_name: str,
    partition_type: str,
    partition_columns: List[str],
    existing_partitions: List[Dict],
    source_data: pd.DataFrame
) -> bool:
    """Create missing Oracle partitions for incoming data."""
    from sqlalchemy import text

    # Get the partition column from source data (case-insensitive match)
    part_col = None
    for col in source_data.columns:
        if col.lower() == partition_columns[0].lower():
            part_col = col
            break

    if part_col is None:
        return True

    with engine.connect() as conn:
        if partition_type == 'RANGE':
            # For RANGE partitions, check if we need new partitions
            if pd.api.types.is_datetime64_any_dtype(source_data[part_col]):
                max_date = pd.to_datetime(source_data[part_col].dropna()).max()
                if pd.notna(max_date):
                    # Oracle often uses MAXVALUE for the last partition
                    # Add a new partition if data exceeds existing bounds
                    partition_name = f"P_{max_date.strftime('%Y%m')}"
                    try:
                        # Use ALTER TABLE to add partition
                        conn.execute(text(f"""
                            ALTER TABLE {table_name} ADD PARTITION {partition_name}
                            VALUES LESS THAN (TO_DATE('{(max_date + pd.DateOffset(months=1)).strftime('%Y-%m-01')}', 'YYYY-MM-DD'))
                        """))
                        conn.commit()
                        safe_print(f"  ✓ Created partition {partition_name}")
                    except Exception as e:
                        # ORA-14074: partition bound must collate higher than that of the last partition
                        # This is expected if partition already covers the range
                        if 'ora-14074' not in str(e).lower() and 'already exists' not in str(e).lower():
                            safe_print(f"  ℹ️ Partition creation note: {str(e)[:100]}")

        elif partition_type == 'LIST':
            # For LIST partitions, add new values
            unique_values = source_data[part_col].dropna().unique()
            for value in unique_values:
                partition_name = f"P_{str(value).replace(' ', '_')[:20]}"
                try:
                    if isinstance(value, str):
                        conn.execute(text(f"""
                            ALTER TABLE {table_name} ADD PARTITION {partition_name}
                            VALUES ('{value}')
                        """))
                    else:
                        conn.execute(text(f"""
                            ALTER TABLE {table_name} ADD PARTITION {partition_name}
                            VALUES ({value})
                        """))
                    conn.commit()
                    safe_print(f"  ✓ Created partition {partition_name}")
                except Exception as e:
                    if 'already exists' not in str(e).lower():
                        pass  # Silently skip if value already in a partition

    return True


def _maintain_mysql_partitions(
    engine,
    table_name: str,
    partition_type: str,
    partition_columns: List[str],
    existing_partitions: List[Dict],
    source_data: pd.DataFrame
) -> bool:
    """Create missing MySQL partitions for incoming data."""
    from sqlalchemy import text

    # Get the partition column from source data (case-insensitive match)
    part_col = None
    for col in source_data.columns:
        if col.lower() == partition_columns[0].lower():
            part_col = col
            break

    if part_col is None:
        return True

    with engine.connect() as conn:
        if partition_type in ('RANGE', 'RANGE COLUMNS'):
            if pd.api.types.is_datetime64_any_dtype(source_data[part_col]):
                max_date = pd.to_datetime(source_data[part_col].dropna()).max()
                if pd.notna(max_date):
                    partition_name = f"p{max_date.strftime('%Y%m')}"
                    next_month = (max_date + pd.DateOffset(months=1)).strftime('%Y-%m-01')
                    try:
                        conn.execute(text(f"""
                            ALTER TABLE {table_name} ADD PARTITION (
                                PARTITION {partition_name} VALUES LESS THAN ('{next_month}')
                            )
                        """))
                        conn.commit()
                        safe_print(f"  ✓ Created partition {partition_name}")
                    except Exception as e:
                        if 'already exists' not in str(e).lower() and 'duplicate' not in str(e).lower():
                            safe_print(f"  ℹ️ Partition note: {str(e)[:100]}")

        elif partition_type in ('LIST', 'LIST COLUMNS'):
            unique_values = source_data[part_col].dropna().unique()
            for value in unique_values:
                partition_name = f"p_{str(value).replace(' ', '_')[:20]}"
                try:
                    if isinstance(value, str):
                        conn.execute(text(f"""
                            ALTER TABLE {table_name} ADD PARTITION (
                                PARTITION {partition_name} VALUES IN ('{value}')
                            )
                        """))
                    else:
                        conn.execute(text(f"""
                            ALTER TABLE {table_name} ADD PARTITION (
                                PARTITION {partition_name} VALUES IN ({value})
                            )
                        """))
                    conn.commit()
                    safe_print(f"  ✓ Created partition {partition_name}")
                except Exception as e:
                    if 'already exists' not in str(e).lower() and 'duplicate' not in str(e).lower():
                        pass  # Silently skip

    return True


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
    apply_masks: bool = True,
    verbose_batch_logs: bool = True,
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
            # Handle batched results with optional per-batch logging
            result = execute_query(connection_url, query, batch_size, source_env, apply_masks)

            first_batch = True
            total_records = 0
            batch_index = 0
            overall_start = time.perf_counter()

            for chunk in result:
                batch_index += 1
                batch_start = time.perf_counter()

                current_mode = file_mode if first_batch else 'a'
                write_to_file(chunk, file_path, format_enum, current_mode)
                total_records += len(chunk)
                first_batch = False

                batch_end = time.perf_counter()
                if verbose_batch_logs:
                    safe_print(
                        f"Batch {batch_index}: wrote {len(chunk):,} records to file in "
                        f"{batch_end - batch_start:.3f}s",
                    )

            overall_end = time.perf_counter()
            safe_print(
                f"Batch copy to file summary: wrote {total_records:,} records in "
                f"{overall_end - overall_start:.3f}s",
            )
        else:
            # Handle single result
            overall_start = time.perf_counter()
            result = execute_query(connection_url, query, None, source_env, apply_masks)
            write_to_file(result, file_path, format_enum, file_mode)
            total_records = len(result)
            overall_end = time.perf_counter()
            safe_print(
                f"Copy to file summary: wrote {total_records:,} records in "
                f"{overall_end - overall_start:.3f}s",
            )

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
    apply_masks: bool = True,
    verbose_batch_logs: bool = True,
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
        read_start = time.perf_counter()
        data = read_from_file(file_path, format_enum)
        read_end = time.perf_counter()
        safe_print(f"✓ Loaded {len(data):,} records from file in {read_end - read_start:.3f}s")

        # Apply masking if needed
        if apply_masks:
            data = apply_masking(data, target_env)

        # Validate target table if requested
        if validate_target:
            validate_target_table(data, connection_url, table, create_if_not_exists)

        # Write to database
        safe_print(f"📊 Writing data to table '{table}'...")
        overall_start = time.perf_counter()
        total_records = write_to_db(data, connection_url, table, mode_enum, batch_size)
        overall_end = time.perf_counter()
        safe_print(
            f"✓ Successfully wrote {total_records:,} records in "
            f"{overall_end - overall_start:.3f}s",
        )

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
    apply_masks: bool = True,
    verbose_batch_logs: bool = True,
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
        verbose_batch_logs: Whether to print per-batch timing logs (in addition to summary)

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

        # Column mapping for filtering source columns to match target table
        # None means use all source columns (table doesn't exist or was just created)
        column_mapping: Optional[List[str]] = None

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
                    # validate_target_table now returns the column mapping
                    column_mapping = validate_target_table(sample_data, target_url, table, create_if_not_exists)
            except Exception as e:
                raise CopyError(f"Error during validation: {str(e)}")
        else:
            # No validation requested, but still check if we need column mapping
            # for an existing table
            if check_table_exists(target_url, table):
                # Create a sample query to get source columns
                if 'mssql' in source_url.lower():
                    if query.strip().upper().startswith('SELECT'):
                        select_pos = query.upper().find('SELECT') + 6
                        sample_query = query[:select_pos] + ' TOP 1' + query[select_pos:]
                    else:
                        sample_query = query
                elif 'oracle' in source_url.lower():
                    sample_query = f"SELECT * FROM ({query}) WHERE ROWNUM <= 1"
                else:
                    sample_query = f"{query} LIMIT 1"

                try:
                    sample_data = execute_query(source_url, sample_query, None, None, False)
                    if len(sample_data) > 0:
                        column_mapping = get_column_mapping(sample_data, target_url, table)
                except Exception:
                    # If we can't get column mapping, proceed without it
                    pass

        # Check if target table is partitioned and cache partition info
        # This is done once before data copy starts
        partition_info: Optional[Dict[str, Any]] = None
        if check_table_exists(target_url, table):
            partition_info = check_table_partitioned(target_url, table)
            if partition_info.get('is_partitioned'):
                safe_print(
                    f"ℹ️ Target table '{table}' is partitioned "
                    f"({partition_info.get('partition_type')} on {partition_info.get('partition_columns')})"
                )

        # Execute query and write to target
        overall_start_ts: Optional[datetime] = None
        overall_start_perf: Optional[float] = None

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

            overall_start_ts = datetime.now()
            overall_start_perf = time.perf_counter()

            # Track if partition maintenance has been performed for this copy operation
            partition_maintenance_done = False

            if not parallel_workers or parallel_workers <= 1:
                # Sequential path (no parallelism requested)
                for chunk in result_iter:
                    batch_number += 1
                    current_mode = mode_enum if batch_number == 1 else WriteMode.APPEND

                    # Filter columns to match target table schema
                    filtered_chunk = filter_dataframe_columns(chunk, column_mapping)

                    # Perform partition maintenance before first write if table is partitioned
                    if batch_number == 1 and partition_info and partition_info.get('is_partitioned'):
                        if not partition_maintenance_done:
                            perform_partition_maintenance(target_url, table, partition_info, filtered_chunk)
                            partition_maintenance_done = True

                    start_ts = datetime.now()
                    start_perf = time.perf_counter()
                    records_written = write_to_db(filtered_chunk, target_url, table, mode=current_mode)
                    duration = time.perf_counter() - start_perf
                    end_ts = datetime.now()

                    total_records += records_written

                    if verbose_batch_logs:
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
                    # Filter columns to match target table schema
                    filtered_first_chunk = filter_dataframe_columns(first_chunk, column_mapping)

                    # Perform partition maintenance before first write if table is partitioned
                    if partition_info and partition_info.get('is_partitioned'):
                        if not partition_maintenance_done:
                            perform_partition_maintenance(target_url, table, partition_info, filtered_first_chunk)
                            partition_maintenance_done = True

                    start_ts = datetime.now()
                    start_perf = time.perf_counter()
                    records_written = write_to_db(filtered_first_chunk, target_url, table, mode=mode_enum)
                    duration = time.perf_counter() - start_perf
                    end_ts = datetime.now()

                    total_records += records_written

                    if verbose_batch_logs:
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

                    # Filter columns to match target table schema
                    filtered_chunk_df = filter_dataframe_columns(chunk_df, column_mapping)

                    start = datetime.now()
                    start_perf_local = time.perf_counter()
                    written = write_to_db(filtered_chunk_df, target_url, table, mode=WriteMode.APPEND)
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
                        if verbose_batch_logs:
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
            # Handle single result (non-batched)
            safe_print("📊 Copying data in single batch...")
            result = execute_query(
                source_url,
                query,
                None,
                target_env if apply_masks else None,
                apply_masks,
            )

            # Filter columns to match target table schema
            filtered_result = filter_dataframe_columns(result, column_mapping)

            # Perform partition maintenance before writing if table is partitioned
            if partition_info and partition_info.get('is_partitioned'):
                perform_partition_maintenance(target_url, table, partition_info, filtered_result)

            overall_start_ts = datetime.now()
            overall_start_perf = time.perf_counter()
            total_records = write_to_db(filtered_result, target_url, table, mode=mode_enum)
            overall_duration = time.perf_counter() - overall_start_perf
            overall_end_ts = datetime.now()

            if overall_duration > 0:
                throughput = total_records / overall_duration
            else:
                throughput = 0.0

            # For single-batch operations, this acts as both per-batch and summary log
            safe_print(
                f"✓ Copied {total_records:,} records "
                f"in {overall_duration:.2f}s "
                f"({throughput:,.2f} records/s) "
                f"| Start: {overall_start_ts.isoformat(timespec='seconds')} "
                f"| End: {overall_end_ts.isoformat(timespec='seconds')}"
            )

        # Emit final summary for batched operations
        if batch_size and overall_start_perf is not None and overall_start_ts is not None:
            overall_end_ts = datetime.now()
            overall_duration = time.perf_counter() - overall_start_perf
            if overall_duration > 0:
                throughput = total_records / overall_duration
            else:
                throughput = 0.0

            safe_print(
                f"📈 Batch copy summary: Copied {total_records:,} records "
                f"in {overall_duration:.2f}s "
                f"({throughput:,.2f} records/s) "
                f"| Start: {overall_start_ts.isoformat(timespec='seconds')} "
                f"| End: {overall_end_ts.isoformat(timespec='seconds')}"
            )

        message = f"Successfully copied {total_records} records to table '{table}'"
        if apply_masks:
            message += " (with masking applied)"

        return create_success_result(message, record_count=total_records)

    except Exception as e:
        return handle_exception(e, "database to database copy")
