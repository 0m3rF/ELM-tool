"""
ELM Tool Streaming Data Copy Operations

High-performance streaming data copy using database-specific bulk loaders.
This module provides optimized streaming for large datasets with LOB data.
"""

import io
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError

from elm.core.types import WriteMode
from elm.core.exceptions import DatabaseError, CopyError
from elm.core.utils import convert_sqlalchemy_mode


def detect_database_type(connection_url: str) -> str:
    """
    Detect database type from connection URL.
    
    Args:
        connection_url: SQLAlchemy connection URL
        
    Returns:
        Database type: 'postgresql', 'oracle', 'mysql', 'mssql', or 'unknown'
    """
    url_lower = connection_url.lower()
    
    if 'postgresql' in url_lower or 'postgres' in url_lower:
        return 'postgresql'
    elif 'oracle' in url_lower:
        return 'oracle'
    elif 'mysql' in url_lower:
        return 'mysql'
    elif 'mssql' in url_lower or 'sqlserver' in url_lower:
        return 'mssql'
    else:
        return 'unknown'


def get_table_columns(engine, table_name: str) -> List[str]:
    """
    Get column names from a table.
    
    Args:
        engine: SQLAlchemy engine
        table_name: Table name
        
    Returns:
        List of column names
    """
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table_name])
    
    if table_name not in metadata.tables:
        raise DatabaseError(f"Table '{table_name}' not found")
    
    table = metadata.tables[table_name]
    return [col.name for col in table.columns]


def write_postgresql_copy(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND
) -> int:
    """
    Write data to PostgreSQL using COPY protocol (fastest method).
    
    Args:
        data: DataFrame to write
        connection_url: PostgreSQL connection URL
        table_name: Target table name
        mode: Write mode
        
    Returns:
        Number of records written
    """
    try:
        import psycopg2
        from psycopg2 import sql
    except ImportError:
        raise CopyError("psycopg2 not installed. Install with: pip install psycopg2-binary")
    
    # Parse connection URL
    parsed = urlparse(connection_url)
    
    # Extract connection parameters
    conn_params = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Handle mode
        if mode == WriteMode.REPLACE:
            cursor.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name)))
        
        # Create CSV buffer
        buffer = io.StringIO()
        data.to_csv(buffer, index=False, header=False)
        buffer.seek(0)
        
        # Use COPY for bulk insert
        columns = list(data.columns)
        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH CSV").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, columns))
        )
        
        cursor.copy_expert(copy_sql, buffer)
        conn.commit()
        
        record_count = len(data)
        
        cursor.close()
        conn.close()
        
        return record_count
        
    except Exception as e:
        raise DatabaseError(f"PostgreSQL COPY error: {str(e)}")


def write_postgresql_executemany(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND
) -> int:
    """
    Write data to PostgreSQL using execute_values (fast bulk insert).
    
    Args:
        data: DataFrame to write
        connection_url: PostgreSQL connection URL
        table_name: Target table name
        mode: Write mode
        
    Returns:
        Number of records written
    """
    try:
        import psycopg2
        from psycopg2 import sql
        from psycopg2.extras import execute_values
    except ImportError:
        raise CopyError("psycopg2 not installed. Install with: pip install psycopg2-binary")
    
    # Parse connection URL
    parsed = urlparse(connection_url)
    
    conn_params = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Handle mode
        if mode == WriteMode.REPLACE:
            cursor.execute(sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name)))
        
        # Prepare data
        columns = list(data.columns)
        values = [tuple(row) for row in data.values]
        
        # Build INSERT statement
        insert_sql = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
            sql.Identifier(table_name),
            sql.SQL(', ').join(map(sql.Identifier, columns))
        )
        
        # Execute bulk insert
        execute_values(cursor, insert_sql, values, page_size=1000)
        conn.commit()
        
        record_count = len(data)
        
        cursor.close()
        conn.close()
        
        return record_count
        
    except Exception as e:
        raise DatabaseError(f"PostgreSQL execute_values error: {str(e)}")


def write_oracle_executemany(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND
) -> int:
    """
    Write data to Oracle using executemany with array binding (fast bulk insert).
    
    Args:
        data: DataFrame to write
        connection_url: Oracle connection URL
        table_name: Target table name
        mode: Write mode
        
    Returns:
        Number of records written
    """
    try:
        import oracledb
    except ImportError:
        raise CopyError("oracledb not installed. Install with: pip install oracledb")
    
    # Parse connection URL to extract credentials
    parsed = urlparse(connection_url)
    
    # Extract connection parameters
    user = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port or 1521
    
    # Extract service name or SID from URL
    if '?service_name=' in connection_url:
        service_name = connection_url.split('?service_name=')[1].split('&')[0]
        dsn = oracledb.makedsn(host, port, service_name=service_name)
    else:
        sid = parsed.path.lstrip('/')
        dsn = oracledb.makedsn(host, port, sid=sid)
    
    try:
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
        cursor = conn.cursor()
        
        # Handle mode
        if mode == WriteMode.REPLACE:
            cursor.execute(f"TRUNCATE TABLE {table_name}")
        
        # Prepare data
        columns = list(data.columns)
        values = [tuple(row) for row in data.values]
        
        # Build INSERT statement with bind variables
        placeholders = ', '.join([f':{i+1}' for i in range(len(columns))])
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Execute bulk insert with array binding
        cursor.executemany(insert_sql, values, batcherrors=True)
        
        # Check for batch errors
        errors = cursor.getbatcherrors()
        if errors:
            error_msg = f"Batch errors occurred: {len(errors)} rows failed"
            for error in errors[:5]:  # Show first 5 errors
                error_msg += f"\n  Row {error.offset}: {error.message}"
            raise DatabaseError(error_msg)
        
        conn.commit()
        
        record_count = len(data)
        
        cursor.close()
        conn.close()
        
        return record_count
        
    except Exception as e:
        raise DatabaseError(f"Oracle executemany error: {str(e)}")


def write_mysql_executemany(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND
) -> int:
    """
    Write data to MySQL using executemany (optimized bulk insert).
    
    Args:
        data: DataFrame to write
        connection_url: MySQL connection URL
        table_name: Target table name
        mode: Write mode
        
    Returns:
        Number of records written
    """
    try:
        import pymysql
    except ImportError:
        raise CopyError("pymysql not installed. Install with: pip install pymysql")
    
    # Parse connection URL
    parsed = urlparse(connection_url)
    
    conn_params = {
        'host': parsed.hostname,
        'port': parsed.port or 3306,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }
    
    try:
        conn = pymysql.connect(**conn_params)
        cursor = conn.cursor()
        
        # Handle mode
        if mode == WriteMode.REPLACE:
            cursor.execute(f"TRUNCATE TABLE {table_name}")
        
        # Prepare data
        columns = list(data.columns)
        values = [tuple(row) for row in data.values]
        
        # Build INSERT statement
        placeholders = ', '.join(['%s'] * len(columns))
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Execute bulk insert
        cursor.executemany(insert_sql, values)
        conn.commit()
        
        record_count = len(data)
        
        cursor.close()
        conn.close()
        
        return record_count
        
    except Exception as e:
        raise DatabaseError(f"MySQL executemany error: {str(e)}")


def write_mssql_fast_executemany(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND
) -> int:
    """
    Write data to SQL Server using fast_executemany (optimized bulk insert).

    Args:
        data: DataFrame to write
        connection_url: SQL Server connection URL
        table_name: Target table name
        mode: Write mode

    Returns:
        Number of records written
    """
    try:
        import pyodbc
    except ImportError:
        raise CopyError("pyodbc not installed. Install with: pip install pyodbc")

    # Parse connection URL
    parsed = urlparse(connection_url)

    # Build ODBC connection string
    driver = "ODBC Driver 17 for SQL Server"
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={parsed.hostname},{parsed.port or 1433};"
        f"DATABASE={parsed.path.lstrip('/')};"
        f"UID={parsed.username};"
        f"PWD={parsed.password}"
    )

    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Enable fast_executemany for bulk operations
        cursor.fast_executemany = True

        # Handle mode
        if mode == WriteMode.REPLACE:
            cursor.execute(f"TRUNCATE TABLE {table_name}")

        # Prepare data
        columns = list(data.columns)
        values = [tuple(row) for row in data.values]

        # Build INSERT statement
        placeholders = ', '.join(['?'] * len(columns))
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

        # Execute bulk insert
        cursor.executemany(insert_sql, values)
        conn.commit()

        record_count = len(data)

        cursor.close()
        conn.close()

        return record_count

    except Exception as e:
        raise DatabaseError(f"SQL Server fast_executemany error: {str(e)}")


def write_to_db_streaming(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND,
    batch_size: Optional[int] = None,
    use_optimized: bool = True
) -> int:
    """
    Write data to database using optimized streaming methods.

    This function automatically detects the database type and uses the most
    efficient bulk loading method available. Falls back to pandas to_sql if
    optimized method is not available or fails.

    Args:
        data: DataFrame to write
        connection_url: Database connection URL
        table_name: Target table name
        mode: Write mode (APPEND, REPLACE, FAIL)
        batch_size: Batch size for large datasets (used for fallback)
        use_optimized: Whether to use optimized methods (default: True)

    Returns:
        Number of records written
    """
    if not use_optimized:
        # Use pandas to_sql fallback
        return _write_pandas_fallback(data, connection_url, table_name, mode, batch_size)

    db_type = detect_database_type(connection_url)

    try:
        # Try optimized method based on database type
        if db_type == 'postgresql':
            try:
                # Try COPY first (fastest)
                return write_postgresql_copy(data, connection_url, table_name, mode)
            except Exception as copy_error:
                # Fall back to execute_values
                try:
                    return write_postgresql_executemany(data, connection_url, table_name, mode)
                except Exception:
                    raise copy_error  # Raise original error

        elif db_type == 'oracle':
            return write_oracle_executemany(data, connection_url, table_name, mode)

        elif db_type == 'mysql':
            return write_mysql_executemany(data, connection_url, table_name, mode)

        elif db_type == 'mssql':
            return write_mssql_fast_executemany(data, connection_url, table_name, mode)

        else:
            # Unknown database type, use pandas fallback
            return _write_pandas_fallback(data, connection_url, table_name, mode, batch_size)

    except Exception as e:
        # If optimized method fails, fall back to pandas
        print(f"⚠ Optimized write failed ({str(e)}), falling back to pandas to_sql...")
        return _write_pandas_fallback(data, connection_url, table_name, mode, batch_size)


def _write_pandas_fallback(
    data: pd.DataFrame,
    connection_url: str,
    table_name: str,
    mode: WriteMode = WriteMode.APPEND,
    batch_size: Optional[int] = None
) -> int:
    """
    Fallback method using pandas to_sql.

    Args:
        data: DataFrame to write
        connection_url: Database connection URL
        table_name: Target table name
        mode: Write mode
        batch_size: Batch size for large datasets

    Returns:
        Number of records written
    """
    from elm.core.environment import _handle_oracle_connection_error

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

        return len(data)

    except Exception as e:
        # Check if this is an Oracle connection and try to handle thin mode errors
        if 'oracle' in connection_url.lower():
            if _handle_oracle_connection_error(connection_url, e):
                # Retry the write after Oracle client initialization
                engine = create_engine(connection_url)
                if_exists = convert_sqlalchemy_mode(mode)

                if batch_size and len(data) > batch_size:
                    for i in range(0, len(data), batch_size):
                        batch = data.iloc[i:i+batch_size]
                        current_if_exists = if_exists if i == 0 else 'append'
                        batch.to_sql(table_name, engine, if_exists=current_if_exists, index=False)
                else:
                    data.to_sql(table_name, engine, if_exists=if_exists, index=False)

                return len(data)
            else:
                # Re-raise the original error
                if isinstance(e, SQLAlchemyError):
                    raise DatabaseError(f"Database error: {str(e)}")
                else:
                    raise CopyError(f"Error writing to database: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            if isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Database error: {str(e)}")
            else:
                raise CopyError(f"Error writing to database: {str(e)}")

