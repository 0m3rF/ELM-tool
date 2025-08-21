"""
ELM Tool Core Data Generation

Unified data generation operations for both CLI and API interfaces.
This module provides consistent random data generation functionality.
"""

import pandas as pd
from typing import List, Dict, Any, Optional

from elm.core.types import GenerationConfig, FileFormat, WriteMode, OperationResult
from elm.core.exceptions import GenerationError, ValidationError
from elm.core.utils import (
    validate_file_format, validate_write_mode, create_success_result, 
    create_error_result, handle_exception, validate_required_params,
    parse_columns_string
)
from elm.core.environment import get_connection_url
from elm.core.copy import check_table_exists, get_table_columns, write_to_db, write_to_file
from elm.elm_utils.random_data import generate_random_data


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
) -> OperationResult:
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
        OperationResult with generated data
    """
    try:
        # Validate inputs
        if num_records <= 0:
            raise ValidationError("Number of records must be greater than 0")
        
        column_list = columns or []
        
        # Get schema from database if environment and table are provided
        if environment and table:
            # Get connection URL
            connection_url = get_connection_url(environment)
            
            # Check if table exists
            if not check_table_exists(connection_url, table):
                raise GenerationError(f"Table '{table}' does not exist in environment '{environment}'")
            
            # Get table columns
            db_columns = get_table_columns(connection_url, table)
            if not db_columns:
                raise GenerationError(f"Could not retrieve columns for table '{table}'")
            
            # Use table columns if no columns were specified
            if not column_list:
                column_list = db_columns
            else:
                # Validate that all provided columns exist in the table
                missing_columns = set(column_list) - set(db_columns)
                if missing_columns:
                    raise ValidationError(f"The following columns do not exist in table '{table}': {', '.join(missing_columns)}")
        
        # Ensure we have columns to generate data for
        if not column_list:
            raise ValidationError("No columns specified. Please provide columns or a table schema.")
        
        # Generate random data
        data = generate_random_data(
            columns=column_list,
            num_records=num_records,
            string_length=string_length,
            pattern=pattern or {},
            min_number=min_number,
            max_number=max_number,
            decimal_places=decimal_places,
            start_date=start_date,
            end_date=end_date,
            date_format=date_format
        )
        
        return create_success_result(
            f"Successfully generated {num_records} records for columns: {', '.join(column_list)}",
            data=data.to_dict('records'),
            record_count=num_records
        )
        
    except Exception as e:
        return handle_exception(e, "data generation")


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
) -> OperationResult:
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
        OperationResult with operation status and details
    """
    try:
        # Generate data first
        generation_result = generate_data(
            num_records=num_records,
            columns=columns,
            environment=environment,
            table=table,
            **kwargs
        )
        
        if not generation_result.success:
            return generation_result
        
        # Convert data back to DataFrame
        data = pd.DataFrame(generation_result.data)
        
        # Write to database if requested
        if write_to_db:
            if not environment or not table:
                return create_error_result(
                    "Environment and table are required when writing to database"
                )
            
            # Validate and convert parameters
            mode_enum = validate_write_mode(mode)
            
            # Get connection URL
            connection_url = get_connection_url(environment, encryption_key)
            
            # Write to database
            write_to_db_func = write_to_db  # Avoid name conflict
            from elm.core.copy import write_to_db as write_db_func
            write_db_func(data, connection_url, table, mode_enum)
            
            return create_success_result(
                f"Successfully wrote {num_records} records to table '{table}'",
                data=data.head(5).to_dict('records'),  # Return first 5 records as preview
                record_count=num_records
            )
        
        # Write to file if output is provided
        elif output_file:
            # Validate and convert parameters
            format_enum = validate_file_format(file_format)
            
            # Write to file
            write_to_file(data, output_file, format_enum)
            
            return create_success_result(
                f"Successfully wrote {num_records} records to file '{output_file}'",
                data=data.head(5).to_dict('records'),  # Return first 5 records as preview
                record_count=num_records
            )
        
        # Otherwise, just return the data
        else:
            return create_success_result(
                f"Successfully generated {num_records} records",
                data=data.to_dict('records'),
                record_count=num_records
            )
            
    except Exception as e:
        return handle_exception(e, "data generation and save")


def generate_from_columns_string(
    columns_str: str,
    num_records: int = 10,
    **kwargs
) -> OperationResult:
    """
    Generate data from a comma-separated columns string (CLI compatibility).
    
    Args:
        columns_str: Comma-separated column names
        num_records: Number of records to generate
        **kwargs: Additional parameters for generate_data
    
    Returns:
        OperationResult with generated data
    """
    try:
        # Parse columns string
        columns = parse_columns_string(columns_str)
        
        if not columns:
            raise ValidationError("No valid columns found in columns string")
        
        return generate_data(
            num_records=num_records,
            columns=columns,
            **kwargs
        )
        
    except Exception as e:
        return handle_exception(e, "data generation from columns string")


def generate_with_single_pattern(
    columns: List[str],
    pattern: str,
    num_records: int = 10,
    **kwargs
) -> OperationResult:
    """
    Generate data with a single pattern applied to all columns (CLI compatibility).
    
    Args:
        columns: List of column names
        pattern: Single pattern to apply to all columns
        num_records: Number of records to generate
        **kwargs: Additional parameters for generate_data
    
    Returns:
        OperationResult with generated data
    """
    try:
        # Convert single pattern to pattern dictionary
        pattern_dict = {col: pattern for col in columns}
        
        return generate_data(
            num_records=num_records,
            columns=columns,
            pattern=pattern_dict,
            **kwargs
        )
        
    except Exception as e:
        return handle_exception(e, "data generation with single pattern")
