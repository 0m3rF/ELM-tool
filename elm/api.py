"""
ELM Tool API - Programmatic interface for the ELM Tool

This module provides functions for programmatically using the ELM Tool
without going through the command-line interface.
"""

import os
import base64
import pandas as pd
from typing import Dict, List, Union, Optional, Any

from elm.elm_utils import variables
from elm.elm_utils.db_utils import (
    get_connection_url, check_table_exists, get_table_columns,
    execute_query, write_to_db, write_to_file
)
from elm.elm_utils.data_utils import apply_masking
from elm.elm_utils.random_data import generate_random_data
from elm.elm_utils.encryption import encrypt_data, decrypt_data, generate_key_from_password
from elm.elm_commands.mask import load_masking_definitions, save_masking_definitions
from elm.elm_utils.data_utils import apply_masking

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
    overwrite: bool = False
) -> bool:
    """
    Create a new database environment.

    Args:
        name: Environment name
        host: Database host
        port: Database port
        user: Database username
        password: Database password
        service: Database service name
        db_type: Database type (ORACLE, MYSQL, MSSQL, POSTGRES)
        encrypt: Whether to encrypt the environment
        encryption_key: Encryption key (required if encrypt=True)
        overwrite: Whether to overwrite if environment already exists

    Returns:
        bool: True if successful, False otherwise
    """
    import configparser
    config = configparser.ConfigParser()

    # Ensure the environment directory exists
    os.makedirs(os.path.dirname(variables.ENVS_FILE), exist_ok=True)

    # Read existing config
    config.read(variables.ENVS_FILE)

    # Check if environment already exists
    if name in config.sections() and not overwrite:
        return False

    # Create or update the environment
    if name not in config.sections():
        config.add_section(name)

    # Set environment properties
    config[name]['host'] = host
    config[name]['port'] = str(port)
    config[name]['user'] = user

    # Handle encryption if needed
    if encrypt:
        if not encryption_key:
            return False

        key, salt = generate_key_from_password(encryption_key)
        config[name]['salt'] = base64.b64encode(salt).decode('utf-8')
        config[name]['password'] = encrypt_data(password, key)
        config[name]['is_encrypted'] = 'True'
    else:
        config[name]['password'] = password
        config[name]['is_encrypted'] = 'False'

    config[name]['service'] = service
    config[name]['type'] = db_type

    # Save the configuration
    with open(variables.ENVS_FILE, 'w') as f:
        config.write(f)

    return True

def list_environments(show_all: bool = False) -> List[Dict[str, Any]]:
    """
    List all environments.

    Args:
        show_all: Whether to show all details (passwords will be masked)

    Returns:
        List of environment dictionaries
    """
    import configparser
    config = configparser.ConfigParser()
    config.read(variables.ENVS_FILE)

    environments = []
    for section in config.sections():
        env = {'name': section}

        if show_all:
            for key in config[section]:
                if key == 'password':
                    env[key] = '********'  # Mask password
                else:
                    env[key] = config[section][key]

        environments.append(env)

    return environments

def get_environment(name: str, encryption_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get details of a specific environment.

    Args:
        name: Environment name
        encryption_key: Encryption key for encrypted environments

    Returns:
        Environment details dictionary or None if not found
    """
    import configparser
    config = configparser.ConfigParser()
    config.read(variables.ENVS_FILE)

    if name not in config.sections():
        return None

    env = {'name': name}
    for key in config[name]:
        if key == 'password' and config[name].get('is_encrypted', 'False') == 'True':
            if encryption_key:
                try:
                    salt = base64.b64decode(config[name]['salt'].encode('utf-8'))
                    key, _ = generate_key_from_password(encryption_key, salt)
                    env[key] = decrypt_data(config[name][key], key)
                except Exception:
                    env[key] = '********'  # Mask password if decryption fails
            else:
                env[key] = '********'  # Mask password if no key provided
        else:
            env[key] = config[name][key]

    return env

def delete_environment(name: str) -> bool:
    """
    Delete an environment.

    Args:
        name: Environment name

    Returns:
        bool: True if successful, False otherwise
    """
    import configparser
    config = configparser.ConfigParser()
    config.read(variables.ENVS_FILE)

    if name not in config.sections():
        return False

    config.remove_section(name)

    with open(variables.ENVS_FILE, 'w') as f:
        config.write(f)

    return True

def test_environment(name: str, encryption_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Test database connection for an environment.

    Args:
        name: Environment name
        encryption_key: Encryption key for encrypted environments

    Returns:
        Dictionary with test results
    """
    try:
        connection_url = get_connection_url(name, encryption_key)
        from sqlalchemy import create_engine, text

        engine = create_engine(connection_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchall()

        return {
            'success': True,
            'message': f"Successfully connected to {name}"
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Failed to connect to {name}: {str(e)}"
        }

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
    connection_url = get_connection_url(environment, encryption_key)
    return execute_query(connection_url, query, params)

# Data Copy Functions

def copy_db_to_file(
    source_env: str,
    query: str,
    file_path: str,
    file_format: str = 'csv',
    encryption_key: Optional[str] = None,
    apply_masks: bool = True
) -> Dict[str, Any]:
    """
    Copy data from database to file.

    Args:
        source_env: Source environment name
        query: SQL query to execute
        file_path: Output file path
        file_format: Output file format (csv, json)
        encryption_key: Encryption key for encrypted environments
        apply_masks: Whether to apply masking rules

    Returns:
        Dictionary with operation results
    """
    try:
        # Get connection URL
        connection_url = get_connection_url(source_env, encryption_key)

        # Execute query
        data = execute_query(connection_url, query)

        # Apply masking if needed
        if apply_masks:
            data = apply_masking(data, source_env)

        # Write to file
        write_to_file(data, file_path, file_format.lower())

        return {
            'success': True,
            'message': f"Successfully copied {len(data)} records to {file_path}",
            'record_count': len(data)
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error copying data: {str(e)}"
        }

def copy_file_to_db(
    file_path: str,
    target_env: str,
    table: str,
    file_format: str = 'csv',
    mode: str = 'APPEND',
    batch_size: int = 1000,
    encryption_key: Optional[str] = None,
    validate_target: bool = False,
    create_if_not_exists: bool = False,
    apply_masks: bool = True
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
        encryption_key: Encryption key for encrypted environments
        validate_target: Whether to validate target table
        create_if_not_exists: Whether to create target table if it doesn't exist
        apply_masks: Whether to apply masking rules

    Returns:
        Dictionary with operation results
    """
    try:
        # Get connection URL
        connection_url = get_connection_url(target_env, encryption_key)

        # Read data from file
        if file_format.lower() == 'csv':
            data = pd.read_csv(file_path)
        elif file_format.lower() == 'json':
            data = pd.read_json(file_path)
        else:
            return {
                'success': False,
                'message': f"Unsupported file format: {file_format}"
            }

        # Apply masking if needed
        if apply_masks:
            data = apply_masking(data, target_env)

        # Validate target table if needed
        if validate_target:
            if not check_table_exists(connection_url, table):
                if create_if_not_exists:
                    # Create table logic would go here
                    pass
                else:
                    return {
                        'success': False,
                        'message': f"Table '{table}' does not exist in environment '{target_env}'"
                    }

        # Write to database
        write_to_db(data, connection_url, table, mode, batch_size)

        return {
            'success': True,
            'message': f"Successfully copied {len(data)} records to table '{table}'",
            'record_count': len(data)
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error copying data: {str(e)}"
        }

def copy_db_to_db(
    source_env: str,
    target_env: str,
    query: str,
    table: str,
    mode: str = 'APPEND',
    batch_size: int = 1000,
    parallel: int = 1,
    source_encryption_key: Optional[str] = None,
    target_encryption_key: Optional[str] = None,
    validate_target: bool = False,
    create_if_not_exists: bool = False,
    apply_masks: bool = True
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
        parallel: Number of parallel workers
        source_encryption_key: Encryption key for source environment
        target_encryption_key: Encryption key for target environment
        validate_target: Whether to validate target table
        create_if_not_exists: Whether to create target table if it doesn't exist
        apply_masks: Whether to apply masking rules

    Returns:
        Dictionary with operation results
    """
    try:
        # Get connection URLs
        source_url = get_connection_url(source_env, source_encryption_key)
        target_url = get_connection_url(target_env, target_encryption_key)

        # Validate target table if needed
        if validate_target:
            if not check_table_exists(target_url, table):
                if create_if_not_exists:
                    # Create table logic would go here
                    pass
                else:
                    return {
                        'success': False,
                        'message': f"Table '{table}' does not exist in environment '{target_env}'"
                    }

        # Execute query
        data = execute_query(source_url, query)

        # Apply masking if needed
        if apply_masks:
            data = apply_masking(data, target_env)

        # Write to database
        write_to_db(data, target_url, table, mode, batch_size)

        return {
            'success': True,
            'message': f"Successfully copied {len(data)} records to table '{table}'",
            'record_count': len(data)
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error copying data: {str(e)}"
        }

# Data Masking Functions

def add_mask(
    column: str,
    algorithm: str,
    environment: Optional[str] = None,
    length: Optional[int] = None
) -> bool:
    """
    Add a masking rule for a column.

    Args:
        column: Column name
        algorithm: Masking algorithm (star, star_length, random, nullify)
        environment: Environment name (None for global)
        length: Length parameter for algorithms that need it

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load existing definitions
        definitions = load_masking_definitions()

        # Prepare the masking configuration
        mask_config = {
            'algorithm': algorithm.lower(),
            'params': {}
        }

        # Add algorithm-specific parameters
        if length is not None:
            mask_config['params']['length'] = length

        # Add to global or environment-specific definitions
        if environment:
            if 'environments' not in definitions:
                definitions['environments'] = {}
            if environment not in definitions['environments']:
                definitions['environments'][environment] = {}
            definitions['environments'][environment][column] = mask_config
        else:
            if 'global' not in definitions:
                definitions['global'] = {}
            definitions['global'][column] = mask_config

        # Save the updated definitions
        return save_masking_definitions(definitions)
    except Exception:
        return False

def remove_mask(column: str, environment: Optional[str] = None) -> bool:
    """
    Remove a masking rule for a column.

    Args:
        column: Column name
        environment: Environment name (None for global)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load existing definitions
        definitions = load_masking_definitions()

        # Remove from global or environment-specific definitions
        if environment:
            if ('environments' in definitions and
                environment in definitions['environments'] and
                column in definitions['environments'][environment]):
                del definitions['environments'][environment][column]
        else:
            if 'global' in definitions and column in definitions['global']:
                del definitions['global'][column]

        # Save the updated definitions
        return save_masking_definitions(definitions)
    except Exception:
        return False

def list_masks(environment: Optional[str] = None) -> Dict[str, Any]:
    """
    List masking rules.

    Args:
        environment: Environment name (None for all)

    Returns:
        Dictionary with masking rules
    """
    # Load existing definitions
    definitions = load_masking_definitions()

    if environment:
        # Return only environment-specific rules
        env_rules = definitions.get('environments', {}).get(environment, {})
        return {'environment': environment, 'rules': env_rules}
    else:
        # Return all rules
        return definitions

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
    # Load existing definitions
    definitions = load_masking_definitions()

    # Create a DataFrame with the value
    df = pd.DataFrame({column: [value]})

    # Apply masking
    masked_df = apply_masking(df, environment)

    # Get the masked value
    masked_value = masked_df[column].iloc[0]

    return {
        'column': column,
        'original': value,
        'masked': masked_value,
        'environment': environment
    }

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
    # Parse columns if provided
    column_list = columns or []

    # Get schema from database if environment and table are provided
    if environment and table:
        # Get connection URL
        connection_url = get_connection_url(environment)

        # Check if table exists
        if not check_table_exists(connection_url, table):
            raise ValueError(f"Table '{table}' does not exist in environment '{environment}'")

        # Get table columns
        db_columns = get_table_columns(connection_url, table)
        if not db_columns:
            raise ValueError(f"Could not retrieve columns for table '{table}'")

        # Use table columns if no columns were specified
        if not column_list:
            column_list = db_columns

    # Generate random data
    return generate_random_data(
        num_records=num_records,
        columns=column_list,
        string_length=string_length,
        pattern=pattern or {},
        min_number=min_number,
        max_number=max_number,
        decimal_places=decimal_places,
        start_date=start_date,
        end_date=end_date,
        date_format=date_format
    )

def generate_and_save(
    num_records: int = 10,
    columns: Optional[List[str]] = None,
    environment: Optional[str] = None,
    table: Optional[str] = None,
    output: Optional[str] = None,
    format: str = 'csv',
    write_to_db: bool = False,
    mode: str = 'APPEND',
    **kwargs
) -> Dict[str, Any]:
    """
    Generate random data and save it to a file or database.

    Args:
        num_records: Number of records to generate
        columns: List of column names
        environment: Environment name to get table schema from
        table: Table name to get schema from
        output: Output file path
        format: Output file format (csv, json)
        write_to_db: Whether to write to database
        mode: Write mode (APPEND, REPLACE, FAIL)
        **kwargs: Additional parameters for generate_data

    Returns:
        Dictionary with operation results
    """
    try:
        # Generate data
        data = generate_data(
            num_records=num_records,
            columns=columns,
            environment=environment,
            table=table,
            **kwargs
        )

        # Write to database if requested
        if write_to_db:
            if not environment or not table:
                return {
                    'success': False,
                    'message': "Environment and table are required when writing to database"
                }

            # Get connection URL
            connection_url = get_connection_url(environment)

            # Write to database
            write_to_db(data, connection_url, table, mode)

            return {
                'success': True,
                'message': f"Successfully wrote {num_records} records to table '{table}'",
                'record_count': num_records,
                'data': data.head(5).to_dict('records')  # Return first 5 records as preview
            }

        # Write to file if output is provided
        elif output:
            # Write to file
            write_to_file(data, output, format.lower())

            return {
                'success': True,
                'message': f"Successfully wrote {num_records} records to file '{output}'",
                'record_count': num_records,
                'data': data.head(5).to_dict('records')  # Return first 5 records as preview
            }

        # Otherwise, just return the data
        else:
            return {
                'success': True,
                'message': f"Successfully generated {num_records} records",
                'record_count': num_records,
                'data': data.to_dict('records')
            }
    except Exception as e:
        return {
            'success': False,
            'message': f"Error generating data: {str(e)}"
        }
