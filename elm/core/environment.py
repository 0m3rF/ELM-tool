"""
ELM Tool Core Environment Management

Unified environment management operations for both CLI and API interfaces.
This module provides consistent environment CRUD operations and database connectivity.
"""

import base64
import configparser
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from elm.core.types import EnvironmentConfig, DatabaseType, OperationResult
from elm.core.exceptions import EnvironmentError, ValidationError, DatabaseError, EncryptionError
from elm.core.utils import (
    validate_database_type, load_environment_config, save_environment_config,
    create_success_result, create_error_result, handle_exception, validate_required_params
)
from elm.elm_utils import encryption


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
) -> OperationResult:
    """
    Create a new database environment.

    Args:
        name: Environment name
        host: Database host
        port: Database port
        user: Database username
        password: Database password
        service: Database service name (or SID for Oracle)
        db_type: Database type (ORACLE, POSTGRES, MYSQL, MSSQL)
        encrypt: Whether to encrypt the environment
        encryption_key: Encryption key (required if encrypt=True)
        overwrite: Whether to overwrite if environment already exists
        connection_type: Oracle connection type ('service_name' or 'sid'). Defaults to 'service_name'
    
    Returns:
        OperationResult with success status and message
    """
    try:
        # Validate inputs
        validate_required_params(
            {'name': name, 'host': host, 'port': port, 'user': user, 
             'password': password, 'service': service, 'db_type': db_type},
            ['name', 'host', 'port', 'user', 'password', 'service', 'db_type']
        )
        
        if name == "*":
            raise ValidationError("Cannot use '*' as environment name")
        
        # Validate database type
        db_type_enum = validate_database_type(db_type)
        
        # Validate encryption requirements
        if encrypt and not encryption_key:
            raise ValidationError("Encryption key is required when encrypt=True")
        
        # Load existing configuration
        config = load_environment_config()
        
        # Check if environment already exists
        if name in config.sections() and not overwrite:
            raise EnvironmentError(f"Environment '{name}' already exists. Use overwrite=True to replace it.")
        
        # Create or update the environment section
        if name not in config.sections():
            config.add_section(name)
        
        # Prepare environment data
        env_data = {
            "host": host,
            "port": str(port),
            "user": user,
            "password": password,
            "service": service,
            "type": db_type_enum.value
        }

        # Add Oracle-specific connection type if specified
        if db_type_enum == DatabaseType.ORACLE and connection_type:
            if connection_type.lower() not in ['service_name', 'sid']:
                raise ValidationError("Oracle connection_type must be 'service_name' or 'sid'")
            env_data["connection_type"] = connection_type.lower()
        elif db_type_enum == DatabaseType.ORACLE:
            # Default to service_name for Oracle if not specified
            env_data["connection_type"] = "service_name"
        
        # Handle encryption
        if encrypt:
            try:
                encrypted_env = encryption.encrypt_environment(env_data, encryption_key)
                for key, value in encrypted_env.items():
                    config[name][key] = value
            except Exception as e:
                raise EncryptionError(f"Failed to encrypt environment: {str(e)}")
        else:
            config[name]["is_encrypted"] = 'False'
            for key, value in env_data.items():
                config[name][key] = value
        
        # Save configuration
        save_environment_config(config)
        
        return create_success_result(f"Environment '{name}' created successfully")
        
    except Exception as e:
        return handle_exception(e, "environment creation")


def list_environments(show_all: bool = False) -> OperationResult:
    """
    List all environments.
    
    Args:
        show_all: Whether to show all details (passwords will be masked)
    
    Returns:
        OperationResult with list of environment dictionaries
    """
    try:
        config = load_environment_config()
        
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
        
        message = f"Found {len(environments)} environment(s)"
        return create_success_result(message, data=environments, record_count=len(environments))
        
    except Exception as e:
        return handle_exception(e, "environment listing")


def get_environment(name: str, encryption_key: Optional[str] = None) -> OperationResult:
    """
    Get details of a specific environment.
    
    Args:
        name: Environment name
        encryption_key: Encryption key for encrypted environments
    
    Returns:
        OperationResult with environment details
    """
    try:
        validate_required_params({'name': name}, ['name'])
        
        config = load_environment_config()
        
        if name not in config.sections():
            raise EnvironmentError(f"Environment '{name}' not found")
        
        env = {'name': name}
        is_encrypted = config[name].get('is_encrypted', 'False') == 'True'
        
        for key in config[name]:
            if key == 'password' and is_encrypted:
                if encryption_key:
                    try:
                        salt = base64.b64decode(config[name]['salt'].encode('utf-8'))
                        decrypt_key, _ = encryption.generate_key_from_password(encryption_key, salt)
                        env[key] = encryption.decrypt_data(config[name][key], decrypt_key)
                    except Exception:
                        env[key] = '********'  # Mask password if decryption fails
                else:
                    env[key] = '********'  # Mask password if no key provided
            else:
                env[key] = config[name][key]
        
        return create_success_result(f"Environment '{name}' retrieved successfully", data=env)
        
    except Exception as e:
        return handle_exception(e, "environment retrieval")


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
) -> OperationResult:
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
        OperationResult with success status and message
    """
    try:
        validate_required_params({'name': name}, ['name'])

        # Check if any field is provided to update
        update_fields = {k: v for k, v in {
            'host': host, 'port': port, 'user': user, 'password': password,
            'service': service, 'db_type': db_type
        }.items() if v is not None}

        if not update_fields and encrypt is None:
            raise ValidationError("At least one field must be provided to update")

        # Validate encryption requirements
        if encrypt and not encryption_key:
            raise ValidationError("Encryption key is required when encrypt=True")

        # Load configuration
        config = load_environment_config()

        if name not in config.sections():
            raise EnvironmentError(f"Environment '{name}' not found")

        # Get current encryption status
        was_encrypted = config[name].get('is_encrypted', 'False') == 'True'

        # Validate database type if provided
        if 'db_type' in update_fields:
            update_fields['db_type'] = validate_database_type(update_fields['db_type']).value

        # Handle different encryption scenarios
        if was_encrypted:
            if not encryption_key:
                raise ValidationError("Encryption key is required to update an encrypted environment")

            try:
                # Decrypt current environment
                decrypted_env = encryption.decrypt_environment(dict(config[name]), encryption_key)

                # Update fields
                for field, value in update_fields.items():
                    if field == 'port':
                        decrypted_env[field] = str(value)
                    elif field == 'db_type':
                        decrypted_env['type'] = value
                    else:
                        decrypted_env[field] = value

                # Re-encrypt
                encrypted_env = encryption.encrypt_environment(decrypted_env, encryption_key)
                for key, value in encrypted_env.items():
                    config[name][key] = value

            except Exception as e:
                raise EncryptionError(f"Failed to update encrypted environment: {str(e)}")

        else:
            # Update unencrypted environment
            for field, value in update_fields.items():
                if field == 'port':
                    config[name][field] = str(value)
                elif field == 'db_type':
                    config[name]['type'] = value
                else:
                    config[name][field] = value

            # Handle encryption request
            if encrypt:
                # Collect all current data
                current_env = {}
                for field in ['host', 'port', 'user', 'password', 'service', 'type']:
                    if field in config[name]:
                        current_env[field] = config[name][field]

                # Encrypt the environment
                try:
                    encrypted_env = encryption.encrypt_environment(current_env, encryption_key)
                    for key, value in encrypted_env.items():
                        config[name][key] = value
                except Exception as e:
                    raise EncryptionError(f"Failed to encrypt environment: {str(e)}")

        # Save configuration
        save_environment_config(config)

        return create_success_result(f"Environment '{name}' updated successfully")

    except Exception as e:
        return handle_exception(e, "environment update")


def delete_environment(name: str) -> OperationResult:
    """
    Delete an environment.

    Args:
        name: Environment name

    Returns:
        OperationResult with success status and message
    """
    try:
        validate_required_params({'name': name}, ['name'])

        config = load_environment_config()

        if name not in config.sections():
            raise EnvironmentError(f"Environment '{name}' not found")

        config.remove_section(name)
        save_environment_config(config)

        return create_success_result(f"Environment '{name}' deleted successfully")

    except Exception as e:
        return handle_exception(e, "environment deletion")


def test_environment(name: str, encryption_key: Optional[str] = None) -> OperationResult:
    """
    Test database connection for an environment.

    Args:
        name: Environment name
        encryption_key: Encryption key for encrypted environments

    Returns:
        OperationResult with test results
    """
    try:
        validate_required_params({'name': name}, ['name'])

        connection_url = get_connection_url(name, encryption_key)

        engine = create_engine(connection_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchall()

        return create_success_result(f"Successfully connected to environment '{name}'")

    except Exception as e:
        return handle_exception(e, "environment connection test")


def execute_sql(
    environment: str,
    query: str,
    encryption_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None
) -> OperationResult:
    """
    Execute SQL query on an environment.

    Args:
        environment: Environment name
        query: SQL query to execute
        encryption_key: Encryption key for encrypted environments
        params: Query parameters

    Returns:
        OperationResult with query results
    """
    try:
        validate_required_params({'environment': environment, 'query': query},
                                ['environment', 'query'])

        connection_url = get_connection_url(environment, encryption_key)

        engine = create_engine(connection_url)
        with engine.connect() as connection:
            if params:
                result = connection.execute(text(query), params)
            else:
                result = connection.execute(text(query))

            if result.returns_rows:
                import pandas as pd
                df = pd.DataFrame(result.fetchall())
                if not df.empty:
                    df.columns = result.keys()
                    return create_success_result(
                        "Query executed successfully",
                        data=df.to_dict('records'),
                        record_count=len(df)
                    )
                else:
                    return create_success_result("Query executed successfully. No rows returned.")
            else:
                return create_success_result("Query executed successfully. No result set.")

    except Exception as e:
        return handle_exception(e, "SQL execution")


def get_connection_url(env_name: str, encryption_key: Optional[str] = None) -> str:
    """
    Get a SQLAlchemy connection URL for the specified environment.

    Args:
        env_name: Environment name
        encryption_key: Encryption key for encrypted environments

    Returns:
        SQLAlchemy connection URL string

    Raises:
        EnvironmentError: If environment not found or decryption fails
        ValidationError: If encryption key required but not provided
    """
    config = load_environment_config()

    if env_name not in config.sections():
        raise EnvironmentError(f"Environment '{env_name}' not found")

    # Check if the environment is encrypted
    is_encrypted = config[env_name].get("is_encrypted", 'False') == 'True'

    # Get environment details
    if is_encrypted:
        if not encryption_key:
            raise ValidationError(f"Environment '{env_name}' is encrypted. Provide an encryption key.")

        try:
            # Decrypt the environment
            decrypted_env = encryption.decrypt_environment(dict(config[env_name]), encryption_key)

            # Get decrypted details
            env_type = decrypted_env["type"].upper()
            host = decrypted_env["host"]
            port = decrypted_env["port"]
            user = decrypted_env["user"]
            password = decrypted_env["password"]
            service = decrypted_env["service"]
            connection_type = decrypted_env.get("connection_type", "service_name")
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt environment: {str(e)}. Check your encryption key.")
    else:
        # Get unencrypted details
        env_type = config[env_name]["type"].upper()
        host = config[env_name]["host"]
        port = config[env_name]["port"]
        user = config[env_name]["user"]
        password = config[env_name]["password"]
        service = config[env_name]["service"]
        connection_type = config[env_name].get("connection_type", "service_name")

    # Create connection URL based on database type
    if env_type == "ORACLE":
        # Handle Oracle connection types: SID vs service_name
        if connection_type == "sid":
            # For SID connections, use the format: oracle+oracledb://user:password@host:port/sid
            return f"oracle+oracledb://{user}:{password}@{host}:{port}/{service}"
        else:
            # For service_name connections, use the format: oracle+oracledb://user:password@host:port?service_name=service
            return f"oracle+oracledb://{user}:{password}@{host}:{port}?service_name={service}"
    elif env_type == "POSTGRES":
        return f"postgresql://{user}:{password}@{host}:{port}/{service}"
    elif env_type == "MYSQL":
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{service}"
    elif env_type == "MSSQL":
        return f"mssql+pyodbc://{user}:{password}@{host}:{port}/{service}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        raise ValidationError(f"Unsupported database type: {env_type}")
