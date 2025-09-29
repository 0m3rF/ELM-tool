import os
import configparser
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from elm.elm_utils import variables, encryption


def _initialize_oracle_client():
    """Initialize Oracle client to handle thin mode issues."""
    try:
        import oracledb
        # Try to initialize Oracle client if not already done
        if not hasattr(oracledb, '_client_initialized'):
            try:
                oracledb.init_oracle_client()
                oracledb._client_initialized = True
                print("✓ Oracle thick mode activated successfully")
                return True
            except Exception as e:
                print(f"⚠ Failed to initialize Oracle thick mode: {str(e)}")
                # If initialization fails, continue with thin mode
                return False
        else:
            # Already initialized
            print("✓ Oracle thick mode already active")
            return True
    except ImportError:
        print("⚠ oracledb package not installed")
        return False


def _handle_oracle_connection_error(connection_url, original_error):
    """Handle Oracle connection errors by trying to initialize client."""
    error_str = str(original_error).lower()

    # Check for thin mode password verifier errors
    if any(keyword in error_str for keyword in [
        'password verifier type', 'dpy-3015', 'not supported by python-oracledb in thin mode',
        'thin mode', 'verifier', '0x939'
    ]):
        print(f"⚠ Detected Oracle thin mode compatibility issue: {str(original_error)}")
        print("🔄 Attempting to activate Oracle thick mode...")

        try:
            # Try to initialize Oracle client and retry connection
            if _initialize_oracle_client():
                print("🔄 Retrying connection with thick mode...")

                # Retry the connection with thick mode
                engine = create_engine(connection_url)
                with engine.connect() as connection:
                    result = connection.execute(text("SELECT 1 FROM DUAL"))
                    result.fetchall()
                print("✓ Connection successful with Oracle thick mode")
                return True
            else:
                print("✗ Failed to activate Oracle thick mode")
                return False
        except Exception as retry_error:
            print(f"✗ Connection failed even with thick mode: {str(retry_error)}")
            # If thick mode also fails, return False to raise the original error
            return False

    return False

# Read the environment configuration
config = configparser.ConfigParser()

def get_connection_url(env_name, encryption_key=None):
    """Get a SQLAlchemy connection URL for the specified environment"""
    config.read(variables.ENVS_FILE)

    # Check if the environment exists
    if not env_name in config.sections():
        raise ValueError(f"Environment '{env_name}' not found")

    # Check if the environment is encrypted
    is_encrypted = config[env_name].get("is_encrypted", 'False') == 'True'

    # Get environment details
    if is_encrypted:
        if not encryption_key:
            raise ValueError(f"Environment '{env_name}' is encrypted. Provide an encryption key.")

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
            raise ValueError(f"Failed to decrypt environment: {str(e)}. Check your encryption key.")
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
        # PostgreSQL connection string format
        return f"postgresql://{user}:{password}@{host}:{port}/{service}"
    elif env_type == "MYSQL":
        # MySQL connection string format
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{service}"
    elif env_type == "MSSQL":
        # MSSQL connection string format
        return f"mssql+pyodbc://{user}:{password}@{host}:{port}/{service}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        raise ValueError(f"Unsupported database type: {env_type}")

def check_table_exists(connection_url, table_name):
    """Check if a table exists in the database"""
    try:
        engine = create_engine(connection_url)
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
                    raise ValueError(f"Database error while checking table existence: {str(e)}")
                else:
                    raise ValueError(f"Error checking table existence: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            if isinstance(e, SQLAlchemyError):
                raise ValueError(f"Database error while checking table existence: {str(e)}")
            else:
                raise ValueError(f"Error checking table existence: {str(e)}")

def get_table_columns(connection_url, table_name):
    """Get the column names of a table"""
    try:
        engine = create_engine(connection_url)
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
                    raise ValueError(f"Database error while getting table columns: {str(e)}")
                else:
                    raise ValueError(f"Error getting table columns: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            if isinstance(e, SQLAlchemyError):
                raise ValueError(f"Database error while getting table columns: {str(e)}")
            else:
                raise ValueError(f"Error getting table columns: {str(e)}")

def execute_query(connection_url, query, batch_size=None, environment=None, apply_mask=True):
    """Execute a query and return the results"""
    try:
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            if batch_size:
                # Execute with batching
                result = pd.read_sql_query(query, connection, chunksize=batch_size)

                # For batched results, we'll apply masking when each batch is processed
                if not apply_mask:
                    return result  # This will be an iterator of DataFrames

                # Create a generator that applies masking to each batch
                def masked_batches():
                    for batch in result:
                        from elm.elm_utils.data_utils import apply_masking
                        yield apply_masking(batch, environment)

                return masked_batches()
            else:
                # Execute without batching
                result = pd.read_sql_query(query, connection)

                # Apply masking if requested
                if apply_mask:
                    from elm.elm_utils.data_utils import apply_masking
                    result = apply_masking(result, environment)

                return result
    except Exception as e:
        # Check if this is an Oracle connection and try to handle thin mode errors
        if 'oracle' in connection_url.lower():
            if _handle_oracle_connection_error(connection_url, e):
                # Retry the query after Oracle client initialization
                engine = create_engine(connection_url)
                with engine.connect() as connection:
                    if batch_size:
                        result = pd.read_sql_query(query, connection, chunksize=batch_size)
                        if not apply_mask:
                            return result
                        def masked_batches():
                            for batch in result:
                                from elm.elm_utils.data_utils import apply_masking
                                yield apply_masking(batch, environment)
                        return masked_batches()
                    else:
                        result = pd.read_sql_query(query, connection)
                        if apply_mask:
                            from elm.elm_utils.data_utils import apply_masking
                            result = apply_masking(result, environment)
                        return result
            else:
                # Re-raise the original error
                if isinstance(e, SQLAlchemyError):
                    raise ValueError(f"Database error: {str(e)}")
                else:
                    raise ValueError(f"Error executing query: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            if isinstance(e, SQLAlchemyError):
                raise ValueError(f"Database error: {str(e)}")
            else:
                raise ValueError(f"Error executing query: {str(e)}")

def write_to_db(data, connection_url, table_name, if_exists='append', batch_size=None):
    """Write data to a database table"""
    try:
        engine = create_engine(connection_url)

        if batch_size and len(data) > batch_size:
            # Write in batches
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                current_if_exists = if_exists if i == 0 else 'append'
                batch.to_sql(table_name, engine, if_exists=current_if_exists, index=False)
        else:
            # Write all at once
            data.to_sql(table_name, engine, if_exists=if_exists, index=False)

        return True
    except Exception as e:
        # Check if this is an Oracle connection and try to handle thin mode errors
        if 'oracle' in connection_url.lower():
            if _handle_oracle_connection_error(connection_url, e):
                # Retry the write after Oracle client initialization
                engine = create_engine(connection_url)

                if batch_size and len(data) > batch_size:
                    for i in range(0, len(data), batch_size):
                        batch = data.iloc[i:i+batch_size]
                        current_if_exists = if_exists if i == 0 else 'append'
                        batch.to_sql(table_name, engine, if_exists=current_if_exists, index=False)
                else:
                    data.to_sql(table_name, engine, if_exists=if_exists, index=False)

                return True
            else:
                # Re-raise the original error
                raise ValueError(f"Error writing to database: {str(e)}")
        else:
            # Not an Oracle connection, re-raise the error
            raise ValueError(f"Error writing to database: {str(e)}")

def write_to_file(data, file_path, file_format='csv', mode='w'):
    """Write data to a file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Write based on format
        if file_format.lower() == 'csv':
            # For CSV, handle append mode specially
            if mode == 'a' and os.path.exists(file_path):
                # Append without header
                data.to_csv(file_path, mode='a', header=False, index=False)
            else:
                # Write with header
                data.to_csv(file_path, index=False)
        elif file_format.lower() == 'json':
            # For JSON, handle append mode specially
            if mode == 'a' and os.path.exists(file_path):
                # Read existing JSON
                try:
                    existing_data = pd.read_json(file_path)
                    # Concatenate with new data
                    combined_data = pd.concat([existing_data, data], ignore_index=True)
                    # Write back
                    combined_data.to_json(file_path, orient='records', indent=2)
                except:
                    # If reading fails, just write the new data
                    data.to_json(file_path, orient='records', indent=2)
            else:
                # Write new JSON
                data.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return True
    except Exception as e:
        raise ValueError(f"Error writing to file: {str(e)}")
