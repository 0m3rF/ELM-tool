import click
import os
import configparser
import concurrent.futures
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from elm_utils import variables, encryption

# Read the environment configuration
config = configparser.ConfigParser()

class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name].name
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)

@click.group(cls=AliasedGroup)
def copy():
    """Data copy commands for database operations"""
    pass

def get_connection_url(env_name, encryption_key=None):
    """Get a SQLAlchemy connection URL for the specified environment"""
    config.read(variables.ENVS_FILE)

    # Check if the environment exists
    if not env_name in config.sections():
        raise click.UsageError(f"Environment '{env_name}' not found")

    # Check if the environment is encrypted
    is_encrypted = config[env_name].get("is_encrypted", 'False') == 'True'

    # Get environment details
    if is_encrypted:
        if not encryption_key:
            raise click.UsageError(f"Environment '{env_name}' is encrypted. Provide an encryption key.")

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
        except Exception as e:
            raise click.UsageError(f"Failed to decrypt environment: {str(e)}. Check your encryption key.")
    else:
        # Get unencrypted details
        env_type = config[env_name]["type"].upper()
        host = config[env_name]["host"]
        port = config[env_name]["port"]
        user = config[env_name]["user"]
        password = config[env_name]["password"]
        service = config[env_name]["service"]

    # Create connection URL based on database type
    if env_type == "ORACLE":
        # Oracle connection string format
        return f"oracle+cx_oracle://{user}:{password}@{host}:{port}/{service}"
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
        raise click.UsageError(f"Unsupported database type: {env_type}")

def execute_query(connection_url, query, batch_size=None):
    """Execute a query and return the results"""
    try:
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            if batch_size:
                # Execute with batching
                result = pd.read_sql_query(query, connection, chunksize=batch_size)
                return result  # This will be an iterator of DataFrames
            else:
                # Execute without batching
                result = pd.read_sql_query(query, connection)
                return result
    except SQLAlchemyError as e:
        raise click.UsageError(f"Database error: {str(e)}")
    except Exception as e:
        raise click.UsageError(f"Error executing query: {str(e)}")

def write_to_file(data, file_path, file_format='csv', mode='w'):
    """Write data to a file in the specified format"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if isinstance(data, pd.DataFrame):
        if file_format.lower() == 'csv':
            data.to_csv(file_path, index=False, mode=mode)
        elif file_format.lower() == 'json':
            if mode == 'a' and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                # Append to existing JSON (this is tricky, we'll need to read, append, and write)
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
                    raise click.UsageError(f"Error appending to JSON: {str(e)}")
            else:
                # Write new JSON file
                data.to_json(file_path, orient='records', indent=2)
        else:
            raise click.UsageError(f"Unsupported file format: {file_format}")
    else:
        raise click.UsageError("Data must be a pandas DataFrame")

def read_from_file(file_path, file_format='csv'):
    """Read data from a file in the specified format"""
    if not os.path.exists(file_path):
        raise click.UsageError(f"File not found: {file_path}")

    try:
        if file_format.lower() == 'csv':
            return pd.read_csv(file_path)
        elif file_format.lower() == 'json':
            return pd.read_json(file_path, orient='records')
        else:
            raise click.UsageError(f"Unsupported file format: {file_format}")
    except Exception as e:
        raise click.UsageError(f"Error reading file: {str(e)}")

def write_to_db(data, connection_url, table_name, if_exists='append', batch_size=None):
    """Write data to a database table"""
    try:
        engine = create_engine(connection_url)

        if batch_size and len(data) > batch_size:
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                batch.to_sql(table_name, engine, if_exists=if_exists, index=False)
                # After first batch, we always append
                if_exists = 'append'
        else:
            # Process all at once
            data.to_sql(table_name, engine, if_exists=if_exists, index=False)

        return True
    except SQLAlchemyError as e:
        raise click.UsageError(f"Database error: {str(e)}")
    except Exception as e:
        raise click.UsageError(f"Error writing to database: {str(e)}")

def process_in_parallel(func, items, max_workers):
    """Process items in parallel using the provided function"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                click.echo(f"Error processing {item}: {str(e)}")
    return results

@copy.command()
@click.option("-s", "--source", required=True, help="Source environment name")
@click.option("-q", "--query", required=True, help="SQL query to extract data")
@click.option("-f", "--file", required=True, help="Output file path")
@click.option("-t", "--format", type=click.Choice(['CSV', 'JSON'], case_sensitive=False), default='CSV', help="Output file format")
@click.option("-m", "--mode", type=click.Choice(['OVERWRITE', 'APPEND'], case_sensitive=False), default='OVERWRITE', help="File write mode")
@click.option("-b", "--batch-size", type=int, default=None, help="Batch size for processing large datasets")
@click.option("-p", "--parallel", type=int, default=1, help="Number of parallel processes")
@click.option("-k", "--encryption-key", required=False, help="Encryption key for encrypted environments")
def db2file(source, query, file, format, mode, batch_size, parallel, encryption_key):
    """Copy data from database to file"""
    try:
        # Get connection URL
        connection_url = get_connection_url(source, encryption_key)

        # Set file mode
        file_mode = 'w' if mode.upper() == 'OVERWRITE' else 'a'

        if parallel > 1 and batch_size:
            # Execute query to get total count for pagination
            engine = create_engine(connection_url)
            with engine.connect() as conn:
                # This is a simplified approach - in a real scenario, you'd need to handle this more carefully
                count_query = f"SELECT COUNT(*) as count FROM ({query}) as subquery"
                try:
                    count_result = pd.read_sql(count_query, conn)
                    total_count = count_result['count'].iloc[0]
                except:
                    click.echo("Could not determine total count, using single batch")
                    total_count = batch_size

                # Create batched queries
                batched_queries = []
                for i in range(0, total_count, batch_size):
                    # This pagination approach works for some databases but not all
                    # You'd need to adapt this based on the specific database type
                    paginated_query = f"{query} LIMIT {batch_size} OFFSET {i}"
                    batched_queries.append((paginated_query, i))

                # Process batches in parallel
                def process_batch(batch_info):
                    batch_query, offset = batch_info
                    batch_data = execute_query(connection_url, batch_query)
                    batch_file = f"{file}.part{offset}" if offset > 0 else file
                    write_to_file(batch_data, batch_file, format.lower(), 'w')
                    return batch_file

                batch_files = process_in_parallel(process_batch, batched_queries, parallel)

                # Combine files if needed
                if format.lower() == 'csv' and len(batch_files) > 1:
                    # Combine CSV files
                    with open(file, 'w') as outfile:
                        # Write header from first file
                        with open(batch_files[0], 'r') as firstfile:
                            header = firstfile.readline()
                            outfile.write(header)

                        # Write data from all files (skipping headers)
                        for batch_file in batch_files:
                            with open(batch_file, 'r') as infile:
                                next(infile)  # Skip header
                                for line in infile:
                                    outfile.write(line)

                            # Remove part file
                            os.remove(batch_file)

                click.echo(f"Data exported to {file} successfully")
        else:
            # Simple execution without parallelism
            result = execute_query(connection_url, query, batch_size)

            if batch_size:
                # Handle batched results
                first_batch = True
                for chunk in result:
                    current_mode = file_mode if first_batch else 'a'
                    write_to_file(chunk, file, format.lower(), current_mode)
                    first_batch = False
            else:
                # Handle single result
                write_to_file(result, file, format.lower(), file_mode)

            click.echo(f"Data exported to {file} successfully")

    except Exception as e:
        click.echo(f"Error: {str(e)}")

@copy.command()
@click.option("-s", "--source", required=True, help="Source file path")
@click.option("-t", "--target", required=True, help="Target environment name")
@click.option("-a", "--table", required=True, help="Target table name")
@click.option("-f", "--format", type=click.Choice(['CSV', 'JSON'], case_sensitive=False), default='CSV', help="Input file format")
@click.option("-m", "--mode", type=click.Choice(['APPEND', 'REPLACE', 'FAIL'], case_sensitive=False), default='APPEND', help="Table write mode")
@click.option("-b", "--batch-size", type=int, default=None, help="Batch size for processing large datasets")
@click.option("-p", "--parallel", type=int, default=1, help="Number of parallel processes")
@click.option("-k", "--encryption-key", required=False, help="Encryption key for encrypted environments")
def file2db(source, target, table, format, mode, batch_size, parallel, encryption_key):
    """Copy data from file to database"""
    try:
        # Get connection URL
        connection_url = get_connection_url(target, encryption_key)

        # Read data from file
        data = read_from_file(source, format.lower())

        # Map mode to SQLAlchemy if_exists parameter
        if_exists_map = {
            'APPEND': 'append',
            'REPLACE': 'replace',
            'FAIL': 'fail'
        }
        if_exists = if_exists_map[mode.upper()]

        if parallel > 1 and batch_size and len(data) > batch_size:
            # Split data into batches
            batches = []
            for i in range(0, len(data), batch_size):
                batches.append(data.iloc[i:i+batch_size])

            # Process batches in parallel
            def process_batch(batch):
                return write_to_db(batch, connection_url, table, if_exists, None)

            process_in_parallel(process_batch, batches, parallel)
        else:
            # Process without parallelism
            write_to_db(data, connection_url, table, if_exists, batch_size)

        click.echo(f"Data imported to table {table} successfully")

    except Exception as e:
        click.echo(f"Error: {str(e)}")

@copy.command()
@click.option("-s", "--source", required=True, help="Source environment name")
@click.option("-t", "--target", required=True, help="Target environment name")
@click.option("-q", "--query", required=True, help="SQL query to extract data from source")
@click.option("-a", "--table", required=True, help="Target table name")
@click.option("-m", "--mode", type=click.Choice(['APPEND', 'REPLACE', 'FAIL'], case_sensitive=False), default='APPEND', help="Table write mode")
@click.option("-b", "--batch-size", type=int, default=None, help="Batch size for processing large datasets")
@click.option("-p", "--parallel", type=int, default=1, help="Number of parallel processes")
@click.option("-sk", "--source-key", required=False, help="Encryption key for source environment")
@click.option("-tk", "--target-key", required=False, help="Encryption key for target environment")
def db2db(source, target, query, table, mode, batch_size, parallel, source_key, target_key):
    """Copy data from one database to another"""
    try:
        # Get connection URLs
        source_url = get_connection_url(source, source_key)
        target_url = get_connection_url(target, target_key)

        # Map mode to SQLAlchemy if_exists parameter
        if_exists_map = {
            'APPEND': 'append',
            'REPLACE': 'replace',
            'FAIL': 'fail'
        }
        if_exists = if_exists_map[mode.upper()]

        if parallel > 1 and batch_size:
            # Execute query to get total count for pagination
            engine = create_engine(source_url)
            with engine.connect() as conn:
                # This is a simplified approach - in a real scenario, you'd need to handle this more carefully
                count_query = f"SELECT COUNT(*) as count FROM ({query}) as subquery"
                try:
                    count_result = pd.read_sql(count_query, conn)
                    total_count = count_result['count'].iloc[0]
                except:
                    click.echo("Could not determine total count, using single batch")
                    total_count = batch_size

                # Create batched queries
                batched_queries = []
                for i in range(0, total_count, batch_size):
                    # This pagination approach works for some databases but not all
                    # You'd need to adapt this based on the specific database type
                    paginated_query = f"{query} LIMIT {batch_size} OFFSET {i}"
                    batched_queries.append(paginated_query)

                # Process batches in parallel
                def process_batch(batch_query):
                    batch_data = execute_query(source_url, batch_query)
                    current_if_exists = if_exists if batched_queries.index(batch_query) == 0 else 'append'
                    write_to_db(batch_data, target_url, table, current_if_exists)
                    return len(batch_data)

                results = process_in_parallel(process_batch, batched_queries, parallel)
                total_rows = sum(results)
                click.echo(f"Copied {total_rows} rows to table {table} successfully")
        else:
            # Simple execution without parallelism
            result = execute_query(source_url, query, batch_size)

            if batch_size:
                # Handle batched results
                first_batch = True
                total_rows = 0
                for chunk in result:
                    current_if_exists = if_exists if first_batch else 'append'
                    write_to_db(chunk, target_url, table, current_if_exists)
                    total_rows += len(chunk)
                    first_batch = False
                click.echo(f"Copied {total_rows} rows to table {table} successfully")
            else:
                # Handle single result
                write_to_db(result, target_url, table, if_exists)
                click.echo(f"Copied {len(result)} rows to table {table} successfully")

    except Exception as e:
        click.echo(f"Error: {str(e)}")

@copy.command()
@click.option("-e", "--environment", required=True, help="Environment name")
@click.option("-q", "--query", required=True, help="SQL query to execute")
@click.option("-k", "--encryption-key", required=False, help="Encryption key for encrypted environments")
def execute(environment, query, encryption_key):
    """Execute a SQL query on a database"""
    try:
        # Get connection URL
        connection_url = get_connection_url(environment, encryption_key)

        # Execute the query
        engine = create_engine(connection_url)
        with engine.connect() as connection:
            result = connection.execute(text(query))
            if result.returns_rows:
                # Convert result to DataFrame for display
                df = pd.DataFrame(result.fetchall())
                if not df.empty:
                    df.columns = result.keys()
                    click.echo(df.to_string(index=False))
                else:
                    click.echo("Query executed successfully. No rows returned.")
            else:
                click.echo("Query executed successfully. No result set.")

    except Exception as e:
        click.echo(f"Error: {str(e)}")

# Define command aliases
ALIASES = {
    "d2f": db2file,
    "f2d": file2db,
    "d2d": db2db,
    "exec": execute,
    "run": execute
}