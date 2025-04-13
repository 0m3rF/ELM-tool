import click
import os
import sys
import subprocess
import configparser
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from elm_utils import variables, encryption

config = configparser.ConfigParser()

# Database-specific dependencies
DB_PACKAGES = {
    "ORACLE": "cx_oracle",
    "MYSQL": "pymysql",
    "MSSQL": "pyodbc",
    "POSTGRES": "psycopg2-binary"
}

def ensure_db_driver_installed(db_type):
    """Ensure that the required database driver is installed"""
    if db_type not in DB_PACKAGES:
        return

    package_name = DB_PACKAGES[db_type]

    # Check if the package is already installed
    try:
        if package_name == "psycopg2-binary":
            __import__("psycopg2")
        else:
            __import__(package_name.replace('-', '_').split('>')[0])
        return  # Package is already installed
    except ImportError:
        # Package is not installed, try to install it
        print(f"Installing required database driver: {package_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Successfully installed {package_name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}: {str(e)}")
            print(f"Please install {package_name} manually using: pip install {package_name}")

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

class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name].name
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)

@click.group(cls=AliasedGroup)
def environment():
    """Environment management commands.

    This group contains commands for managing database environments.
    Use these commands to create, list, show, update, delete, and test
    database connection environments.

    Examples:

        List all available commands:
          elm environment --help

        List all environments:
          elm environment list
    """
    pass


@environment.command()
@click.argument('name')
@click.option("-h", "--host", required=True, help="Host of the environment")
@click.option("-p", "--port", required=True, help="Port of the environment", type=int)
@click.option("-u", "--user", required=True, help="User of the environment")
@click.option("-P", "--password", required=True, help="Password of the environment")
@click.option("-s", "--service", required=True, help="Service of the environment")
@click.option("-t", "--type", required=True, type=click.Choice(['ORACLE', 'POSTGRES', 'MYSQL', 'MSSQL'], case_sensitive=False),  help="Type of the environment")
@click.option("-o", "--overwrite", is_flag=True, default= False, help="Overwrite existing environment definition")
@click.option("-e", "--encrypt", is_flag=True, default= False, help="Encrypt sensitive environment information")
@click.option("-k", "--encryption-key", required=False, help="The key to use for encryption. Required if --encrypt is used. Unused if no encrypt has given.")
def create(name, host, port, user, password, service, type, overwrite, encrypt, encryption_key ):
    """Create a new environment.

    Examples:

        Create a PostgreSQL environment:
          elm environment create dev-pg --host localhost --port 5432 --user postgres --password password --service postgres --type postgres

        Create an Oracle environment:
          elm environment create prod-ora --host oraserver --port 1521 --user system --password oracle --service XE --type oracle

        Create an encrypted MySQL environment:
          elm environment create secure-mysql --host dbserver --port 3306 --user root --password secret --service mysql --type mysql --encrypt --encryption-key mypassword

        Create an environment and overwrite if it already exists:
          elm environment create dev-pg --host localhost --port 5432 --user postgres --password password --service postgres --type postgres --overwrite
    """

    if name == "*":
        raise click.UsageError("Cannot use '*' as environment name.")

    if encrypt and not encryption_key:
        # Raise an error if --encrypt is True but --encryption-key is missing
        raise click.UsageError("Option '--encryption-key' / '-k' is required when using '--encrypt' / '-e'.")

    # Check if environment file exists
    if( not os.path.isfile(variables.ENVS_FILE)): # create it if not exists
        with open(variables.ENVS_FILE, "w") as env_file:
            env_file.close()

    config.read(variables.ENVS_FILE)

    if name in config and not overwrite:
        raise click.UsageError("Name is already exists. To overwrite use '-o' / '--overwrite' existing environment definition.")
    else:
        config[name] = {}

    # Prepare environment data
    env_data = {
        "host": host,
        "port": str(port),
        "user": user,
        "password": password,
        "service": service,
        "type": type
    }

    if encrypt:
        # Encrypt the environment data
        encrypted_env = encryption.encrypt_environment(env_data, encryption_key)
        # Update config with encrypted data
        for key, value in encrypted_env.items():
            config[name][key] = value
    else:
        # Store unencrypted data
        config[name]["is_encrypted"] = 'False'
        for key, value in env_data.items():
            config[name][key] = value

    with open(variables.ENVS_FILE, 'w') as configfile:
        config.write(configfile)
        configfile.close()

    print("Environment created successfully")

@environment.command()
@click.option("-a", "--all", is_flag=True, default=False, help="Show all content of the environment")
@click.option("-h", "--host", is_flag=True, default=False, help="Show host of the environment")
@click.option("-p", "--port", is_flag=True, default=False, help="Show port of the environment")
@click.option("-u", "--user", is_flag=True, default=False, help="Show user of the environment")
@click.option("-P", "--password", is_flag=True, default=False, help="Show password of the environment")
@click.option("-s", "--service", is_flag=True, default=False, help="Show service of the environment")
@click.option("-t", "--type", is_flag=True, default=False, help="Show type of the environment")
def list(all, host, port, user, password, service, type):
    """List all environments.

    Examples:

        List all environments:
          elm environment list

        Show all details of all environments:
          elm environment list --all

        Show only host and port information:
          elm environment list --host --port

        Show specific information (user and service):
          elm environment list --user --service
    """
    config.read(variables.ENVS_FILE)

    if not config.sections():
        print("No environments defined.")
        return

    for envs in config.sections():
        is_encrypted = config[envs].get("is_encrypted", 'False') == 'True'

        if is_encrypted:
            print(f"[{envs}] (ENCRYPTED)")
        else:
            print(f"[{envs}]")
            if(host or all):
                print("host = " + config[envs]["host"])
            if(port  or all):
                print("port = " + config[envs]["port"])
            if(user or all):
                print("user = " + config[envs]["user"])
            if(password or all):
                print("password = " + config[envs]["password"])
            if(service or all):
                print("service = " + config[envs]["service"])
            if(type or all):
                print("type = " + config[envs]["type"])
        print("")

@environment.command()
@click.argument('name')
def delete(name):
    """Remove a system environment.

    Examples:

        Delete an environment:
          elm environment delete dev-pg

        Using the alias:
          elm environment rm old-env
    """
    config.read(variables.ENVS_FILE)

    if (not name in config.sections()):
        raise click.UsageError("Environment is not found")

    del config[name]

    with open(variables.ENVS_FILE, 'w') as configfile:
        config.write(configfile)
        configfile.close()


@environment.command()
@click.argument('name')
@click.option("-k", "--encryption-key", required=False, help="The key to decrypt the environment if it's encrypted")
def show(name, encryption_key):
    """Show a system environment.

    Examples:
        Show an environment:
          elm environment show dev-pg

        Show an encrypted environment:
          elm environment show secure-env --encryption-key mypassword

        Using the inspect alias:
          elm environment inspect dev-pg
    """
    config.read(variables.ENVS_FILE)

    if (not name in config.sections()):
        raise click.UsageError("Environment is not found")

    is_encrypted = config[name].get("is_encrypted", 'False') == 'True'

    if not is_encrypted:
        # Show unencrypted environment
        print(f"[{name}]")
        print("host = " + config[name]["host"])
        print("port = " + config[name]["port"])
        print("user = " + config[name]["user"])
        print("password = " + config[name]["password"])
        print("service = " + config[name]["service"])
        print("type = " + config[name]["type"])
    else:
        # Handle encrypted environment
        if not encryption_key:
            print(f"Environment '{name}' is encrypted. Provide an encryption key to view it.")
            return

        try:
            # Decrypt the environment
            decrypted_env = encryption.decrypt_environment(dict(config[name]), encryption_key)

            # Display decrypted environment
            print(f"[{name}] (Decrypted)")
            print("host = " + decrypted_env["host"])
            print("port = " + decrypted_env["port"])
            print("user = " + decrypted_env["user"])
            print("password = " + decrypted_env["password"])
            print("service = " + decrypted_env["service"])
            print("type = " + decrypted_env["type"])
        except Exception as e:
            print(f"Failed to decrypt environment: {str(e)}. Check your encryption key.")

@environment.command()
@click.argument('name')
@click.option("-h", "--host", required=False, help="Host of the environment")
@click.option("-p", "--port", required=False, help="Port of the environment", type=int)
@click.option("-u", "--user", required=False, help="User of the environment")
@click.option("-P", "--password", required=False, help="Password of the environment")
@click.option("-s", "--service", required=False, help="Service of the environment")
@click.option("-t", "--type", required=False, type=click.Choice(['ORACLE', 'POSTGRES', 'MYSQL', 'MSSQL'], case_sensitive=False), help="Type of the environment")
@click.option("-e", "--encrypt", is_flag=True, default=False, help="Encrypt the environment")
@click.option("-k", "--encryption-key", required=False, help="The key to use for encryption. Required if --encrypt is used.")
def update(name, host, port, user, password, service, type, encrypt, encryption_key):
    """Update a system environment.

    Examples:

        Update the host and port of an environment:
          elm environment update dev-pg --host new-host --port 5433

        Update the password:
          elm environment update prod-ora --password new-password

        Encrypt an existing environment:
          elm environment update dev-mysql --encrypt --encryption-key mypassword

        Update multiple fields at once:
          elm environment update dev-pg --host new-host --port 5433 --user new-user

        Using the edit alias:
          elm environment edit dev-pg --host new-host
    """
    # Check if encryption key is provided when encrypt flag is set
    if encrypt and not encryption_key:
        raise click.UsageError("Option '--encryption-key' / '-k' is required when using '--encrypt' / '-e'.")

    # Read the config file
    config.read(variables.ENVS_FILE)

    # Check if the environment exists
    if not name in config.sections():
        raise click.UsageError(f"Environment '{name}' not found")

    # Check if any field is provided to update
    if not any([host, port, user, password, service, type, encrypt]):
        raise click.UsageError("At least one field must be provided to update")

    # Get current encryption status
    was_encrypted = config[name].get("is_encrypted", 'False') == 'True'

    # Collect current environment data
    current_env = {}
    for field in ['host', 'port', 'user', 'password', 'service', 'type']:
        if field in config[name]:
            current_env[field] = config[name][field]

    # Collect updated fields
    updated_fields = {}
    if host:
        updated_fields['host'] = host
    if port:
        updated_fields['port'] = str(port)
    if user:
        updated_fields['user'] = user
    if password:
        updated_fields['password'] = password
    if service:
        updated_fields['service'] = service
    if type:
        updated_fields['type'] = type

    # Handle encryption scenarios
    if was_encrypted and not encrypt:
        # Decrypt first, then update, then re-encrypt
        if not encryption_key:
            raise click.UsageError("Option '--encryption-key' / '-k' is required to update an encrypted environment.")

        try:
            # Decrypt the environment
            decrypted_env = encryption.decrypt_environment(dict(config[name]), encryption_key)

            # Update the decrypted environment
            for field, value in updated_fields.items():
                decrypted_env[field] = value

            # Re-encrypt the environment
            encrypted_env = encryption.encrypt_environment(decrypted_env, encryption_key)

            # Update config with re-encrypted data
            for key, value in encrypted_env.items():
                config[name][key] = value

        except Exception as e:
            raise click.UsageError(f"Failed to decrypt environment: {str(e)}. Check your encryption key.")

    elif not was_encrypted and encrypt:
        # Encrypt the environment
        # Merge current environment with updates
        merged_env = current_env.copy()
        merged_env.update(updated_fields)

        # Encrypt the merged environment
        encrypted_env = encryption.encrypt_environment(merged_env, encryption_key)

        # Update config with encrypted data
        for key, value in encrypted_env.items():
            config[name][key] = value

    elif was_encrypted and encrypt:
        # Re-encrypt with a new key
        try:
            # Decrypt with old key
            decrypted_env = encryption.decrypt_environment(dict(config[name]), encryption_key)

            # Update decrypted environment
            for field, value in updated_fields.items():
                decrypted_env[field] = value

            # Re-encrypt with the same key (or could use a new key parameter)
            encrypted_env = encryption.encrypt_environment(decrypted_env, encryption_key)

            # Update config
            for key, value in encrypted_env.items():
                config[name][key] = value

        except Exception as e:
            raise click.UsageError(f"Failed to re-encrypt environment: {str(e)}. Check your encryption key.")

    else:  # not was_encrypted and not encrypt
        # Simple update of unencrypted environment
        for field, value in updated_fields.items():
            config[name][field] = value

    # Save the updated configuration
    with open(variables.ENVS_FILE, 'w') as configfile:
        config.write(configfile)
        configfile.close()

    print(f"Environment '{name}' updated successfully")

@environment.command()
@click.argument('name')
@click.option("-k", "--encryption-key", required=False, help="The key to decrypt the environment if it's encrypted")
def test(name, encryption_key=None):
    """Test a system environment by attempting to connect to the database.

    Examples:

        Test a database connection:
          elm environment test dev-pg

        Test an encrypted environment connection:
          elm environment test secure-mysql --encryption-key mypassword

        Using the validate alias:
          elm environment validate dev-pg
    """
    # Use the name directly as it's now a required argument
    env_name = name
    from sqlalchemy import create_engine
    from sqlalchemy.exc import SQLAlchemyError

    # Read the config file
    config.read(variables.ENVS_FILE)

    # Check if the environment exists
    if not env_name in config.sections():
        raise click.UsageError(f"Environment '{env_name}' not found")

    # Check if the environment is encrypted
    is_encrypted = config[env_name].get("is_encrypted", 'False') == 'True'

    # Get environment details
    if is_encrypted:
        if not encryption_key:
            raise click.UsageError(f"Environment '{env_name}' is encrypted. Provide an encryption key to test it.")

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

    # Ensure the required database driver is installed
    ensure_db_driver_installed(env_type)

    # Create connection URL based on database type
    connection_url = None
    if env_type == "ORACLE":
        # Oracle connection string format
        connection_url = f"oracle+cx_oracle://{user}:{password}@{host}:{port}/{service}"
    elif env_type == "POSTGRES":
        # PostgreSQL connection string format
        connection_url = f"postgresql://{user}:{password}@{host}:{port}/{service}"
    elif env_type == "MYSQL":
        # MySQL connection string format
        connection_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{service}"
    elif env_type == "MSSQL":
        # MSSQL connection string format
        connection_url = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{service}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        raise click.UsageError(f"Unsupported database type: {env_type}")

    print(f"Testing connection to {env_type} database at {host}:{port}...")

    try:
        # Create engine and attempt to connect
        engine = create_engine(connection_url)
        connection = engine.connect()
        connection.close()
        print(f"✓ Connection successful to {env_type} database at {host}:{port}")
    except SQLAlchemyError as e:
        print(f"✗ Connection failed: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False

    return True

@environment.command()
@click.argument('name')
@click.option("-q", "--query", required=True, help="SQL query to execute")
@click.option("-k", "--encryption-key", required=False, help="Encryption key for encrypted environments")
def execute(name, query, encryption_key):
    """Execute a SQL query on a database

    Examples:

        Execute a simple query:
          elm environment execute dev-pg --query "SELECT * FROM users LIMIT 10"

        Execute a query on an encrypted environment:
          elm environment execute secure-mysql --query "SHOW TABLES" --encryption-key mypassword

        Using the exec alias:
          elm environment exec dev-pg --query "SELECT COUNT(*) FROM orders"
    """
    try:
        # Get connection URL
        connection_url = get_connection_url(name, encryption_key)

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

ALIASES = {
    "new": create,
    "ls": list,
    "rm": delete,
    "remove": delete,
    "inspect": show,
    "edit": update,
    "validate": test,
    "exec": execute,
    "run": execute
}
