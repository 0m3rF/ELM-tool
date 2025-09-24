import click
import sys
import subprocess
import pandas as pd
from elm.core import environment as core_env
from elm.elm_utils.command_utils import AliasedGroup

# Database-specific dependencies
DB_PACKAGES = {
    "ORACLE": "oracledb",
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

@click.group(cls=AliasedGroup)
@click.help_option('-h', '--help')
def environment():
    """Environment management commands.

    This group contains commands for managing database environments.
    Use these commands to create, list, show, update, delete, and test
    database connection environments.

    Examples:

        List all available commands:
          elm-tool environment --help

        List all environments:
          elm-tool environment list
    """
    pass


@environment.command()
@click.argument('name')
@click.option("-h", "--host", required=False, help="Host of the environment")
@click.option("-p", "--port", required=False, help="Port of the environment", type=int)
@click.option("-u", "--user", required=False, help="User of the environment")
@click.option("-P", "--password", required=False, help="Password of the environment")
@click.option("-s", "--service", required=False, help="Service of the environment")
@click.option("-d", "--database", required=False, type=click.Choice(['ORACLE', 'POSTGRES', 'MYSQL', 'MSSQL'], case_sensitive=False),  help="Database type of the environment")
@click.option("-o", "--overwrite", is_flag=True, default= False, help="Overwrite existing environment definition")
@click.option("-e", "--encrypt", is_flag=True, default= False, help="Encrypt sensitive environment information")
@click.option("-k", "--encryption-key", required=False, help="The key to use for encryption. Required if --encrypt is used. Unused if no encrypt has given.")
@click.option("-c", "--connection-type", required=False, type=click.Choice(['service_name', 'sid'], case_sensitive=False), help="Oracle connection type: 'service_name' (default) or 'sid'. Only applies to Oracle databases.")
@click.option("-U", "--user-input", "-i", "--interactive", "--input", "--prompt", is_flag=True, default=False, help="Get input from user with prompts.")
@click.help_option('--help')
def create(name, host, port, user, password, service, database, overwrite, encrypt, encryption_key, connection_type, user_input):
    """Create a new environment.

    Examples:

        Create a PostgreSQL environment:
          elm-tool environment create dev-pg --host localhost --port 5432 --user postgres --password password --service postgres --database postgres

        Create an Oracle environment with service name:
          elm-tool environment create prod-ora --host oraserver --port 1521 --user system --password oracle --service XE --database oracle --connection-type service_name

        Create an Oracle environment with SID:
          elm-tool environment create prod-ora-sid --host oraserver --port 1521 --user system --password oracle --service ORCL --database oracle --connection-type sid

        Create an encrypted MySQL environment:
          elm-tool environment create secure-mysql --host dbserver --port 3306 --user root --password secret --service mysql --database mysql --encrypt --encryption-key mypassword

        Create an environment and overwrite if it already exists:
          elm-tool environment create dev-pg --host localhost --port 5432 --user postgres --password password --service postgres --database postgres --overwrite

        Create an environment with user input prompts:
          elm-tool environment create dev-pg --user-input
    """
    # Handle user input mode
    if user_input:
        # Prompt for all required fields if not provided
        if not host:
            host = click.prompt("Host", type=str)
        if not port:
            port = click.prompt("Port", type=int)
        if not user:
            user = click.prompt("User", type=str)
        if not password:
            password = click.prompt("Password", type=str, hide_input=True, confirmation_prompt=True, prompt_suffix=": ")
        if not service:
            service = click.prompt("Service", type=str)
        if not database:
            database = click.prompt("Database", type=click.Choice(['ORACLE', 'POSTGRES', 'MYSQL', 'MSSQL'], case_sensitive=False))
        if not encrypt:
            encrypt = click.confirm("Encrypt environment?", default=False)
        if encrypt and not encryption_key:
            encryption_key = click.prompt("Encryption key", type=str, hide_input=True, confirmation_prompt=True, prompt_suffix=": ")
        if not connection_type and database.upper() == 'ORACLE':
            connection_type = click.prompt("Oracle connection type", type=click.Choice(['service_name', 'sid'], case_sensitive=False), default='service_name')
    else:
        # Validate required fields when not in user input mode
        if not all([host, port, user, password, service, database]):
            missing_fields = []
            if not host: missing_fields.append("host")
            if not port: missing_fields.append("port")
            if not user: missing_fields.append("user")
            if not password: missing_fields.append("password")
            if not service: missing_fields.append("service")
            if not database: missing_fields.append("database")
            raise click.UsageError(f"Missing required fields: {', '.join(missing_fields)}. Use --user-input flag to be prompted for values.")

    if encrypt and not encryption_key:
        # Raise an error if --encrypt is True but --encryption-key is missing
        raise click.UsageError("Option '--encryption-key' / '-k' is required when using '--encrypt' / '-e'.")

    # Use core module to create environment
    result = core_env.create_environment(
        name=name,
        host=host,
        port=port,
        user=user,
        password=password,
        service=service,
        db_type=database,
        encrypt=encrypt,
        encryption_key=encryption_key,
        overwrite=overwrite,
        connection_type=connection_type
    )

    if result.success:
        click.echo("Environment created successfully")
    else:
        raise click.UsageError(result.message)

@environment.command()
@click.option("-a", "--all", is_flag=True, default=False, help="Show all content of the environment")
@click.option("-h", "--host", is_flag=True, default=False, help="Show host of the environment")
@click.option("-p", "--port", is_flag=True, default=False, help="Show port of the environment")
@click.option("-u", "--user", is_flag=True, default=False, help="Show user of the environment")
@click.option("-P", "--password", is_flag=True, default=False, help="Show password of the environment")
@click.option("-s", "--service", is_flag=True, default=False, help="Show service of the environment")
@click.option("-d", "--database", is_flag=True, default=False, help="Show database type of the environment")
@click.help_option('--help')
def list(all, host, port, user, password, service, database):
    """List all environments.

    Examples:

        List all environments:
          elm-tool environment list

        Show all details of all environments:
          elm-tool environment list --all

        Show only host and port information:
          elm-tool environment list --host --port

        Show specific information (user and service):
          elm-tool environment list --user --service
    """
    # Use core module to list environments
    result = core_env.list_environments(show_all=all)

    if not result.success:
        click.echo(f"Error: {result.message}")
        return

    environments = result.data

    if not environments:
        click.echo("No environments defined.")
        return

    for env in environments:
        env_name = env['name']
        is_encrypted = env.get("is_encrypted", 'False') == 'True'

        if is_encrypted:
            click.echo(f"[{env_name}] (ENCRYPTED)")
        else:
            click.echo(f"[{env_name}]")
            if all:
                # Show all fields when --all is specified
                for key, value in env.items():
                    if key != 'name':
                        click.echo(f"{key} = {value}")
            else:
                # Show specific fields based on flags
                if host and 'host' in env:
                    click.echo(f"host = {env['host']}")
                if port and 'port' in env:
                    click.echo(f"port = {env['port']}")
                if user and 'user' in env:
                    click.echo(f"user = {env['user']}")
                if password and 'password' in env:
                    click.echo(f"password = {env['password']}")
                if service and 'service' in env:
                    click.echo(f"service = {env['service']}")
                if database and 'type' in env:
                    click.echo(f"type = {env['type']}")
        click.echo("")

@environment.command()
@click.argument('name')
@click.help_option('-h', '--help')
def delete(name):
    """Remove a system environment.

    Examples:

        Delete an environment:
          elm-tool environment delete dev-pg

        Using the alias:
          elm-tool environment rm old-env
    """
    result = core_env.delete_environment(name=name)

    if result.success:
        click.echo(f"Environment '{name}' deleted successfully")
    else:
        raise click.UsageError(result.message)

@environment.command()
@click.argument('name')
@click.option("-k", "--encryption-key", required=False, help="The key to decrypt the environment if it's encrypted")
@click.help_option('-h', '--help')
def show(name, encryption_key):
    """Show a system environment.

    Examples:
        Show an environment:
          elm-tool environment show dev-pg

        Show an encrypted environment:
          elm-tool environment show secure-env --encryption-key mypassword

        Using the inspect alias:
          elm-tool environment inspect dev-pg
    """
    result = core_env.get_environment(name=name, encryption_key=encryption_key)

    if not result.success:
        raise click.UsageError(result.message)

    env = result.data
    is_encrypted = env.get("is_encrypted", 'False') == 'True'

    if is_encrypted and encryption_key:
        click.echo(f"[{name}] (Decrypted)")
    else:
        click.echo(f"[{name}]")

    # Display environment details
    for key, value in env.items():
        if key != 'name':
            click.echo(f"{key} = {value}")

@environment.command()
@click.argument('name')
@click.option("-h", "--host", required=False, help="Host of the environment")
@click.option("-p", "--port", required=False, help="Port of the environment", type=int)
@click.option("-u", "--user", required=False, help="User of the environment")
@click.option("-P", "--password", required=False, help="Password of the environment")
@click.option("-s", "--service", required=False, help="Service of the environment")
@click.option("-d", "--database", required=False, type=click.Choice(['ORACLE', 'POSTGRES', 'MYSQL', 'MSSQL'], case_sensitive=False), help="Database type of the environment")
@click.option("-e", "--encrypt", is_flag=True, default=False, help="Encrypt the environment")
@click.option("-k", "--encryption-key", required=False, help="The key to use for encryption. Required if --encrypt is used.")
@click.help_option('--help')
def update(name, host, port, user, password, service, database, encrypt, encryption_key):
    """Update a system environment.

    Examples:

        Update the host and port of an environment:
          elm-tool environment update dev-pg --host new-host --port 5433

        Update the password:
          elm-tool environment update prod-ora --password new-password

        Encrypt an existing environment:
          elm-tool environment update dev-mysql --encrypt --encryption-key mypassword

        Update multiple fields at once:
          elm-tool environment update dev-pg --host new-host --port 5433 --user new-user

        Using the edit alias:
          elm-tool environment edit dev-pg --host new-host
    """
    # Check if encryption key is provided when encrypt flag is set
    if encrypt and not encryption_key:
        raise click.UsageError("Option '--encryption-key' / '-k' is required when using '--encrypt' / '-e'.")

    # Check if any field is provided to update
    if not any([host, port, user, password, service, database, encrypt]):
        raise click.UsageError("At least one field must be provided to update")

    # Use core module to update environment
    result = core_env.update_environment(
        name=name,
        host=host,
        port=port,
        user=user,
        password=password,
        service=service,
        db_type=database,
        encrypt=encrypt,
        encryption_key=encryption_key
    )

    if result.success:
        click.echo(f"Environment '{name}' updated successfully")
    else:
        raise click.UsageError(result.message)

@environment.command()
@click.argument('name')
@click.option("-k", "--encryption-key", required=False, help="The key to decrypt the environment if it's encrypted")
@click.help_option('-h', '--help')
def test(name, encryption_key=None):
    """Test a system environment by attempting to connect to the database.

    Examples:

        Test a database connection:
          elm-tool environment test dev-pg

        Test an encrypted environment connection:
          elm-tool environment test secure-mysql --encryption-key mypassword

        Using the validate alias:
          elm-tool environment validate dev-pg
    """
    # Ensure the required database driver is installed for the environment
    try:
        # Get environment details to check database type
        env_result = core_env.get_environment(name=name, encryption_key=encryption_key)
        if env_result.success and env_result.data:
            env_type = env_result.data.get('type', '').upper()
            if env_type:
                ensure_db_driver_installed(env_type)
    except Exception:
        pass  # Continue with test even if driver check fails

    # Use core module to test environment
    result = core_env.test_environment(name=name, encryption_key=encryption_key)

    if result.success:
        click.echo(f"✓ {result.message}")
        return True
    else:
        click.echo(f"✗ {result.message}")
        return False

@environment.command()
@click.argument('name')
@click.option("-q", "--query", required=True, help="SQL query to execute")
@click.option("-k", "--encryption-key", required=False, help="Encryption key for encrypted environments")
@click.help_option('-h', '--help')
def execute(name, query, encryption_key):
    """Execute a SQL query on a database

    Examples:

        Execute a simple query:
          elm-tool environment execute dev-pg --query "SELECT * FROM users LIMIT 10"

        Execute a query on an encrypted environment:
          elm-tool environment execute secure-mysql --query "SHOW TABLES" --encryption-key mypassword

        Using the exec alias:
          elm-tool environment exec dev-pg --query "SELECT COUNT(*) FROM orders"
    """
    # Use core module to execute SQL
    result = core_env.execute_sql(
        environment=name,
        query=query,
        encryption_key=encryption_key
    )

    if result.success:
        if result.data:
            # Convert data to DataFrame and display
            df = pd.DataFrame(result.data)
            click.echo(df.to_string(index=False))
        else:
            click.echo(result.message)
    else:
        click.echo(f"Error: {result.message}")

ALIASES = {
    "new": create,
    "ls": list,
    "rm": delete,
    "remove": delete,
    "del": delete,
    "inspect": show,
    "edit": update,
    "validate": test,
    "exec": execute,
    "run": execute
}
