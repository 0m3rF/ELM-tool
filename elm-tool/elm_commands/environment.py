import click
import os
import configparser
from elm_utils import variables, encryption

config = configparser.ConfigParser()

class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name].name
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)

@click.group(cls=AliasedGroup)
def environment():
    """Environment management commands"""
    pass


@environment.command()
@click.option("-n", "--name", required=True, help="Name of the environment")
@click.option("-h", "--host", required=True, help="Host of the environment")
@click.option("-p", "--port", required=True, help="Port of the environment", type=int)
@click.option("-u", "--user", required=True, help="User of the environment")
@click.option("-P", "--password", required=True, help="Password of the environment")
@click.option("-s", "--service", required=True, help="Service of the environment")
@click.option("-t", "--type", required=True, type=click.Choice(['ORACLE', 'POSTGRES', 'MYSQL', 'MSSQL'], case_sensitive=False),  help="Type of the environment")
@click.option("-o", "--overwrite", is_flag=True, default= False, help="Overwrites existing environment definition")
@click.option("-e", "--encrypt", is_flag=True, default= False, help="Overwrites existing environment definition")
@click.option("-k", "--encryption-key", required=False, help="The key to use for encryption. Required if --encrypt is used. Unused if no encrypt has given.")
def create(name, host, port, user, password, service, type, overwrite, encrypt, encryption_key ):
    """Create a new environment"""
    if encrypt and not encryption_key:
        # Raise an error if --encrypt is True but --encryption-key is missing
        raise click.UsageError("Option '--encryption-key' / '-k' is required when using '--encrypt' / '-e'.")

    # Check if environment file exists
    if( not os.path.isfile(variables.ENVS_FILE)): # create it if not exists
        with open(variables.ENVS_FILE, "w") as env_file:
            env_file.close()

    config.read(variables.ENVS_FILE)

    if name in config and not overwrite:
        raise click.UsageError("Name is already exists. To overwrite use '-o' / '--overwrite' exisisting environment definition.")
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

    print("environment create sucsessfully")

@environment.command()
@click.option("-a", "--all", is_flag=True, default=False, help="Show all content of the environment")
@click.option("-h", "--host", is_flag=True, default=False, help="Show host of the environment")
@click.option("-p", "--port", is_flag=True, default=False, help="Show port of the environment")
@click.option("-u", "--user", is_flag=True, default=False, help="Show user of the environment")
@click.option("-P", "--password", is_flag=True, default=False, help="Show password of the environment")
@click.option("-s", "--service", is_flag=True, default=False, help="Show service of the environment")
@click.option("-t", "--type", is_flag=True, default=False, help="Show type of the environment")
def list(all, host, port, user, password, service, type):
    """List all environments"""
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
@click.option("-n", "--name", required=True, help="Name of the environment to delete")
def delete(name):
    """Remove a system environment"""
    config.read(variables.ENVS_FILE)

    if (not name in config.sections()):
        raise click.UsageError("Environment is not found")

    del config[name]

    with open(variables.ENVS_FILE, 'w') as configfile:
        config.write(configfile)
        configfile.close()


@environment.command()
@click.option("-n", "--name", required=True, help="Name of the environment to show")
@click.option("-k", "--encryption-key", required=False, help="The key to decrypt the environment if it's encrypted")
def show(name, encryption_key):
    """Show a system environment"""
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
@click.option("-n", "--name", required=True, help="Name of the environment to update")
@click.option("-h", "--host", required=False, help="Host of the environment")
@click.option("-p", "--port", required=False, help="Port of the environment", type=int)
@click.option("-u", "--user", required=False, help="User of the environment")
@click.option("-P", "--password", required=False, help="Password of the environment")
@click.option("-s", "--service", required=False, help="Service of the environment")
@click.option("-t", "--type", required=False, type=click.Choice(['ORACLE', 'POSTGRES', 'MYSQL', 'MSSQL'], case_sensitive=False), help="Type of the environment")
@click.option("-e", "--encrypt", is_flag=True, default=False, help="Encrypt the environment")
@click.option("-k", "--encryption-key", required=False, help="The key to use for encryption. Required if --encrypt is used.")
def update(name, host, port, user, password, service, type, encrypt, encryption_key):
    """Update a system environment"""
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
@click.option("-n", "--name", required=True, help="Name of the environment to test")
@click.option("-k", "--encryption-key", required=False, help="The key to decrypt the environment if it's encrypted")
def test(name, encryption_key):
    """Test a system environment by attempting to connect to the database"""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import SQLAlchemyError

    # Read the config file
    config.read(variables.ENVS_FILE)

    # Check if the environment exists
    if not name in config.sections():
        raise click.UsageError(f"Environment '{name}' not found")

    # Check if the environment is encrypted
    is_encrypted = config[name].get("is_encrypted", 'False') == 'True'

    # Get environment details
    if is_encrypted:
        if not encryption_key:
            raise click.UsageError(f"Environment '{name}' is encrypted. Provide an encryption key to test it.")

        try:
            # Decrypt the environment
            decrypted_env = encryption.decrypt_environment(dict(config[name]), encryption_key)

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
        env_type = config[name]["type"].upper()
        host = config[name]["host"]
        port = config[name]["port"]
        user = config[name]["user"]
        password = config[name]["password"]
        service = config[name]["service"]

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

ALIASES = {
    "new": create,
    "ls": list,
    "rm": delete,
    "remove": delete,
    "inspect": show,
    "edit": update,
    "validate": test
}
