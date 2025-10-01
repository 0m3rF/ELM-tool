# ELM Tool

Extract, Load and Mask Tool for Database Operations

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
    - [Environment Management](#environment-management)
    - [Data Copy Operations](#data-copy-operations)
    - [Data Masking](#data-masking)
    - [Test Data Generation](#test-data-generation)
    - [Configuration Management](#configuration-management)
  - [Python API](#python-api)
    - [Environment Management API](#environment-management-api)
    - [Data Copy API](#data-copy-api)
    - [Data Masking API](#data-masking-api)
    - [Data Generation API](#data-generation-api)
    - [Configuration Management API](#configuration-management-api)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Description

ELM Tool is a powerful database utility designed to simplify database operations across different environments. It helps you:

- **Extract** data from various database systems
- **Load** data between different database environments
- **Mask** sensitive data for testing and development
- **Generate** test data with customizable properties

The tool provides a unified interface for working with multiple database types, making it easier to manage data across development, testing, and production environments.

## Features

- **Multi-database support**: Works with PostgreSQL, Oracle, MySQL, and MSSQL
- **Environment management**: Create, update, and manage database connection profiles
- **Configuration management**: Customize tool settings, directories, and behavior
- **Data masking**: Protect sensitive information with various masking algorithms
- **Test data generation**: Create realistic test data with customizable properties
- **Cross-database operations**: Copy data between different database systems
- **Batch processing**: Handle large datasets efficiently with batching and parallel processing
- **Secure storage**: Optional encryption for sensitive connection information
- **File export/import**: Export query results to CSV or JSON and import back to databases

## Installation

```bash
pip install elm-tool
```

Or install from source:

```bash
git clone https://github.com/0m3rF/elm-tool.git
cd elm-tool
pip install -e .
```

## Usage

ELM Tool can be used both as a command-line tool and as a Python library.

### Command Line Interface

ELM Tool provides a command-line interface with several command groups:

```bash
elm-tool --help
```

**Available Commands:**
- `config` - Configuration management commands
- `environment` - Environment management commands
- `copy` - Data copy commands for database operations
- `mask` - Data masking commands for sensitive information
- `generate` - Data generation commands for testing

### Environment Management

Environments are database connection profiles that store connection details.

```bash
# Create a new PostgreSQL environment
elm-tool environment create dev-pg --host localhost --port 5432 --user postgres --password password --service postgres --database postgres

# Create an Oracle environment with service name (default)
elm-tool environment create prod-ora --host oraserver --port 1521 --user system --password oracle --service XE --database oracle --connection-type service_name

# Create an Oracle environment with SID
elm-tool environment create prod-ora-sid --host oraserver --port 1521 --user system --password oracle --service ORCL --database oracle --connection-type sid

# Create an encrypted MySQL environment
elm-tool environment create secure-mysql --host dbserver --port 3306 --user root --password secret --service mysql --database mysql --encrypt --encryption-key mypassword

# List all environments
elm-tool environment list

# Show all details of environments
elm-tool environment list --all

# Show specific environment details
elm-tool environment show dev-pg

# Test database connection
elm-tool environment test dev-pg

# Update environment settings
elm-tool environment update dev-pg --host new-host --port 5433

# Delete an environment
elm-tool environment delete dev-pg

# Execute a query on an environment
elm-tool environment execute dev-pg --query "SELECT * FROM users LIMIT 10"
```

### Data Copy Operations

Copy data between databases or to/from files with **high-performance streaming** for large datasets and LOB data.

**🚀 Performance Features:**
- **10-100x faster** than traditional methods for large datasets
- **Optimized streaming** with database-specific bulk loaders
- **Real-time progress reporting** for batch operations
- **Efficient LOB handling** (CLOB, BLOB, TEXT, etc.)
- **Low memory usage** with streaming architecture

```bash
# Export query results to a file
elm-tool copy db2file --source dev-pg --query "SELECT * FROM users" --file users.csv --format CSV

# Import data from a file to a database table
elm-tool copy file2db --source users.csv --target prod-pg --table users --format CSV --mode APPEND

# Copy data directly between databases
elm-tool copy db2db --source dev-pg --target prod-pg --query "SELECT * FROM users" --table users --mode APPEND

# Process large datasets with batching (shows progress per batch)
elm-tool copy db2db --source dev-pg --target prod-pg --query "SELECT * FROM users" --table users --batch-size 10000 --parallel 4
```

**Optimized Methods by Database:**
- **PostgreSQL**: COPY protocol or execute_values (50-100x faster)
- **Oracle**: executemany with array binding (20-100x faster)
- **SQL Server**: fast_executemany (15-40x faster)
- **MySQL**: optimized executemany (10-30x faster)

### Data Masking

Mask sensitive data to protect privacy.

```bash
# Add a masking rule for a column
elm-tool mask add --column password --algorithm star

# Add environment-specific masking
elm-tool mask add --column credit_card --algorithm star_length --environment prod --length 6

# List all masking rules
elm-tool mask list

# Test a masking rule
elm-tool mask test --column credit_card --value "1234-5678-9012-3456" --environment prod

# Remove a masking rule
elm-tool mask remove --column password
```

### Test Data Generation

Generate realistic test data for development and testing.

```bash
# Generate data for specific columns
elm-tool generate data --columns "id,name,email,created_at" --num-records 100

# Generate data based on table schema
elm-tool generate data --environment dev-pg --table users --num-records 100

# Generate data with specific patterns
elm-tool generate data --columns "id,name,email" --pattern "email:[a-z]{5}@example.com" --num-records 50

# Generate data with specific ranges
elm-tool generate data --columns "id,price,created_at" --min-number 100 --max-number 999 --start-date "2023-01-01" --end-date "2023-12-31"

# Save generated data to a file
elm-tool generate data --columns "id,name,email" --output "test_data.csv" --num-records 200

# Write generated data directly to a database
elm-tool generate data --environment dev-pg --table users --num-records 100 --write-to-db
```

### Configuration Management

Manage ELM Tool configuration settings including tool home directory, virtual environment settings, and other configurable parameters.

```bash
# Show current configuration and file paths
elm-tool config show

# Set ELM_TOOL_HOME directory
elm-tool config set ELM_TOOL_HOME /path/to/elm/home

# Set custom virtual environment name
elm-tool config set VENV_NAME my_custom_venv

# Get a specific configuration value
elm-tool config get ELM_TOOL_HOME

# Show file paths with existence indicators
elm-tool config paths

# Reset configuration to defaults
elm-tool config reset

# Using aliases
elm-tool config info          # Same as 'show'
elm-tool config dirs          # Same as 'paths'
elm-tool config update KEY VALUE  # Same as 'set'
```

**Configurable Settings:**
- `ELM_TOOL_HOME`: The home directory for ELM Tool files and configurations
- `VENV_NAME`: The name of the virtual environment directory
- `APP_NAME`: The application name used in various contexts

**Configuration Storage:**
- Configuration is stored in JSON format at `{ELM_TOOL_HOME}/config.json`
- Settings persist across tool restarts
- Automatic fallback to defaults if configuration file is corrupted

### Python API

ELM Tool can also be used as a Python library, allowing you to integrate its functionality directly into your Python applications.

```python
import elm

# Create a database environment
elm.create_environment(
    name="dev-pg",
    host="localhost",
    port=5432,
    user="postgres",
    password="password",
    service="postgres",
    database="postgres"
)

# Test the connection
result = elm.test_environment("dev-pg")
print(f"Connection successful: {result['success']}")

# Execute a query
data = elm.execute_sql("dev-pg", "SELECT * FROM users LIMIT 10")
print(data)
```

#### Environment Management API

```python
# Create a new environment
elm.create_environment(
    name="dev-pg",
    host="localhost",
    port=5432,
    user="postgres",
    password="password",
    service="postgres",
    database="postgres"
)

# Create an Oracle environment with service name
elm.create_environment(
    name="oracle-env",
    host="oraserver",
    port=1521,
    user="system",
    password="oracle",
    service="XE",
    database="oracle",
    connection_type="service_name"
)

# Create an Oracle environment with SID
elm.create_environment(
    name="oracle-sid-env",
    host="oraserver",
    port=1521,
    user="system",
    password="oracle",
    service="ORCL",
    database="oracle",
    connection_type="sid"
)

# Create an encrypted environment
elm.create_environment(
    name="secure-mysql",
    host="dbserver",
    port=3306,
    user="root",
    password="secret",
    service="mysql",
    database="mysql",
    encrypt=True,
    encryption_key="mypassword"
)

# List all environments
environments = elm.list_environments()
for env in environments:
    print(env['name'])

# Get details of a specific environment
env_details = elm.get_environment("dev-pg")
print(env_details)

# Test a connection
result = elm.test_environment("dev-pg")
if result['success']:
    print("Connection successful!")
else:
    print(f"Connection failed: {result['message']}")

# Execute a query
data = elm.execute_sql("dev-pg", "SELECT * FROM users LIMIT 10")
print(data)

# Delete an environment
elm.delete_environment("dev-pg")
```

#### Data Copy API

```python
# Copy data from database to file
result = elm.copy_db_to_file(
    source_env="dev-pg",
    query="SELECT * FROM users",
    file_path="users.csv",
    file_format="csv"
)
print(f"Exported {result['record_count']} records")

# Copy data from file to database
result = elm.copy_file_to_db(
    file_path="users.csv",
    target_env="prod-pg",
    table="users",
    file_format="csv",
    mode="APPEND"
)
print(f"Imported {result['record_count']} records")

# Copy data between databases
result = elm.copy_db_to_db(
    source_env="dev-pg",
    target_env="prod-pg",
    query="SELECT * FROM users",
    table="users",
    mode="APPEND",
    batch_size=1000
)
print(f"Copied {result['record_count']} records")
```

#### Data Masking API

```python
# Add a masking rule
elm.add_mask(column="password", algorithm="star")

# Add environment-specific masking
elm.add_mask(
    column="credit_card",
    algorithm="star_length",
    environment="prod",
    length=6
)

# List all masking rules
masks = elm.list_masks()
print(masks)

# Test a masking rule
result = elm.test_mask(
    column="credit_card",
    value="1234-5678-9012-3456",
    environment="prod"
)
print(f"Original: {result['original']}")
print(f"Masked: {result['masked']}")

# Remove a masking rule
elm.remove_mask(column="password")
```

#### Data Generation API

```python
# Generate random data
data = elm.generate_data(
    num_records=100,
    columns=["id", "name", "email", "created_at"]
)
print(data)

# Generate data with specific patterns
data = elm.generate_data(
    num_records=50,
    columns=["id", "name", "email"],
    pattern={"email": "[a-z]{5}@example.com"}
)
print(data)

# Generate data and save to file
result = elm.generate_and_save(
    num_records=200,
    columns=["id", "name", "email"],
    output="test_data.csv",
    format="csv"
)
print(f"Generated {result['record_count']} records")

# Generate data and write to database
result = elm.generate_and_save(
    num_records=100,
    environment="dev-pg",
    table="users",
    write_to_db=True,
    mode="APPEND"
)
print(f"Generated and wrote {result['record_count']} records to database")
```

#### Configuration Management API

```python
# Get current configuration
config = elm.get_config()
print(f"ELM_TOOL_HOME: {config['ELM_TOOL_HOME']}")
print(f"VENV_NAME: {config['VENV_NAME']}")

# Set configuration values
success = elm.set_config("ELM_TOOL_HOME", "/path/to/custom/home")
print(f"Configuration updated: {success}")

# Get detailed configuration information
info = elm.get_config_info()
print("Configuration values:")
for key, value in info['config'].items():
    print(f"  {key}: {value}")

print("File paths:")
for key, path in info['paths'].items():
    print(f"  {key}: {path}")

# Reset configuration to defaults
success = elm.reset_config()
print(f"Configuration reset: {success}")
```

## Configuration

ELM Tool provides comprehensive configuration management through the `config` command and API.

### Configuration Files and Directories

By default, ELM Tool stores its configuration and data files in the following locations:

- **Configuration**: `{ELM_TOOL_HOME}/config.json` - Tool configuration settings
- **Environments**: `{ELM_TOOL_HOME}/environments.ini` - Database connection profiles
- **Masking Rules**: `{ELM_TOOL_HOME}/masking.json` - Data masking definitions
- **Virtual Environment**: `{ELM_TOOL_HOME}/{VENV_NAME}` - Python virtual environment

Where `{ELM_TOOL_HOME}` defaults to the user's configuration directory (e.g., `~/.config/ELMtool` on Linux, `%APPDATA%\ELMtool` on Windows).

### Configurable Settings

You can customize the following settings using the `config` command:

- **ELM_TOOL_HOME**: Change the base directory for all ELM Tool files
- **VENV_NAME**: Customize the virtual environment directory name
- **APP_NAME**: Modify the application name used in various contexts

### Configuration Management

```bash
# View current configuration
elm-tool config show

# Change the tool's home directory
elm-tool config set ELM_TOOL_HOME /custom/path

# Reset to defaults
elm-tool config reset
```

### Security

You can encrypt sensitive environment information using the `--encrypt` flag when creating or updating environments, or by setting `encrypt=True` when using the API. Encrypted environments require an encryption key for access.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

GNU GENERAL PUBLIC LICENSE Version 3

## Author

Ömer Faruk Kırlı (omerfarukkirli@gmail.com)
