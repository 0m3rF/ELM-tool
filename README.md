# ELM Tool

Extract, Load and Mask Tool for Database Operations

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Environment Management](#environment-management)
  - [Data Copy Operations](#data-copy-operations)
  - [Data Masking](#data-masking)
  - [Test Data Generation](#test-data-generation)
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
git clone https://github.com/omerfarukkirli/elm-tool.git
cd elm-tool
pip install -e .
```

## Usage

ELM Tool provides a command-line interface with several command groups:

```bash
elm-tool --help
```

### Environment Management

Environments are database connection profiles that store connection details.

```bash
# Create a new PostgreSQL environment
elm-tool environment create dev-pg --host localhost --port 5432 --user postgres --password password --service postgres --type postgres

# Create an Oracle environment
elm-tool environment create prod-ora --host oraserver --port 1521 --user system --password oracle --service XE --type oracle

# Create an encrypted MySQL environment
elm-tool environment create secure-mysql --host dbserver --port 3306 --user root --password secret --service mysql --type mysql --encrypt --encryption-key mypassword

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

Copy data between databases or to/from files.

```bash
# Export query results to a file
elm-tool copy db2file --source dev-pg --query "SELECT * FROM users" --file users.csv --format CSV

# Import data from a file to a database table
elm-tool copy file2db --source users.csv --target prod-pg --table users --format CSV --mode APPEND

# Copy data directly between databases
elm-tool copy db2db --source dev-pg --target prod-pg --query "SELECT * FROM users" --table users --mode APPEND

# Process large datasets with batching
elm-tool copy db2db --source dev-pg --target prod-pg --query "SELECT * FROM users" --table users --batch-size 1000 --parallel 4
```

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

## Configuration

ELM Tool stores environment configurations in `~/.elm/environments.ini` by default. Masking rules are stored in `~/.elm/masking.json`.

You can encrypt sensitive environment information using the `--encrypt` flag when creating or updating environments.

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
