# Database Setup Tool

A standalone CLI tool for setting up database containers for development and testing on local machines. Built with a user-friendly command-line interface using the Click library.

## Features

- **Docker Detection**: Automatically checks if Docker is installed and running
- **Multiple Database Support**: PostgreSQL, MySQL, MSSQL, Oracle
- **Parallel/Sequential Setup**: Choose between parallel or sequential container startup
- **Health Checks**: Comprehensive health checking for all databases
- **Connection Strings**: Easy access to database connection information
- **Container Management**: Start, stop, remove, and check status of containers
- **Timeout Handling**: Configurable timeouts for slow connections
- **Error Recovery**: Retry logic and fallback options for problematic containers

## Prerequisites

- Python 3.7+
- Click library
- Docker Desktop (will be checked and installation guidance provided)

## Installation

The script is ready to use as-is. No additional installation required.

## Usage

### Basic Commands

```bash
# Check if Docker is installed and running
python database_setup.py check-docker

# Setup all databases (parallel mode by default)
python database_setup.py setup

# Setup specific databases
python database_setup.py setup -d postgresql -d mysql

# Setup databases sequentially (slower but more reliable)
python database_setup.py setup --sequential

# Setup with fresh containers (removes existing ones first)
python database_setup.py setup --remove-existing

# Pull latest images before setup
python database_setup.py setup --pull

# Check status of all containers
python database_setup.py status

# Check status of specific databases
python database_setup.py status -d postgresql -d mysql

# Show connection strings
python database_setup.py connections

# Remove all containers
python database_setup.py remove

# Remove specific containers
python database_setup.py remove -d mssql

# Force remove without confirmation
python database_setup.py remove --force

# Debug MSSQL connection issues
python database_setup.py debug mssql
```

### Advanced Options

```bash
# Set custom timeout (default: 600 seconds)
python database_setup.py --timeout 900 setup

# Get help for any command
python database_setup.py --help
python database_setup.py setup --help
```

## Database Configurations

The tool sets up the following databases with these default configurations:

### PostgreSQL
- **Container Name**: ELM_TOOL_postgresql
- **Port**: 5433 (mapped from 5432)
- **Database**: ELM_TOOL_db
- **User**: ELM_TOOL_user
- **Password**: ELM_TOOL_password
- **Connection**: `postgresql://ELM_TOOL_user:ELM_TOOL_password@localhost:5433/ELM_TOOL_db`

### MySQL
- **Container Name**: ELM_TOOL_mysql
- **Port**: 3307 (mapped from 3306)
- **Database**: ELM_TOOL_db
- **User**: ELM_TOOL_user
- **Password**: ELM_TOOL_password
- **Connection**: `mysql://ELM_TOOL_user:ELM_TOOL_password@localhost:3307/ELM_TOOL_db`

### MSSQL
- **Container Name**: ELM_TOOL_mssql
- **Port**: 1434 (mapped from 1433)
- **User**: sa
- **Password**: ELM_TOOL_Password123!
- **Connection**: `Server=localhost,1434;Database=master;User Id=sa;Password=ELM_TOOL_Password123!;TrustServerCertificate=true;`

### Oracle
- **Container Name**: ELM_TOOL_oracle
- **Port**: 1522 (mapped from 1521)
- **User**: ELM_TOOL_user
- **Password**: ELM_TOOL_password
- **Connection**: `oracle://ELM_TOOL_user:ELM_TOOL_password@localhost:1522/XE`

## Startup Priority

Databases are started in the following order for optimal performance:

1. **PostgreSQL** (Priority 1) - Fastest to start
2. **MySQL** (Priority 2) - Medium startup time
3. **MSSQL** (Priority 3) - Slower startup
4. **Oracle** (Priority 4) - Slowest to start

## Timeout Considerations

The tool includes generous timeouts for slow connections:

- **Default timeout**: 600 seconds (10 minutes)
- **PostgreSQL**: 30 seconds wait time
- **MySQL**: 45 seconds wait time
- **MSSQL**: 60 seconds wait time
- **Oracle**: 90 seconds wait time

For very slow connections, you can increase the timeout:
```bash
python database_setup.py --timeout 1200 setup
```

## Docker Installation Guidance

If Docker is not installed, the tool will provide platform-specific installation instructions:

- **Windows**: Docker Desktop download and installation steps
- **macOS**: Docker Desktop installation guide
- **Linux**: Package manager commands for various distributions

## Error Handling

The tool includes comprehensive error handling:

- **Docker not available**: Provides installation guidance
- **Container startup failures**: Continues with other databases
- **Health check failures**: Retries with multiple methods (especially for MSSQL)
- **Network issues**: Timeout handling and retry logic

## Examples

### Quick Start
```bash
# Check Docker and setup all databases
python database_setup.py check-docker
python database_setup.py setup
```

### Development Workflow
```bash
# Setup only the databases you need
python database_setup.py setup -d postgresql -d mysql

# Check if they're running
python database_setup.py status

# Get connection strings
python database_setup.py connections
```

### Troubleshooting
```bash
# If MSSQL has issues
python database_setup.py debug mssql

# Clean restart
python database_setup.py remove --force
python database_setup.py setup --pull
```

## Notes

- Ports are incremented by 1 from standard ports to avoid conflicts with existing installations
- MSSQL includes special SSL handling for development environments
- All containers use the ELM_TOOL prefix for easy identification
- The tool preserves existing containers unless explicitly asked to remove them
