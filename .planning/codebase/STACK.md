# Tech Stack

## Core Language & Runtime
- **Language**: Python (>=3.7)
- **Primary Tooling**: ELM is a CLI pipeline tool, relying on `click` for its UI.

## Key Frameworks & Libraries
- **CLI**: `click` (>=8.0.0)
- **Data Manipulation**: `pandas` (>=2.0.0)
- **Database ORM/Connection**: `sqlalchemy` (>=2.0.0)
- **Cryptography**: `cryptography` (>=3.4.7)
- **Data Mocking/Obfuscation**: `Faker` (>=37.4.2)
- **Configuration Parsing**: `configparser` (>=5.0.0), `pyyaml` (>=6.0)
- **Cross-Platform Storage**: `platformdirs` (>=3.0.0)

## Database Drivers (Optional Dependencies)
- **PostgreSQL**: `psycopg2-binary` (>=2.9.1)
- **Oracle**: `oracledb` (>=1.0.0)
- **MySQL**: `pymysql` (>=1.0.2)
- **SQL Server**: `pyodbc` (>=4.0.34)

## Build System
- **Backend**: `hatchling` (configured in `pyproject.toml`)
- **Package Management**: PIP standard tools

## Key Configuration Files
- `pyproject.toml`: Contains metadata, dependencies, and build configurations.
- `requirements.txt`: Flat dependency list.
