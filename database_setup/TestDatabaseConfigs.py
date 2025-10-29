import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elm.models import DatabaseConfig
from typing import Dict

MSSQL_PASSWORD = os.environ.get("ELM_TOOL_TEST_MSSQL_PASS", "ELM_TOOL_Password123!")
POSTGRES_PASSWORD = os.environ.get("ELM_TOOL_TEST_POSTGRES_PASS", "ELM_TOOL_password")
MYSQL_PASSWORD = os.environ.get("ELM_TOOL_TEST_MYSQL_PASS", "ELM_TOOL_password")
MYSQL_ROOT_PASSWORD = os.environ.get("ELM_TOOL_TEST_MYSQL_ROOT_PASS", "ELM_TOOL_root_password")
ORACLE_PASSWORD = os.environ.get("ELM_TOOL_TEST_ORACLE_PASS", "ELM_TOOL_password")

class DatabaseConfigs:
    """Database configuration definitions with dependencies."""

    @staticmethod
    def get_configs() -> Dict[str, DatabaseConfig]:
        """Get all database configurations with startup priorities."""
        return {
            "postgresql": DatabaseConfig(
                name="ELM_TOOL_postgresql",
                image="postgres:15-alpine",
                target_port=5433,  # Host port
                env_vars={
                    "POSTGRES_DB": "ELM_TOOL_db",
                    "POSTGRES_USER": "ELM_TOOL_user",
                    "POSTGRES_PASSWORD": POSTGRES_PASSWORD
                },
                health_check_cmd=["pg_isready", "-U", "ELM_TOOL_user", "-d", "ELM_TOOL_db"],
                default_port=5432,  # Default PostgreSQL port inside container
                wait_time=30,
                startup_priority=1  # Start first (fastest to start)
            ),
            "mysql": DatabaseConfig(
                name="ELM_TOOL_mysql",
                image="mysql:8.0",
                target_port=3307,  # Host port
                env_vars={
                    "MYSQL_DATABASE": "ELM_TOOL_db",
                    "MYSQL_USER": "ELM_TOOL_user",
                    "MYSQL_PASSWORD": MYSQL_PASSWORD,
                    "MYSQL_ROOT_PASSWORD": MYSQL_ROOT_PASSWORD
                },
                health_check_cmd=["mysqladmin", "ping", "-h", "localhost"],
                default_port=3306,  # Default MySQL port inside container
                wait_time=45,
                startup_priority=2  # Start second
            ),
            "mssql": DatabaseConfig(
                name="ELM_TOOL_mssql",
                image="mcr.microsoft.com/mssql/server:2022-latest",
                target_port=1434,  # Host port
                env_vars={
                    "ACCEPT_EULA": "Y",
                    "SA_PASSWORD": MSSQL_PASSWORD,
                    "MSSQL_PID": "Express"
                },
                health_check_cmd=["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", MSSQL_PASSWORD, "-C", "-Q", "SELECT 1"],
                default_port=1433,  # Default MSSQL port inside container
                wait_time=60,
                startup_priority=3  # Start third
            ),
            "oracle": DatabaseConfig(
                name="ELM_TOOL_oracle",
                image="gvenzl/oracle-xe:21-slim",
                target_port=1522,  # Host port
                env_vars={
                    "ORACLE_PASSWORD": ORACLE_PASSWORD,
                    "APP_USER": "ELM_TOOL_user",
                    "APP_USER_PASSWORD": ORACLE_PASSWORD
                },
                health_check_cmd=["bash", "-c", f'''sqlplus -s ELM_TOOL_user/{ORACLE_PASSWORD}@localhost:1521/XEPDB1 <<EOF
                                    SELECT 1 FROM DUAL;
                                    EXIT;
                                    EOF'''],
                default_port=1521,  # Default Oracle port inside container
                wait_time=90,
                startup_priority=4  # Start last (slowest to start)
            )
        }
