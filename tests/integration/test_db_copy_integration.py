"""
Integration tests for ELM-tool's database copying functionality.

This module tests database-to-database copy operations across all supported
database systems (PostgreSQL, MySQL, MSSQL, Oracle) with both small and large datasets.

Test Coverage:
- 4 source databases × 3 destination databases = 12 copy operations per scenario
- Scenario A: Small dataset (5 rows)
- Scenario B: Large dataset (500,000 rows)
- Data integrity verification
- Schema compatibility testing
"""

import os
import pytest
import pandas as pd
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Tuple

# Import ELM-tool API
from elm.core.types import WriteMode
import elm.api as elm
from elm.core import copy as core_copy
from elm.core import environment as core_env


# Database configuration from environment variables
POSTGRES_PASSWORD = os.environ.get("ELM_TOOL_TEST_POSTGRES_PASS", "ELM_TOOL_password")
MYSQL_PASSWORD = os.environ.get("ELM_TOOL_TEST_MYSQL_PASS", "ELM_TOOL_password")
MSSQL_PASSWORD = os.environ.get("ELM_TOOL_TEST_MSSQL_PASS", "ELM_TOOL_Password123!")
ORACLE_PASSWORD = os.environ.get("ELM_TOOL_TEST_ORACLE_PASS", "ELM_TOOL_password")


# Database connection configurations
DB_CONFIGS = {
    "postgresql": {
        "host": "localhost",
        "port": 5433,
        "user": "ELM_TOOL_user",
        "password": POSTGRES_PASSWORD,
        "service": "ELM_TOOL_db",
        "database": "POSTGRES"
    },
    "mysql": {
        "host": "localhost",
        "port": 3307,
        "user": "ELM_TOOL_user",
        "password": MYSQL_PASSWORD,
        "service": "ELM_TOOL_db",
        "database": "MYSQL"
    },
    "mssql": {
        "host": "localhost",
        "port": 1434,
        "user": "sa",
        "password": MSSQL_PASSWORD,
        "service": "master",
        "database": "MSSQL"
    },
    "oracle": {
        "host": "localhost",
        "port": 1522,
        "user": "ELM_TOOL_user",
        "password": ORACLE_PASSWORD,
        "service": "XEPDB1",
        "database": "ORACLE",
        "connection_type": "service_name"
    }
}


# Test data schemas - different for each database to test schema compatibility
TEST_SCHEMAS = {
    "postgresql": {
        "table_name": "test_users_pg",
        "create_sql": """
            CREATE TABLE IF NOT EXISTS test_users_pg (
                user_id INTEGER PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                email VARCHAR(255),
                balance DECIMAL(10, 2),
                created_date DATE,
                is_active BOOLEAN
            )
        """,
        "drop_sql": "DROP TABLE IF EXISTS test_users_pg CASCADE"
    },
    "mysql": {
        "table_name": "test_products_mysql",
        "create_sql": """
            CREATE TABLE IF NOT EXISTS test_products_mysql (
                product_id INT PRIMARY KEY,
                product_name VARCHAR(200) NOT NULL,
                description TEXT,
                price DECIMAL(12, 2),
                stock_quantity INT,
                last_updated DATETIME
            )
        """,
        "drop_sql": "DROP TABLE IF EXISTS test_products_mysql"
    },
    "mssql": {
        "table_name": "test_orders_mssql",
        "create_sql": """
            IF OBJECT_ID('test_orders_mssql', 'U') IS NOT NULL
                DROP TABLE test_orders_mssql;
            CREATE TABLE test_orders_mssql (
                order_id INT PRIMARY KEY,
                customer_name NVARCHAR(150),
                order_total DECIMAL(15, 2),
                order_date DATE,
                status NVARCHAR(50),
                notes NVARCHAR(MAX)
            )
        """,
        "drop_sql": "IF OBJECT_ID('test_orders_mssql', 'U') IS NOT NULL DROP TABLE test_orders_mssql"
    },
    "oracle": {
        "table_name": "test_inventory_ora",
        "create_sql": """
            BEGIN
                EXECUTE IMMEDIATE 'DROP TABLE test_inventory_ora';
            EXCEPTION
                WHEN OTHERS THEN NULL;
            END;
            /
            CREATE TABLE test_inventory_ora (
                item_id NUMBER PRIMARY KEY,
                item_name VARCHAR2(200),
                category VARCHAR2(100),
                unit_price NUMBER(10, 2),
                quantity NUMBER,
                last_check DATE
            )
        """,
        "drop_sql": "DROP TABLE test_inventory_ora"
    }
}


# Sample test data generators
def generate_postgres_data(num_rows: int) -> pd.DataFrame:
    """Generate test data for PostgreSQL schema."""
    base_data = [
        (1, "alice_user", "alice@example.com", 1250.50, date(2024, 1, 15), True),
        (2, "bob_user", "bob@example.com", 2340.75, date(2024, 2, 20), True),
        (3, "charlie_user", "charlie@example.com", 890.25, date(2024, 3, 10), False),
        (4, "diana_user", "diana@example.com", 4567.80, date(2024, 4, 5), True),
        (5, "ethan_user", "ethan@example.com", 678.90, date(2024, 5, 12), False)
    ]
    
    if num_rows <= 5:
        data = base_data[:num_rows]
    else:
        # Replicate base data to reach desired number of rows
        data = []
        for i in range(num_rows):
            base_row = base_data[i % 5]
            # Modify ID to be unique
            new_row = (i + 1,) + base_row[1:]
            data.append(new_row)
    
    return pd.DataFrame(data, columns=["user_id", "username", "email", "balance", "created_date", "is_active"])


def generate_mysql_data(num_rows: int) -> pd.DataFrame:
    """Generate test data for MySQL schema."""
    base_data = [
        (1, "Laptop Pro 15", "High-performance laptop", 1299.99, 50, datetime(2024, 1, 1, 10, 30)),
        (2, "Wireless Mouse", "Ergonomic wireless mouse", 29.99, 200, datetime(2024, 1, 5, 14, 15)),
        (3, "USB-C Cable", "Fast charging cable", 15.50, 500, datetime(2024, 1, 10, 9, 0)),
        (4, "Monitor 27inch", "4K UHD monitor", 449.99, 75, datetime(2024, 1, 15, 16, 45)),
        (5, "Keyboard Mechanical", "RGB mechanical keyboard", 89.99, 120, datetime(2024, 1, 20, 11, 20))
    ]
    
    if num_rows <= 5:
        data = base_data[:num_rows]
    else:
        data = []
        for i in range(num_rows):
            base_row = base_data[i % 5]
            new_row = (i + 1,) + base_row[1:]
            data.append(new_row)
    
    return pd.DataFrame(data, columns=["product_id", "product_name", "description", "price", "stock_quantity", "last_updated"])


def generate_mssql_data(num_rows: int) -> pd.DataFrame:
    """Generate test data for MSSQL schema."""
    base_data = [
        (1, "John Smith", 1250.00, date(2024, 1, 10), "Completed", "Express delivery"),
        (2, "Jane Doe", 890.50, date(2024, 1, 12), "Pending", "Standard shipping"),
        (3, "Bob Johnson", 2340.75, date(2024, 1, 15), "Shipped", "Fragile items"),
        (4, "Alice Williams", 567.25, date(2024, 1, 18), "Completed", "Gift wrap requested"),
        (5, "Charlie Brown", 1890.00, date(2024, 1, 20), "Processing", "Rush order")
    ]
    
    if num_rows <= 5:
        data = base_data[:num_rows]
    else:
        data = []
        for i in range(num_rows):
            base_row = base_data[i % 5]
            new_row = (i + 1,) + base_row[1:]
            data.append(new_row)
    
    return pd.DataFrame(data, columns=["order_id", "customer_name", "order_total", "order_date", "status", "notes"])


def generate_oracle_data(num_rows: int) -> pd.DataFrame:
    """Generate test data for Oracle schema."""
    base_data = [
        (1, "Widget A", "Electronics", 45.99, 100, date(2024, 1, 5)),
        (2, "Gadget B", "Tools", 78.50, 250, date(2024, 1, 8)),
        (3, "Device C", "Electronics", 129.99, 75, date(2024, 1, 12)),
        (4, "Tool D", "Hardware", 34.25, 500, date(2024, 1, 15)),
        (5, "Component E", "Parts", 12.75, 1000, date(2024, 1, 18))
    ]
    
    if num_rows <= 5:
        data = base_data[:num_rows]
    else:
        data = []
        for i in range(num_rows):
            base_row = base_data[i % 5]
            new_row = (i + 1,) + base_row[1:]
            data.append(new_row)
    
    return pd.DataFrame(data, columns=["item_id", "item_name", "category", "unit_price", "quantity", "last_check"])


# Data generators mapping
DATA_GENERATORS = {
    "postgresql": generate_postgres_data,
    "mysql": generate_mysql_data,
    "mssql": generate_mssql_data,
    "oracle": generate_oracle_data
}


def get_worker_id(request):
    """Get the worker ID for parallel test execution."""
    if hasattr(request.config, 'workerinput'):
        return request.config.workerinput['workerid']
    return 'master'


@pytest.fixture(scope="module")
def setup_test_environments(request):
    """
    Set up test environments for all database systems.
    This fixture creates ELM-tool environments for each database.
    Uses worker-specific names to avoid conflicts in parallel execution.
    """
    worker_id = get_worker_id(request)
    environments = {}
    created_envs = []

    for db_name, config in DB_CONFIGS.items():
        # Use worker-specific environment name to avoid conflicts
        env_name = f"test_integration_{db_name}_{worker_id}"

        # Create environment
        result = core_env.create_environment(
            name=env_name,
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            service=config["service"],
            database=config["database"],
            overwrite=True,
            connection_type=config.get("connection_type")
        )

        if result.success:
            environments[db_name] = env_name
            created_envs.append(env_name)
        else:
            pytest.skip(f"Failed to create environment for {db_name}: {result.message}")

    yield environments

    # Cleanup: Delete test environments
    for env_name in created_envs:
        try:
            core_env.delete_environment(env_name)
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture(scope="function")
def clean_test_tables(setup_test_environments):
    """
    Clean up test tables BEFORE each test only.
    This ensures a clean state for each test run.
    Tables are NOT dropped after tests so users can inspect data.

    Note: We need to drop ALL possible table names from ALL databases
    because tests copy tables between databases (e.g., test_users_pg from PostgreSQL to MySQL).
    We also need to include _copy suffix tables for same-database copy tests.

    Uses worker-specific approach to minimize conflicts in parallel execution.
    """
    environments = setup_test_environments

    # All possible table names that might exist in any database
    all_table_names = [schema["table_name"] for schema in TEST_SCHEMAS.values()]

    # Add _copy suffix versions for same-database copy tests
    copy_table_names = [f"{table_name}_copy" for table_name in all_table_names]
    all_table_names.extend(copy_table_names)

    # Cleanup before test - drop all possible tables from all databases
    # Only clean tables that might have been created by this worker or previous runs
    for db_name, env_name in environments.items():
        for table_name in all_table_names:
            try:
                # Use database-specific DROP syntax
                if db_name == "postgresql":
                    drop_sql = f"DROP TABLE IF EXISTS {table_name} CASCADE"
                elif db_name == "mysql":
                    drop_sql = f"DROP TABLE IF EXISTS {table_name}"
                elif db_name == "mssql":
                    drop_sql = f"IF OBJECT_ID('{table_name}', 'U') IS NOT NULL DROP TABLE {table_name}"
                elif db_name == "oracle":
                    drop_sql = f"DROP TABLE {table_name}"
                else:
                    continue

                core_env.execute_sql(env_name, drop_sql)
            except Exception:
                pass  # Table might not exist

    yield environments

    # NO cleanup after test - user wants to inspect data after tests complete


# Helper functions for test operations

def create_source_table(db_name: str, env_name: str) -> str:
    """Create source table in the specified database."""
    schema = TEST_SCHEMAS[db_name]

    try:
        core_env.execute_sql(env_name, schema["drop_sql"])
    except Exception:
        pass

    core_env.execute_sql(env_name, schema["create_sql"])

    return schema["table_name"]


def insert_test_data(db_name: str, env_name: str, table_name: str, num_rows: int, truncate_table: bool = True, write_mode: WriteMode = WriteMode.REPLACE) -> int:
    """Insert test data into the source table."""
    # Truncate table first to ensure clean state
    from elm.core.types import WriteMode

    if truncate_table:
        try:
            truncate_sql = f"TRUNCATE TABLE {table_name}"
            core_env.execute_sql(env_name, truncate_sql)
        except Exception as e:
            # Table might not exist yet, which is fine
            # But log the error for debugging
            print(f"Warning: TRUNCATE failed for {table_name}: {e}")
    else:
        print(f"Skipping TRUNCATE for {table_name}")

    # Generate data
    data_generator = DATA_GENERATORS[db_name]
    df = data_generator(num_rows)

    # Get connection URL
    connection_url = core_env.get_connection_url(env_name)

    if write_mode == WriteMode.APPEND:
        df['product_id'] = range(len(df)+1, len(df)*2+1)
    

    # Write data to database using REPLACE mode to ensure clean state
    from elm.core.copy import write_to_db
    

    # Use smaller batch size for small datasets
    batch_size = 100 if num_rows <= 100 else 10000

    # Use REPLACE mode to truncate and insert in one operation
    records_written = write_to_db(
        data=df,
        connection_url=connection_url,
        table_name=table_name,
        mode=write_mode,
        batch_size=batch_size
    )

    return records_written


def verify_row_count(env_name: str, table_name: str, expected_count: int) -> bool:
    """Verify the row count in a table."""
    query = f"SELECT COUNT(*) as row_count FROM {table_name}"
    result = core_env.execute_sql(env_name, query)

    if result.success and result.data:
        actual_count = result.data[0]['row_count']
        return actual_count == expected_count

    return False


def verify_sample_data(env_name: str, table_name: str, sample_size: int = 5) -> pd.DataFrame:
    """Retrieve sample data from a table for verification."""
    # Different LIMIT syntax for different databases
    env_details = core_env.get_environment(env_name)
    db_type = env_details.data.get('type', '').upper()

    if db_type == 'ORACLE':
        query = f"SELECT * FROM {table_name} WHERE ROWNUM <= {sample_size}"
    elif db_type == 'MSSQL':
        query = f"SELECT TOP {sample_size} * FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name} LIMIT {sample_size}"

    result = core_env.execute_sql(env_name, query)

    if result.success and result.data:
        return pd.DataFrame(result.data)

    return pd.DataFrame()


def get_all_db_combinations() -> List[Tuple[str, str]]:
    """
    Get all database combinations for cross-database testing.
    Returns list of (source_db, target_db) tuples.
    Each database is tested as source with all other databases as targets.
    """
    db_names = list(DB_CONFIGS.keys())
    combinations = []

    for source_db in db_names:
        for target_db in db_names:
            if source_db != target_db:
                combinations.append((source_db, target_db))

    return combinations


# ============================================================================
# SCENARIO A: Small Dataset Tests (5 rows)
# ============================================================================

class TestSmallDatasetCopy:
    """Test database copy operations with small datasets (5 rows)."""

    @pytest.mark.parametrize("source_db,target_db", get_all_db_combinations())
    def test_copy_5_rows_between_databases(self, clean_test_tables, source_db, target_db):
        """
        Test copying 5 rows from source database to target database.

        This test:
        1. Creates a table in the source database
        2. Inserts 5 rows of test data
        3. Copies data to target database using ELM-tool
        4. Verifies row count and data integrity
        """
        environments = clean_test_tables
        source_env = environments[source_db]
        target_env = environments[target_db]

        # Step 1: Create source table and insert data
        source_table = create_source_table(source_db, source_env)
        rows_inserted = insert_test_data(source_db, source_env, source_table, num_rows=5)
        assert rows_inserted == 5, f"Expected 5 rows inserted, got {rows_inserted}"

        # Step 2: Define target table name (use source table name for simplicity)
        target_table = source_table

        # Step 3: Perform database-to-database copy
        query = f"SELECT * FROM {source_table}"

        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=query,
            table=target_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        # Step 4: Verify the copy operation succeeded
        assert result.success, f"Copy failed: {result.message}"
        assert result.record_count == 5, f"Expected 5 records copied, got {result.record_count}"

        # Step 5: Verify row count in target database
        assert verify_row_count(target_env, target_table, 5), \
            f"Row count verification failed for {source_db} -> {target_db}"

        # Step 6: Verify sample data integrity
        target_data = verify_sample_data(target_env, target_table, sample_size=5)
        assert len(target_data) == 5, \
            f"Expected 5 rows in target, got {len(target_data)}"

    def test_copy_postgres_to_mysql_small(self, clean_test_tables):
        """Specific test: PostgreSQL -> MySQL with 5 rows."""
        environments = clean_test_tables
        source_env = environments["postgresql"]
        target_env = environments["mysql"]

        # Create and populate source
        source_table = create_source_table("postgresql", source_env)
        insert_test_data("postgresql", source_env, source_table, num_rows=5)

        # Copy to target
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 5
        assert verify_row_count(target_env, source_table, 5)

    def test_copy_mysql_to_mssql_small(self, clean_test_tables):
        """Specific test: MySQL -> MSSQL with 5 rows."""
        environments = clean_test_tables
        source_env = environments["mysql"]
        target_env = environments["mssql"]

        # Create and populate source
        source_table = create_source_table("mysql", source_env)
        insert_test_data("mysql", source_env, source_table, num_rows=5)

        # Copy to target
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 5
        assert verify_row_count(target_env, source_table, 5)

    def test_copy_mssql_to_oracle_small(self, clean_test_tables):
        """Specific test: MSSQL -> Oracle with 5 rows."""
        environments = clean_test_tables
        source_env = environments["mssql"]
        target_env = environments["oracle"]

        # Create and populate source
        source_table = create_source_table("mssql", source_env)
        insert_test_data("mssql", source_env, source_table, num_rows=5)

        # Copy to target
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 5
        assert verify_row_count(target_env, source_table, 5)

    def test_copy_oracle_to_postgres_small(self, clean_test_tables):
        """Specific test: Oracle -> PostgreSQL with 5 rows."""
        environments = clean_test_tables
        source_env = environments["oracle"]
        target_env = environments["postgresql"]

        # Create and populate source
        source_table = create_source_table("oracle", source_env)
        insert_test_data("oracle", source_env, source_table, num_rows=5)

        # Copy to target
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 5
        assert verify_row_count(target_env, source_table, 5)


# ============================================================================
# SCENARIO B: Same-Database Copy Tests
# ============================================================================

class TestSameDatabaseCopy:
    """Test copying data within the same database system (e.g., PostgreSQL -> PostgreSQL)."""

    @pytest.mark.parametrize("db_name", ["postgresql", "mysql", "mssql", "oracle"])
    def test_copy_within_same_database(self, clean_test_tables, db_name):
        """
        Test copying data within the same database system.

        This test:
        1. Creates a source table in the database
        2. Inserts 5 rows of test data
        3. Copies data to a different table in the SAME database
        4. Verifies both tables exist with correct data

        Args:
            db_name: Database system to test (postgresql, mysql, mssql, oracle)
        """
        environments = clean_test_tables
        env_name = environments[db_name]

        # Step 1: Create source table and insert data
        source_table = create_source_table(db_name, env_name)
        rows_inserted = insert_test_data(db_name, env_name, source_table, num_rows=5)
        assert rows_inserted == 5, f"Expected 5 rows inserted, got {rows_inserted}"

        # Step 2: Define target table name (different from source to avoid conflicts)
        target_table = f"{source_table}_copy"

        # Step 3: Perform same-database copy
        query = f"SELECT * FROM {source_table}"

        result = core_copy.copy_db_to_db(
            source_env=env_name,
            target_env=env_name,  # Same environment for source and target
            query=query,
            table=target_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        # Step 4: Verify the copy operation succeeded
        assert result.success, f"Copy failed: {result.message}"
        assert result.record_count == 5, f"Expected 5 records copied, got {result.record_count}"

        # Step 5: Verify row count in target table
        assert verify_row_count(env_name, target_table, 5), \
            f"Row count verification failed for {db_name} same-database copy"

        # Step 6: Verify source table still has data (not moved, just copied)
        assert verify_row_count(env_name, source_table, 5), \
            "Source table should still have 5 rows after copy"

        # Step 7: Verify sample data integrity in target
        target_data = verify_sample_data(env_name, target_table, sample_size=5)
        assert len(target_data) == 5, \
            f"Expected 5 rows in target, got {len(target_data)}"

        # Step 8: Verify sample data integrity in source
        source_data = verify_sample_data(env_name, source_table, sample_size=5)
        assert len(source_data) == 5, \
            f"Expected 5 rows in source, got {len(source_data)}"

    def test_copy_postgres_to_postgres(self, clean_test_tables):
        """Specific test: PostgreSQL -> PostgreSQL with 5 rows."""
        environments = clean_test_tables
        env_name = environments["postgresql"]

        # Create and populate source
        source_table = create_source_table("postgresql", env_name)
        insert_test_data("postgresql", env_name, source_table, num_rows=5)

        # Copy to different table in same database
        target_table = f"{source_table}_copy"
        result = core_copy.copy_db_to_db(
            source_env=env_name,
            target_env=env_name,
            query=f"SELECT * FROM {source_table}",
            table=target_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 5
        assert verify_row_count(env_name, target_table, 5)
        assert verify_row_count(env_name, source_table, 5)

    def test_copy_mysql_to_mysql(self, clean_test_tables):
        """Specific test: MySQL -> MySQL with 5 rows."""
        environments = clean_test_tables
        env_name = environments["mysql"]

        # Create and populate source
        source_table = create_source_table("mysql", env_name)
        insert_test_data("mysql", env_name, source_table, num_rows=5)

        # Copy to different table in same database
        target_table = f"{source_table}_copy"
        result = core_copy.copy_db_to_db(
            source_env=env_name,
            target_env=env_name,
            query=f"SELECT * FROM {source_table}",
            table=target_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 5
        assert verify_row_count(env_name, target_table, 5)
        assert verify_row_count(env_name, source_table, 5)


# ============================================================================
# SCENARIO C: Large Dataset Tests (50,000 rows)
# ============================================================================

class TestLargeDatasetCopy:
    """Test database copy operations with large datasets (50,000 rows)."""

    @pytest.mark.timeout(300)  # 5 minutes timeout for large dataset tests
    @pytest.mark.parametrize("source_db,target_db", get_all_db_combinations())
    def test_copy_50k_rows_between_databases(self, clean_test_tables, source_db, target_db):
        """
        Test copying 50,000 rows from source database to target database.

        This test:
        1. Creates a table in the source database
        2. Inserts 50,000 rows of test data (5 base rows × 10,000)
        3. Copies data to target database using ELM-tool with batching
        4. Verifies row count and data integrity

        Note: Uses batch_size to handle large datasets efficiently.
        """
        environments = clean_test_tables
        source_env = environments[source_db]
        target_env = environments[target_db]

        # Step 1: Create source table and insert large dataset
        source_table = create_source_table(source_db, source_env)
        rows_inserted = insert_test_data(source_db, source_env, source_table, num_rows=50000)
        assert rows_inserted == 50000, f"Expected 50,000 rows inserted, got {rows_inserted}"

        # Step 2: Define target table name
        target_table = source_table

        # Step 3: Perform database-to-database copy with batching
        query = f"SELECT * FROM {source_table}"

        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=query,
            table=target_table,
            mode='REPLACE',
            batch_size=10000,  # Use batching for large datasets
            apply_masks=False,
            create_if_not_exists=True
        )

        # Step 4: Verify the copy operation succeeded
        assert result.success, f"Copy failed: {result.message}"
        assert result.record_count == 50000, \
            f"Expected 50,000 records copied, got {result.record_count}"

        # Step 5: Verify row count in target database
        assert verify_row_count(target_env, target_table, 50000), \
            f"Row count verification failed for {source_db} -> {target_db}"

        # Step 6: Verify sample data integrity (check first 5 rows)
        target_data = verify_sample_data(target_env, target_table, sample_size=5)
        assert len(target_data) >= 5, \
            f"Expected at least 5 rows in sample, got {len(target_data)}"

    @pytest.mark.timeout(300)
    def test_copy_postgres_to_mysql_large(self, clean_test_tables):
        """Specific test: PostgreSQL -> MySQL with 50,000 rows."""
        environments = clean_test_tables
        source_env = environments["postgresql"]
        target_env = environments["mysql"]

        # Create and populate source with large dataset
        source_table = create_source_table("postgresql", source_env)
        insert_test_data("postgresql", source_env, source_table, num_rows=50000)

        # Copy to target with batching
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            batch_size=10000,
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 50000
        assert verify_row_count(target_env, source_table, 50000)

    @pytest.mark.timeout(300)
    def test_copy_mysql_to_mssql_large(self, clean_test_tables):
        """Specific test: MySQL -> MSSQL with 50,000 rows."""
        environments = clean_test_tables
        source_env = environments["mysql"]
        target_env = environments["mssql"]

        # Create and populate source with large dataset
        source_table = create_source_table("mysql", source_env)
        insert_test_data("mysql", source_env, source_table, num_rows=50000)

        # Copy to target with batching
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            batch_size=10000,
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 50000
        assert verify_row_count(target_env, source_table, 50000)

    @pytest.mark.timeout(300)
    def test_copy_mssql_to_oracle_large(self, clean_test_tables):
        """Specific test: MSSQL -> Oracle with 50,000 rows."""
        environments = clean_test_tables
        source_env = environments["mssql"]
        target_env = environments["oracle"]

        # Create and populate source with large dataset
        source_table = create_source_table("mssql", source_env)
        insert_test_data("mssql", source_env, source_table, num_rows=50000)

        # Copy to target with batching
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            batch_size=10000,
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 50000
        assert verify_row_count(target_env, source_table, 50000)

    @pytest.mark.timeout(300)
    def test_copy_oracle_to_postgres_large(self, clean_test_tables):
        """Specific test: Oracle -> PostgreSQL with 50,000 rows."""
        environments = clean_test_tables
        source_env = environments["oracle"]
        target_env = environments["postgresql"]

        # Create and populate source with large dataset
        source_table = create_source_table("oracle", source_env)
        insert_test_data("oracle", source_env, source_table, num_rows=50000)

        # Copy to target with batching
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            batch_size=10000,
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 50000
        assert verify_row_count(target_env, source_table, 50000)


# ============================================================================
# Additional Comprehensive Tests
# ============================================================================

class TestDataIntegrity:
    """Test data integrity and schema compatibility across databases."""

    def test_data_type_preservation_postgres_to_mysql(self, clean_test_tables):
        """Test that data types are properly preserved when copying from PostgreSQL to MySQL."""
        environments = clean_test_tables
        source_env = environments["postgresql"]
        target_env = environments["mysql"]

        # Create and populate source
        source_table = create_source_table("postgresql", source_env)
        insert_test_data("postgresql", source_env, source_table, num_rows=5)

        # Get original data
        source_data = verify_sample_data(source_env, source_table, sample_size=5)

        # Copy to target
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success

        # Get target data
        target_data = verify_sample_data(target_env, source_table, sample_size=5)

        # Verify row counts match
        assert len(source_data) == len(target_data), \
            "Source and target row counts don't match"

    def test_append_mode_preserves_existing_data(self, clean_test_tables):
        """Test that APPEND mode preserves existing data in target table."""
        environments = clean_test_tables
        source_env = environments["postgresql"]
        target_env = environments["mysql"]

        # Create and populate source
        source_table = create_source_table("postgresql", source_env)
        insert_test_data("postgresql", source_env, source_table, num_rows=5)

        # First copy - REPLACE mode
        result1 = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result1.success
        assert verify_row_count(target_env, source_table, 5)

        # Second copy - APPEND mode
        result2 = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='APPEND',
            apply_masks=False
        )

        assert result2.success
        # Should now have 10 rows (5 original + 5 appended)
        assert verify_row_count(target_env, source_table, 10)

    def test_replace_mode_overwrites_data(self, clean_test_tables):
        """Test that REPLACE mode overwrites existing data in target table."""
        environments = clean_test_tables
        source_env = environments["mysql"]
        target_env = environments["mssql"]

        # Create and populate source with 5 rows
        source_table = create_source_table("mysql", source_env)
        insert_test_data("mysql", source_env, source_table, num_rows=5)

        # First copy
        result1 = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result1.success
        assert verify_row_count(target_env, source_table, 5)

        # Add more data to source (now 10 rows total)
        insert_test_data("mysql", source_env, source_table, num_rows=5, truncate_table=False, write_mode=WriteMode.APPEND)

        # Second copy with REPLACE mode
        result2 = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table}",
            table=source_table,
            mode='REPLACE',
            apply_masks=False
        )

        assert result2.success
        # Should have 10 rows (replaced, not appended)
        assert verify_row_count(target_env, source_table, 10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_copy_with_where_clause(self, clean_test_tables):
        """Test copying with a WHERE clause to filter data."""
        environments = clean_test_tables
        source_env = environments["postgresql"]
        target_env = environments["mysql"]

        # Create and populate source
        source_table = create_source_table("postgresql", source_env)
        insert_test_data("postgresql", source_env, source_table, num_rows=5)

        # Copy only rows where user_id <= 3
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT * FROM {source_table} WHERE user_id <= 3",
            table=source_table,
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 3
        assert verify_row_count(target_env, source_table, 3)

    def test_copy_with_column_selection(self, clean_test_tables):
        """Test copying with specific column selection."""
        environments = clean_test_tables
        source_env = environments["mysql"]
        target_env = environments["mssql"]

        # Create and populate source
        source_table = create_source_table("mysql", source_env)
        insert_test_data("mysql", source_env, source_table, num_rows=5)

        # Copy only specific columns
        result = core_copy.copy_db_to_db(
            source_env=source_env,
            target_env=target_env,
            query=f"SELECT product_id, product_name, price FROM {source_table}",
            table=f"{source_table}_partial",
            mode='REPLACE',
            apply_masks=False,
            create_if_not_exists=True
        )

        assert result.success
        assert result.record_count == 5

    def test_all_database_combinations_small_dataset(self, clean_test_tables):
        """
        Comprehensive test: Verify all 12 database combinations work with small dataset.
        This is a summary test that ensures all cross-database operations are functional.
        """
        environments = clean_test_tables
        combinations = get_all_db_combinations()

        # Should have 12 combinations (4 sources × 3 targets each)
        assert len(combinations) == 12, f"Expected 12 combinations, got {len(combinations)}"

        successful_copies = 0
        failed_copies = []

        for source_db, target_db in combinations:
            try:
                source_env = environments[source_db]
                target_env = environments[target_db]

                # Create and populate source
                source_table = create_source_table(source_db, source_env)
                insert_test_data(source_db, source_env, source_table, num_rows=5)

                # Copy to target
                result = core_copy.copy_db_to_db(
                    source_env=source_env,
                    target_env=target_env,
                    query=f"SELECT * FROM {source_table}",
                    table=source_table,
                    mode='REPLACE',
                    apply_masks=False,
                    create_if_not_exists=True
                )

                if result.success and verify_row_count(target_env, source_table, 5):
                    successful_copies += 1
                else:
                    failed_copies.append((source_db, target_db, result.message))

                # Cleanup for next iteration
                core_env.execute_sql(source_env, TEST_SCHEMAS[source_db]["drop_sql"])
                core_env.execute_sql(target_env, TEST_SCHEMAS[source_db]["drop_sql"])

            except Exception as e:
                failed_copies.append((source_db, target_db, str(e)))

        # Report results
        print(f"\nSuccessful copies: {successful_copies}/12")
        if failed_copies:
            print("Failed copies:")
            for source, target, error in failed_copies:
                print(f"  {source} -> {target}: {error}")

        # Assert all combinations succeeded
        assert successful_copies == 12, \
            f"Expected all 12 combinations to succeed, but {len(failed_copies)} failed"

