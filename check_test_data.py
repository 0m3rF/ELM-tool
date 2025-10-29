"""Quick script to check if test data persists after tests."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elm.core import environment as core_env

# Database configurations
DB_CONFIGS = {
    "postgresql": {
        "host": "localhost",
        "port": 5433,
        "user": "ELM_TOOL_user",
        "password": "ELM_TOOL_password",
        "service": "ELM_TOOL_db",
        "database": "POSTGRES"
    },
    "mysql": {
        "host": "localhost",
        "port": 3307,
        "user": "ELM_TOOL_user",
        "password": "ELM_TOOL_password",
        "service": "ELM_TOOL_db",
        "database": "MYSQL"
    }
}

def check_table_data(db_name, table_name):
    """Check if a table exists and has data."""
    config = DB_CONFIGS[db_name]
    env_name = f"temp_check_{db_name}"
    
    # Create environment
    result = core_env.create_environment(
        name=env_name,
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        service=config["service"],
        database=config["database"]
    )
    
    if not result.success:
        print(f"❌ Failed to create environment for {db_name}: {result.message}")
        return
    
    # Check table
    try:
        count_result = core_env.execute_sql(env_name, f"SELECT COUNT(*) FROM {table_name}")
        if count_result.success:
            print(f"✅ {db_name}.{table_name}: {count_result.data} rows")
        else:
            print(f"❌ {db_name}.{table_name}: {count_result.message}")
    except Exception as e:
        print(f"❌ {db_name}.{table_name}: {e}")
    finally:
        core_env.delete_environment(env_name)

if __name__ == "__main__":
    print("Checking test data persistence...\n")
    
    # Check PostgreSQL tables
    print("PostgreSQL:")
    check_table_data("postgresql", "test_users_pg")
    check_table_data("postgresql", "test_users_pg_copy")
    
    print("\nMySQL:")
    check_table_data("mysql", "test_products_mysql")
    check_table_data("mysql", "test_products_mysql_copy")
    
    print("\nDone!")

