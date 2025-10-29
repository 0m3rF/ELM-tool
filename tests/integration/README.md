# ELM-Tool Integration Tests

This directory contains comprehensive integration tests for ELM-tool's database copying functionality across all supported database systems.

## Overview

The integration tests verify that ELM-tool can successfully copy data between all supported database systems:
- **PostgreSQL** (port 5433)
- **MySQL** (port 3307)
- **MSSQL** (port 1434)
- **Oracle** (port 1522)

## Test Coverage

### Test Scenarios

#### Scenario A: Small Dataset Tests (5 rows)
- Tests copying 5 rows of data between databases
- Verifies basic functionality and schema compatibility
- **12 test combinations**: 4 source databases × 3 destination databases

#### Scenario B: Large Dataset Tests (500,000 rows)
- Tests copying 500,000 rows of data between databases
- Verifies performance and batch processing
- Tests timeout handling (25-second limit per pytest configuration)
- **12 test combinations**: 4 source databases × 3 destination databases

### Test Classes

1. **TestSmallDatasetCopy**: Tests with 5-row datasets
   - `test_copy_5_rows_between_databases`: Parametrized test for all 12 combinations
   - Specific tests for key database pairs (PostgreSQL→MySQL, MySQL→MSSQL, etc.)

2. **TestLargeDatasetCopy**: Tests with 500,000-row datasets
   - `test_copy_500k_rows_between_databases`: Parametrized test for all 12 combinations
   - Specific tests for key database pairs with large datasets
   - Uses batch_size=10000 for efficient processing

3. **TestDataIntegrity**: Data integrity and mode testing
   - Data type preservation tests
   - APPEND mode tests (preserves existing data)
   - REPLACE mode tests (overwrites existing data)

4. **TestEdgeCases**: Edge cases and advanced queries
   - WHERE clause filtering
   - Column selection
   - Comprehensive all-combinations test

## Database Schemas

Each database uses a different table schema to test schema compatibility:

- **PostgreSQL**: `test_users_pg` - User data with various data types
- **MySQL**: `test_products_mysql` - Product catalog with TEXT and DATETIME
- **MSSQL**: `test_orders_mssql` - Order data with NVARCHAR and DECIMAL
- **Oracle**: `test_inventory_ora` - Inventory data with NUMBER and VARCHAR2

## Prerequisites

### 1. Database Containers

All database containers must be running. Use the database setup tool:

```bash
# From the project root
cd database_setup
python database_setup.py setup
```

Or setup specific databases:

```bash
python database_setup.py setup -d postgresql -d mysql -d mssql -d oracle
```

### 2. Database Drivers

Install all database drivers:

```bash
pip install psycopg2-binary pymysql pyodbc oracledb
```

Or install from project dependencies:

```bash
pip install -e ".[all-db]"
```

### 3. Environment Variables (Optional)

You can override default passwords using environment variables:

```bash
export ELM_TOOL_TEST_POSTGRES_PASS="your_password"
export ELM_TOOL_TEST_MYSQL_PASS="your_password"
export ELM_TOOL_TEST_MSSQL_PASS="your_password"
export ELM_TOOL_TEST_ORACLE_PASS="your_password"
```

## Running the Tests

### Run All Integration Tests

```bash
# From project root
pytest tests/integration/test_db_copy_integration.py -v
```

### Run Specific Test Classes

```bash
# Small dataset tests only
pytest tests/integration/test_db_copy_integration.py::TestSmallDatasetCopy -v

# Large dataset tests only
pytest tests/integration/test_db_copy_integration.py::TestLargeDatasetCopy -v

# Data integrity tests
pytest tests/integration/test_db_copy_integration.py::TestDataIntegrity -v

# Edge case tests
pytest tests/integration/test_db_copy_integration.py::TestEdgeCases -v
```

### Run Specific Test Cases

```bash
# Test specific database combination (small dataset)
pytest tests/integration/test_db_copy_integration.py::TestSmallDatasetCopy::test_copy_postgres_to_mysql_small -v

# Test specific database combination (large dataset)
pytest tests/integration/test_db_copy_integration.py::TestLargeDatasetCopy::test_copy_postgres_to_mysql_large -v

# Test all combinations summary
pytest tests/integration/test_db_copy_integration.py::TestEdgeCases::test_all_database_combinations_small_dataset -v
```

### Run with Coverage

```bash
pytest tests/integration/test_db_copy_integration.py --cov=elm.core.copy --cov-report=html -v
```

### Run Parametrized Tests for Specific Combinations

```bash
# Run small dataset test for PostgreSQL to MySQL
pytest tests/integration/test_db_copy_integration.py::TestSmallDatasetCopy::test_copy_5_rows_between_databases[postgresql-mysql] -v

# Run large dataset test for MySQL to MSSQL
pytest tests/integration/test_db_copy_integration.py::TestLargeDatasetCopy::test_copy_500k_rows_between_databases[mysql-mssql] -v
```

## Test Execution Time

- **Small dataset tests**: ~1-2 seconds per combination
- **Large dataset tests**: ~15-20 seconds per combination
- **Total execution time**: Approximately 5-10 minutes for all tests

All tests are configured with a 25-second timeout to comply with pytest configuration.

## Test Data

### Small Dataset (5 rows)
Each database has 5 unique sample records with different data types:
- Integers, strings, decimals, dates, booleans
- Different column names and types per database

### Large Dataset (500,000 rows)
The 5 base records are replicated 100,000 times with unique IDs:
- Tests batch processing capabilities
- Verifies streaming performance
- Ensures timeout handling works correctly

## Verification

Each test performs the following verifications:

1. **Copy Success**: Verifies the copy operation returns success
2. **Record Count**: Verifies the correct number of records were copied
3. **Row Count Verification**: Queries the target database to confirm row count
4. **Sample Data Verification**: Retrieves sample data to verify data integrity

## Troubleshooting

### Database Connection Issues

If tests fail with connection errors:

1. Verify all database containers are running:
   ```bash
   cd database_setup
   python database_setup.py status
   ```

2. Test individual database connections:
   ```bash
   python database_setup.py debug postgresql
   python database_setup.py debug mysql
   python database_setup.py debug mssql
   python database_setup.py debug oracle
   ```

3. Check database logs:
   ```bash
   docker logs ELM_TOOL_postgresql
   docker logs ELM_TOOL_mysql
   docker logs ELM_TOOL_mssql
   docker logs ELM_TOOL_oracle
   ```

### Timeout Issues

If tests timeout:

1. Reduce the dataset size for testing:
   - Modify `num_rows` parameter in test functions
   - Use smaller batch sizes

2. Increase timeout in pytest.ini (current: 25 seconds)

3. Run tests individually instead of all at once

### Schema Issues

If schema creation fails:

1. Manually drop tables:
   ```sql
   -- PostgreSQL
   DROP TABLE IF EXISTS test_users_pg CASCADE;
   
   -- MySQL
   DROP TABLE IF EXISTS test_products_mysql;
   
   -- MSSQL
   DROP TABLE IF EXISTS test_orders_mssql;
   
   -- Oracle
   DROP TABLE test_inventory_ora;
   ```

2. Check database permissions for the test user

## Test Results

Expected results for a successful test run:

- ✅ All 12 small dataset combinations pass
- ✅ All 12 large dataset combinations pass
- ✅ Data integrity tests pass
- ✅ Edge case tests pass
- ✅ Total: ~50+ test cases pass

## Contributing

When adding new integration tests:

1. Follow the existing test structure
2. Use the provided fixtures (`setup_test_environments`, `clean_test_tables`)
3. Add proper assertions for verification
4. Document any new test scenarios
5. Ensure tests complete within the 25-second timeout

## Notes

- Tests are self-contained and can run in any order
- Each test cleans up its own data (setup and teardown)
- Tests use ELM-tool's API directly for maximum reliability
- No external dependencies (Kafka, etc.) are required
- Tests verify both functionality and performance

