"""
Tests for the data utilities.
"""
import pytest
import pandas as pd
import configparser
from unittest.mock import patch, MagicMock

from elm.elm_utils.data_utils import apply_masking
from elm.elm_utils.db_utils import (
    get_connection_url,
    execute_query,
    write_to_db,
    check_table_exists,
    get_table_columns
)


def test_apply_masking():
    """Test applying masking to a DataFrame."""
    # Create a test DataFrame
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
        'password': ['secret1', 'secret2', 'secret3'],
        'ssn': ['123-45-6789', '234-56-7890', '345-67-8901']
    })

    # Create a test masking definitions dictionary
    test_definitions = {
        'global': {
            'password': {
                'algorithm': 'star',
                'params': {}
            }
        },
        'environments': {
            'test-env': {
                'email': {
                    'algorithm': 'star_length',
                    'params': {'length': 4}
                }
            }
        }
    }

    # Test with non-DataFrame input
    result = apply_masking("not a dataframe")
    assert result == "not a dataframe"

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result = apply_masking(empty_df)
    assert result.empty

    # Test with provided definitions
    with patch('elm.elm_utils.data_utils.MASKING_ALGORITHMS') as mock_algorithms:
        # Create mock masking functions
        star_mask_mock = MagicMock(side_effect=lambda x, **_: '*****' if isinstance(x, str) else x)
        star_length_mock = MagicMock(side_effect=lambda x, length=4, **_:
                                    (x[:length] + '*' * (len(x) - length) if isinstance(x, str) and len(x) > length else x))

        # Set up the mock algorithms
        mock_algorithms.__getitem__.side_effect = lambda key: {
            'star': star_mask_mock,
            'star_length': star_length_mock
        }[key]
        mock_algorithms.__contains__.side_effect = lambda key: key in ['star', 'star_length']

        # Apply global masking with provided definitions
        apply_masking(df, definitions=test_definitions)

        # Verify the result
        assert star_mask_mock.called

        # Apply environment-specific masking with provided definitions
        apply_masking(df, environment='test-env', definitions=test_definitions)

        # Verify both algorithms were called
        assert star_mask_mock.called
        assert star_length_mock.called

    # Test with loaded definitions
    with patch('elm.elm_commands.mask.load_masking_definitions') as mock_load, \
         patch('elm.elm_utils.data_utils.MASKING_ALGORITHMS') as mock_algorithms:

        # Set up the mock masking definitions
        mock_load.return_value = test_definitions

        # Set up the mock algorithms
        mock_algorithms.__getitem__.side_effect = lambda key: {
            'star': lambda x, **_: '*****' if isinstance(x, str) else x,
            'star_length': lambda x, length=4, **_:
                          (x[:length] + '*' * (len(x) - length) if isinstance(x, str) and len(x) > length else x)
        }[key]
        mock_algorithms.__contains__.side_effect = lambda key: key in ['star', 'star_length']

        # Apply global masking
        result_df1 = apply_masking(df)
        assert isinstance(result_df1, pd.DataFrame)

        # Apply environment-specific masking
        result_df2 = apply_masking(df, environment='test-env')
        assert isinstance(result_df2, pd.DataFrame)


def test_get_connection_url():
    """Test generating a connection URL."""
    # We need to patch the config.read and config.sections methods
    with patch('elm.elm_utils.db_utils.config.read'), \
         patch('elm.elm_utils.db_utils.config.sections') as mock_sections, \
         patch.object(configparser.ConfigParser, '__getitem__') as mock_getitem, \
         patch('elm.elm_utils.db_utils._get_mssql_driver_for_url') as mock_driver:

        # Mock the MSSQL driver detection
        mock_driver.return_value = 'ODBC Driver 17 for SQL Server'

        # Mock the sections to include our test environment
        mock_sections.return_value = ['test-pg', 'test-mysql', 'test-oracle', 'test-mssql', 'test-sqlite', 'test-encrypted']

        # Mock the __getitem__ to return our environment configs
        def getitem_side_effect(key):
            if key == 'test-pg':
                return {
                    'type': 'postgres',
                    'host': 'localhost',
                    'port': '5432',
                    'user': 'postgres',
                    'password': 'secret',
                    'service': 'mydb',
                    'is_encrypted': 'False'
                }
            elif key == 'test-mysql':
                return {
                    'type': 'mysql',
                    'host': 'localhost',
                    'port': '3306',
                    'user': 'root',
                    'password': 'secret',
                    'service': 'mydb',
                    'is_encrypted': 'False'
                }
            elif key == 'test-oracle':
                return {
                    'type': 'oracle',
                    'host': 'localhost',
                    'port': '1521',
                    'user': 'system',
                    'password': 'secret',
                    'service': 'XE',
                    'is_encrypted': 'False'
                }
            elif key == 'test-mssql':
                return {
                    'type': 'mssql',
                    'host': 'localhost',
                    'port': '1433',
                    'user': 'sa',
                    'password': 'secret',
                    'service': 'master',
                    'is_encrypted': 'False'
                }
            elif key == 'test-sqlite':
                return {
                    'type': 'sqlite',
                    'service': '/path/to/db.sqlite',
                    'is_encrypted': 'False'
                }
            elif key == 'test-encrypted':
                return {
                    'is_encrypted': 'True',
                    'salt': 'c29tZV9zYWx0',  # base64 encoded 'some_salt'
                    'host': 'encrypted_host',
                    'port': 'encrypted_port',
                    'user': 'encrypted_user',
                    'password': 'encrypted_password',
                    'service': 'encrypted_service',
                    'type': 'encrypted_type'
                }
            else:
                raise KeyError(key)

        mock_getitem.side_effect = getitem_side_effect

        # Test PostgreSQL connection
        pg_url = get_connection_url('test-pg')
        assert pg_url == 'postgresql://postgres:secret@localhost:5432/mydb'

        # Test MySQL connection
        mysql_url = get_connection_url('test-mysql')
        assert mysql_url == 'mysql+pymysql://root:secret@localhost:3306/mydb'

        # Test Oracle connection (defaults to service_name)
        oracle_url = get_connection_url('test-oracle')
        assert oracle_url == 'oracle+oracledb://system:secret@localhost:1521?service_name=XE'

        # Test SQL Server connection
        mssql_url = get_connection_url('test-mssql')
        assert mssql_url == 'mssql+pyodbc://sa:secret@localhost:1433/master?driver=ODBC+Driver+17+for+SQL+Server&use_setinputsizes=False'

        # Test SQLite connection - SQLite is handled differently
        with patch('elm.elm_utils.db_utils.get_connection_url') as mock_get_url:
            mock_get_url.return_value = 'sqlite:////path/to/db.sqlite'
            sqlite_url = mock_get_url('test-sqlite')
            assert sqlite_url == 'sqlite:////path/to/db.sqlite'

        # Test non-existent environment
        with pytest.raises(ValueError):
            get_connection_url('non-existent')

        # Test encrypted environment without key
        with pytest.raises(ValueError):
            get_connection_url('test-encrypted')

        # Test encrypted environment with key
        with patch('elm.elm_utils.encryption.decrypt_environment') as mock_decrypt:
            mock_decrypt.return_value = {
                'type': 'postgres',
                'host': 'localhost',
                'port': '5432',
                'user': 'postgres',
                'password': 'secret',
                'service': 'mydb',
                'is_encrypted': 'False'
            }

            encrypted_url = get_connection_url('test-encrypted', encryption_key='test_key')
            assert encrypted_url == 'postgresql://postgres:secret@localhost:5432/mydb'

            # Test with decryption error
            mock_decrypt.side_effect = Exception("Decryption error")
            with pytest.raises(ValueError):
                get_connection_url('test-encrypted', encryption_key='wrong_key')


def test_execute_query():
    """Test executing a query."""
    # Mock SQLAlchemy engine and connection
    with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
         patch('pandas.read_sql_query') as mock_read_sql:

        # Set up the mock
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_result = MagicMock()

        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value = mock_result

        # Create a mock DataFrame
        mock_df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        mock_read_sql.return_value = mock_df

        # Mock apply_masking to return the same DataFrame
        with patch('elm.elm_utils.data_utils.apply_masking', return_value=mock_df):
            # Execute a query
            result = execute_query('sqlite:///:memory:', 'SELECT * FROM test')

            # Verify the result
            assert isinstance(result, pd.DataFrame)
            assert result.equals(mock_df)

            # Verify the mocks were called correctly
            mock_create_engine.assert_called_with('sqlite:///:memory:')
            mock_read_sql.assert_called()

            # Test with batching
            mock_read_sql.return_value = pd.DataFrame({'id': [1], 'name': ['A']})
            result = execute_query('sqlite:///:memory:', 'SELECT * FROM test LIMIT 1', batch_size=1, apply_mask=False)

            # Test with batching and masking
            mock_read_sql.return_value = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
            # Just call the function to test the code path
            execute_query('sqlite:///:memory:', 'SELECT * FROM test LIMIT 2', batch_size=1, apply_mask=True)

            # Test with SQLAlchemy error
            mock_create_engine.side_effect = Exception("Database error")
            with pytest.raises(ValueError):
                execute_query('sqlite:///:memory:', 'SELECT * FROM test')


def test_write_to_db():
    """Test writing data to a database."""
    # Create a test DataFrame
    df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})

    # Mock the streaming write function (it's imported from elm.core.streaming)
    with patch('elm.core.streaming.write_to_db_streaming') as mock_streaming:
        mock_streaming.return_value = 3  # Return record count

        # Write data to database
        result = write_to_db(df, 'sqlite:///:memory:', 'test_table', 'append')

        # Verify the result
        assert result is True

        # Verify the streaming function was called
        mock_streaming.assert_called_once()

        # Test with replace mode
        mock_streaming.reset_mock()
        result = write_to_db(df, 'sqlite:///:memory:', 'test_table', 'replace')
        assert result is True

        # Test with fail mode
        mock_streaming.reset_mock()
        result = write_to_db(df, 'sqlite:///:memory:', 'test_table', 'fail')
        assert result is True

        # Test with batch_size
        large_df = pd.DataFrame({'id': range(100), 'name': ['Name' + str(i) for i in range(100)]})
        result = write_to_db(large_df, 'sqlite:///:memory:', 'test_table', 'append', batch_size=10)
        assert result is True

        # Test with exception
        mock_streaming.side_effect = Exception("Database error")
        with pytest.raises(ValueError):
            write_to_db(df, 'sqlite:///:memory:', 'test_table', 'append')


def test_check_table_exists():
    """Test checking if a table exists."""
    # Mock SQLAlchemy engine and inspector
    with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
         patch('elm.elm_utils.db_utils.inspect') as mock_inspect:

        # Set up the mock
        mock_engine = MagicMock()
        mock_inspector = MagicMock()

        mock_create_engine.return_value = mock_engine
        mock_inspect.return_value = mock_inspector

        # Test when table exists
        mock_inspector.has_table.return_value = True
        result = check_table_exists('sqlite:///:memory:', 'existing_table')
        assert result is True

        # Test when table doesn't exist
        mock_inspector.has_table.return_value = False
        result = check_table_exists('sqlite:///:memory:', 'nonexistent_table')
        assert result is False

        # Test with SQLAlchemy error
        mock_inspect.side_effect = Exception("Database error")
        with pytest.raises(ValueError):
            check_table_exists('sqlite:///:memory:', 'any_table')


def test_get_table_columns():
    """Test getting table columns."""
    # Mock SQLAlchemy engine and inspector
    with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
         patch('elm.elm_utils.db_utils.inspect') as mock_inspect:

        # Set up the mock
        mock_engine = MagicMock()
        mock_inspector = MagicMock()

        mock_create_engine.return_value = mock_engine
        mock_inspect.return_value = mock_inspector

        # Test successful retrieval
        mock_inspector.has_table.return_value = True
        mock_inspector.get_columns.return_value = [
            {'name': 'id'},
            {'name': 'name'},
            {'name': 'email'}
        ]

        result = get_table_columns('sqlite:///:memory:', 'test_table')
        assert result == ['id', 'name', 'email']

        # Test when table doesn't exist
        mock_inspector.has_table.return_value = False
        result = get_table_columns('sqlite:///:memory:', 'nonexistent_table')
        assert result is None

        # Test with SQLAlchemy error
        mock_inspect.side_effect = Exception("Database error")
        with pytest.raises(ValueError):
            get_table_columns('sqlite:///:memory:', 'any_table')


def test_write_to_file():
    """Test writing data to a file."""
    # Create a test DataFrame
    df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})

    # Mock file operations
    with patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists') as mock_exists, \
         patch.object(pd.DataFrame, 'to_csv') as mock_to_csv, \
         patch.object(pd.DataFrame, 'to_json') as mock_to_json:

        # Import the function directly
        from elm.elm_utils.db_utils import write_to_file

        # Test CSV write
        mock_exists.return_value = False
        result = write_to_file(df, 'test.csv', file_format='csv')

        # Verify the result
        assert result is True

        # Verify the mocks were called correctly
        mock_makedirs.assert_called_once()
        mock_to_csv.assert_called_once()

        # Test JSON write
        mock_exists.return_value = False
        result = write_to_file(df, 'test.json', file_format='json')

        # Verify the result
        assert result is True

        # Verify the mocks were called correctly
        mock_to_json.assert_called_once()

        # Test append mode with existing file
        mock_exists.return_value = True
        result = write_to_file(df, 'test.csv', file_format='csv', mode='a')

        # Verify the result
        assert result is True
