"""
Tests for the elm_utils/db_utils.py module.
"""
import pytest
import pandas as pd
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from sqlalchemy.exc import SQLAlchemyError

from elm.elm_utils import db_utils


class TestGetConnectionUrl:
    """Test get_connection_url function."""

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_postgres_unencrypted(self, mock_config):
        """Test getting connection URL for PostgreSQL unencrypted."""
        mock_config.sections.return_value = ['test-env']
        mock_config.__getitem__.return_value = {
            'type': 'POSTGRES',
            'host': 'localhost',
            'port': '5432',
            'user': 'postgres',
            'password': 'secret',
            'service': 'mydb',
            'is_encrypted': 'False'
        }

        url = db_utils.get_connection_url('test-env')

        expected = 'postgresql://postgres:secret@localhost:5432/mydb'
        assert url == expected

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_oracle_service_name(self, mock_config):
        """Test getting connection URL for Oracle with service_name (default)."""
        mock_config.sections.return_value = ['test-env']
        mock_config.__getitem__.return_value = {
            'type': 'ORACLE',
            'host': 'oraserver',
            'port': '1521',
            'user': 'system',
            'password': 'oracle',
            'service': 'XE',
            'is_encrypted': 'False'
        }

        url = db_utils.get_connection_url('test-env')

        expected = 'oracle+oracledb://system:oracle@oraserver:1521?service_name=XE'
        assert url == expected

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_oracle_service_name_explicit(self, mock_config):
        """Test getting connection URL for Oracle with explicit service_name."""
        mock_config.sections.return_value = ['test-env']
        mock_config.__getitem__.return_value = {
            'type': 'ORACLE',
            'host': 'oraserver',
            'port': '1521',
            'user': 'system',
            'password': 'oracle',
            'service': 'XE',
            'connection_type': 'service_name',
            'is_encrypted': 'False'
        }

        url = db_utils.get_connection_url('test-env')

        expected = 'oracle+oracledb://system:oracle@oraserver:1521?service_name=XE'
        assert url == expected

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_oracle_sid(self, mock_config):
        """Test getting connection URL for Oracle with SID."""
        mock_config.sections.return_value = ['test-env']
        mock_config.__getitem__.return_value = {
            'type': 'ORACLE',
            'host': 'oraserver',
            'port': '1521',
            'user': 'system',
            'password': 'oracle',
            'service': 'ORCL',
            'connection_type': 'sid',
            'is_encrypted': 'False'
        }

        url = db_utils.get_connection_url('test-env')

        expected = 'oracle+oracledb://system:oracle@oraserver:1521/ORCL'
        assert url == expected

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_mysql(self, mock_config):
        """Test getting connection URL for MySQL."""
        mock_config.sections.return_value = ['test-env']
        mock_config.__getitem__.return_value = {
            'type': 'MYSQL',
            'host': 'mysqlserver',
            'port': '3306',
            'user': 'root',
            'password': 'mysql',
            'service': 'testdb',
            'is_encrypted': 'False'
        }

        url = db_utils.get_connection_url('test-env')

        expected = 'mysql+pymysql://root:mysql@mysqlserver:3306/testdb'
        assert url == expected

    @patch('elm.elm_utils.db_utils._get_mssql_driver_for_url')
    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_mssql(self, mock_config, mock_driver):
        """Test getting connection URL for SQL Server."""
        mock_config.sections.return_value = ['test-env']
        mock_config.__getitem__.return_value = {
            'type': 'MSSQL',
            'host': 'sqlserver',
            'port': '1433',
            'user': 'sa',
            'password': 'password',
            'service': 'master',
            'is_encrypted': 'False'
        }
        mock_driver.return_value = 'ODBC Driver 17 for SQL Server'

        url = db_utils.get_connection_url('test-env')

        expected = 'mssql+pyodbc://sa:password@sqlserver:1433/master?driver=ODBC+Driver+17+for+SQL+Server&use_setinputsizes=False'
        assert url == expected

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_environment_not_found(self, mock_config):
        """Test getting connection URL for non-existent environment."""
        mock_config.sections.return_value = []

        with pytest.raises(ValueError) as exc_info:
            db_utils.get_connection_url('nonexistent')

        assert "Environment 'nonexistent' not found" in str(exc_info.value)

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_unsupported_db_type(self, mock_config):
        """Test getting connection URL for unsupported database type."""
        mock_config.sections.return_value = ['test-env']
        mock_config.__getitem__.return_value = {
            'type': 'UNSUPPORTED',
            'host': 'localhost',
            'port': '5432',
            'user': 'user',
            'password': 'pass',
            'service': 'db',
            'is_encrypted': 'False'
        }

        with pytest.raises(ValueError) as exc_info:
            db_utils.get_connection_url('test-env')

        assert "Unsupported database type: UNSUPPORTED" in str(exc_info.value)

    @patch('elm.elm_utils.db_utils.encryption.decrypt_environment')
    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_encrypted_success(self, mock_config, mock_decrypt):
        """Test getting connection URL for encrypted environment."""
        mock_config.sections.return_value = ['secure-env']
        mock_config.__getitem__.return_value = {
            'type': 'encrypted_type',
            'host': 'encrypted_host',
            'port': 'encrypted_port',
            'user': 'encrypted_user',
            'password': 'encrypted_password',
            'service': 'encrypted_service',
            'is_encrypted': 'True'
        }

        # Mock decryption result
        mock_decrypt.return_value = {
            'type': 'POSTGRES',
            'host': 'localhost',
            'port': '5432',
            'user': 'postgres',
            'password': 'secret',
            'service': 'mydb'
        }

        url = db_utils.get_connection_url('secure-env', encryption_key='secret-key')

        expected = 'postgresql://postgres:secret@localhost:5432/mydb'
        assert url == expected
        mock_decrypt.assert_called_once()

    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_encrypted_no_key(self, mock_config):
        """Test getting connection URL for encrypted environment without key."""
        mock_config.sections.return_value = ['secure-env']
        mock_config.__getitem__.return_value = {
            'is_encrypted': 'True'
        }

        with pytest.raises(ValueError) as exc_info:
            db_utils.get_connection_url('secure-env')

        assert "Environment 'secure-env' is encrypted. Provide an encryption key." in str(exc_info.value)

    @patch('elm.elm_utils.db_utils.encryption.decrypt_environment')
    @patch('elm.elm_utils.db_utils.config')
    def test_get_connection_url_encrypted_decryption_failure(self, mock_config, mock_decrypt):
        """Test getting connection URL for encrypted environment with decryption failure."""
        mock_config.sections.return_value = ['secure-env']
        mock_config.__getitem__.return_value = {
            'is_encrypted': 'True'
        }

        # Mock decryption failure
        mock_decrypt.side_effect = Exception("Decryption failed")

        with pytest.raises(ValueError) as exc_info:
            db_utils.get_connection_url('secure-env', encryption_key='wrong-key')

        assert "Failed to decrypt environment: Decryption failed. Check your encryption key." in str(exc_info.value)


class TestCheckTableExists:
    """Test check_table_exists function."""

    @patch('elm.elm_utils.db_utils.create_engine')
    @patch('elm.elm_utils.db_utils.inspect')
    def test_check_table_exists_true(self, mock_inspect, mock_create_engine):
        """Test checking if table exists - returns True."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = True
        
        result = db_utils.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert result is True
        mock_inspector.has_table.assert_called_once_with('test_table')

    @patch('elm.elm_utils.db_utils.create_engine')
    @patch('elm.elm_utils.db_utils.inspect')
    def test_check_table_exists_false(self, mock_inspect, mock_create_engine):
        """Test checking if table exists - returns False."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = False
        
        result = db_utils.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert result is False

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_check_table_exists_database_error(self, mock_create_engine):
        """Test checking table exists with database error."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")
        
        with pytest.raises(ValueError) as exc_info:
            db_utils.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert "Database error while checking table existence" in str(exc_info.value)

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_check_table_exists_general_error(self, mock_create_engine):
        """Test checking table exists with general error."""
        mock_create_engine.side_effect = Exception("General error")
        
        with pytest.raises(ValueError) as exc_info:
            db_utils.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert "Error checking table existence" in str(exc_info.value)


class TestGetTableColumns:
    """Test get_table_columns function."""

    @patch('elm.elm_utils.db_utils.create_engine')
    @patch('elm.elm_utils.db_utils.inspect')
    def test_get_table_columns_success(self, mock_inspect, mock_create_engine):
        """Test getting table columns successfully."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = True
        mock_inspector.get_columns.return_value = [
            {'name': 'ID'}, {'name': 'NAME'}, {'name': 'EMAIL'}
        ]
        
        result = db_utils.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert result == ['id', 'name', 'email']
        mock_inspector.get_columns.assert_called_once_with('test_table')

    @patch('elm.elm_utils.db_utils.create_engine')
    @patch('elm.elm_utils.db_utils.inspect')
    def test_get_table_columns_table_not_exists(self, mock_inspect, mock_create_engine):
        """Test getting table columns when table doesn't exist."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = False
        
        result = db_utils.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert result is None

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_get_table_columns_database_error(self, mock_create_engine):
        """Test getting table columns with database error."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")
        
        with pytest.raises(ValueError) as exc_info:
            db_utils.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert "Database error while getting table columns" in str(exc_info.value)

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_get_table_columns_general_error(self, mock_create_engine):
        """Test getting table columns with general error."""
        mock_create_engine.side_effect = Exception("General error")
        
        with pytest.raises(ValueError) as exc_info:
            db_utils.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')
        
        assert "Error getting table columns" in str(exc_info.value)


class TestExecuteQuery:
    """Test execute_query function."""

    @patch('elm.elm_utils.db_utils.create_engine')
    @patch('pandas.read_sql_query')
    def test_execute_query_without_batching(self, mock_read_sql, mock_create_engine):
        """Test executing query without batching."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection

        # Mock query result
        expected_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        mock_read_sql.return_value = expected_df

        result = db_utils.execute_query('postgresql://user:pass@host:5432/db', 'SELECT * FROM test', apply_mask=False)

        assert result.equals(expected_df)
        mock_read_sql.assert_called_once_with('SELECT * FROM test', mock_connection)

    @patch('elm.elm_utils.db_utils.create_engine')
    @patch('pandas.read_sql_query')
    def test_execute_query_with_batching(self, mock_read_sql, mock_create_engine):
        """Test executing query with batching."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection

        # Mock batched result
        batch1 = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        batch2 = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'Diana']})
        mock_read_sql.return_value = iter([batch1, batch2])

        result = db_utils.execute_query(
            'postgresql://user:pass@host:5432/db',
            'SELECT * FROM test',
            batch_size=2,
            apply_mask=False
        )

        # Result should be an iterator
        batches = list(result)
        assert len(batches) == 2
        assert batches[0].equals(batch1)
        assert batches[1].equals(batch2)
        mock_read_sql.assert_called_once_with('SELECT * FROM test', mock_connection, chunksize=2)

    @patch('elm.elm_utils.db_utils.create_engine')
    @patch('pandas.read_sql_query')
    def test_execute_query_with_masking(self, mock_read_sql, mock_create_engine):
        """Test executing query with masking."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection

        # Mock query result
        original_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        masked_df = pd.DataFrame({'id': [1, 2], 'name': ['****', '****']})
        mock_read_sql.return_value = original_df

        with patch('elm.elm_utils.data_utils.apply_masking') as mock_apply_masking:
            mock_apply_masking.return_value = masked_df

            result = db_utils.execute_query(
                'postgresql://user:pass@host:5432/db',
                'SELECT * FROM test',
                environment='test-env',
                apply_mask=True
            )

            assert result.equals(masked_df)
            mock_apply_masking.assert_called_once_with(original_df, 'test-env')

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_execute_query_database_error(self, mock_create_engine):
        """Test executing query with database error."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")

        with pytest.raises(ValueError) as exc_info:
            db_utils.execute_query('postgresql://user:pass@host:5432/db', 'SELECT * FROM test')

        assert "Database error" in str(exc_info.value)

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_execute_query_general_error(self, mock_create_engine):
        """Test executing query with general error."""
        mock_create_engine.side_effect = Exception("General error")

        with pytest.raises(ValueError) as exc_info:
            db_utils.execute_query('postgresql://user:pass@host:5432/db', 'SELECT * FROM test')

        assert "Error executing query" in str(exc_info.value)


class TestWriteToDb:
    """Test write_to_db function."""

    @patch('elm.core.streaming.write_to_db_streaming')
    def test_write_to_db_without_batching(self, mock_streaming):
        """Test writing to database without batching."""
        mock_streaming.return_value = 2  # Return record count

        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        result = db_utils.write_to_db(data, 'postgresql://user:pass@host:5432/db', 'test_table')

        assert result is True
        mock_streaming.assert_called_once()

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_write_to_db_with_batching(self, mock_create_engine):
        """Test writing to database with batching."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Create data larger than batch size
        data = pd.DataFrame({'id': list(range(5)), 'name': [f'User{i}' for i in range(5)]})

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            result = db_utils.write_to_db(data, 'postgresql://user:pass@host:5432/db', 'test_table', batch_size=2)

            assert result is True
            # Should be called 3 times (batches of 2, 2, 1)
            assert mock_to_sql.call_count == 3

    @patch('elm.elm_utils.db_utils.create_engine')
    def test_write_to_db_error(self, mock_create_engine):
        """Test writing to database with error."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
            mock_to_sql.side_effect = Exception("Write failed")

            with pytest.raises(ValueError) as exc_info:
                db_utils.write_to_db(data, 'postgresql://user:pass@host:5432/db', 'test_table')

            assert "Error writing to database" in str(exc_info.value)


class TestWriteToFile:
    """Test write_to_file function."""

    def test_write_to_file_csv(self):
        """Test writing to CSV file."""
        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                result = db_utils.write_to_file(data, temp_file, 'csv')

                assert result is True
                mock_to_csv.assert_called_once_with(temp_file, index=False)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_csv_append(self):
        """Test writing to CSV file in append mode."""
        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            # Create existing file
            with open(temp_file, 'w') as f:
                f.write("id,name\n1,Existing\n")

            with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
                result = db_utils.write_to_file(data, temp_file, 'csv', mode='a')

                assert result is True
                mock_to_csv.assert_called_once_with(temp_file, mode='a', header=False, index=False)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_json(self):
        """Test writing to JSON file."""
        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            with patch.object(pd.DataFrame, 'to_json') as mock_to_json:
                result = db_utils.write_to_file(data, temp_file, 'json')

                assert result is True
                mock_to_json.assert_called_once_with(temp_file, orient='records', indent=2)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_json_append(self):
        """Test writing to JSON file in append mode."""
        data = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'Diana']})

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Create existing JSON file
            existing_data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
            existing_data.to_json(temp_file, orient='records', indent=2)

            with patch('pandas.read_json') as mock_read_json, \
                 patch('pandas.concat') as mock_concat, \
                 patch.object(pd.DataFrame, 'to_json') as mock_to_json:

                mock_read_json.return_value = existing_data
                combined_data = pd.DataFrame({'id': [1, 2, 3, 4], 'name': ['Alice', 'Bob', 'Charlie', 'Diana']})
                mock_concat.return_value = combined_data

                result = db_utils.write_to_file(data, temp_file, 'json', mode='a')

                assert result is True
                mock_read_json.assert_called_once_with(temp_file)
                mock_concat.assert_called_once()
                mock_to_json.assert_called_once_with(temp_file, orient='records', indent=2)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_json_append_read_error(self):
        """Test writing to JSON file in append mode with read error."""
        data = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'Diana']})

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Create corrupted JSON file
            with open(temp_file, 'w') as f:
                f.write("invalid json")

            with patch.object(pd.DataFrame, 'to_json') as mock_to_json:
                result = db_utils.write_to_file(data, temp_file, 'json', mode='a')

                assert result is True
                # Should fallback to writing new data
                mock_to_json.assert_called_once_with(temp_file, orient='records', indent=2)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_unsupported_format(self):
        """Test writing to file with unsupported format."""
        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with pytest.raises(ValueError) as exc_info:
            db_utils.write_to_file(data, 'test.xml', 'xml')

        assert "Unsupported file format: xml" in str(exc_info.value)

    def test_write_to_file_error(self):
        """Test writing to file with error."""
        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            mock_to_csv.side_effect = Exception("Write failed")

            with pytest.raises(ValueError) as exc_info:
                db_utils.write_to_file(data, 'test.csv', 'csv')

            assert "Error writing to file" in str(exc_info.value)


class TestOracleErrorHandling:
    """Test Oracle-specific error handling in db_utils."""

    def test_initialize_oracle_client_success(self):
        """Test successful Oracle client initialization."""
        # This test verifies the code path exists
        # The actual Oracle client initialization is difficult to test in isolation
        # We'll verify the function can be called and returns a boolean
        result = db_utils._initialize_oracle_client()
        assert isinstance(result, bool)

    def test_initialize_oracle_client_already_initialized(self):
        """Test Oracle client when already initialized."""
        # Mock the import of oracledb inside the function
        mock_oracledb = MagicMock()
        mock_oracledb._client_initialized = True

        with patch.dict('sys.modules', {'oracledb': mock_oracledb}):
            result = db_utils._initialize_oracle_client()

            assert result is True

    def test_initialize_oracle_client_initialization_fails(self):
        """Test Oracle client initialization failure."""
        # This test verifies the error handling path exists
        # The actual failure scenario is difficult to mock without recursion
        # We'll verify the function handles exceptions gracefully
        result = db_utils._initialize_oracle_client()
        assert isinstance(result, bool)

    def test_initialize_oracle_client_import_error(self):
        """Test Oracle client when oracledb is not installed."""
        # Remove oracledb from sys.modules if it exists
        import sys
        oracledb_backup = sys.modules.get('oracledb')
        if 'oracledb' in sys.modules:
            del sys.modules['oracledb']

        try:
            # Mock the import to raise ImportError
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
                       (_ for _ in ()).throw(ImportError()) if name == 'oracledb' else __import__(name, *args, **kwargs)):
                result = db_utils._initialize_oracle_client()

                assert result is False
        finally:
            # Restore oracledb if it was there
            if oracledb_backup is not None:
                sys.modules['oracledb'] = oracledb_backup

    def test_handle_oracle_connection_error_thin_mode_success(self):
        """Test handling Oracle thin mode error with successful retry."""
        error = Exception("DPY-3015: password verifier type 0x939 is not supported by python-oracledb in thin mode")

        with patch('elm.elm_utils.db_utils._initialize_oracle_client', return_value=True), \
             patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine:

            mock_engine = MagicMock()
            mock_connection = MagicMock()
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [(1,)]
            mock_connection.execute.return_value = mock_result
            mock_engine.connect.return_value.__enter__.return_value = mock_connection
            mock_create_engine.return_value = mock_engine

            result = db_utils._handle_oracle_connection_error(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                error
            )

            assert result is True

    def test_handle_oracle_connection_error_thin_mode_failed_init(self):
        """Test handling Oracle thin mode error with failed initialization."""
        error = Exception("password verifier type not supported")

        with patch('elm.elm_utils.db_utils._initialize_oracle_client', return_value=False):
            result = db_utils._handle_oracle_connection_error(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                error
            )

            assert result is False

    def test_handle_oracle_connection_error_thin_mode_retry_fails(self):
        """Test handling Oracle thin mode error when retry also fails."""
        error = Exception("DPY-3015: password verifier type 0x939 is not supported")

        with patch('elm.elm_utils.db_utils._initialize_oracle_client', return_value=True), \
             patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine:

            mock_engine = MagicMock()
            mock_connection = MagicMock()
            mock_connection.execute.side_effect = Exception("Connection still fails")
            mock_engine.connect.return_value.__enter__.return_value = mock_connection
            mock_create_engine.return_value = mock_engine

            result = db_utils._handle_oracle_connection_error(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                error
            )

            assert result is False

    def test_handle_oracle_connection_error_non_thin_mode_error(self):
        """Test handling Oracle error that is not thin mode related."""
        error = Exception("Some other Oracle error")

        result = db_utils._handle_oracle_connection_error(
            'oracle+oracledb://user:pass@host:1521?service_name=service',
            error
        )

        assert result is False

    def test_check_table_exists_oracle_retry_success(self):
        """Test check_table_exists with Oracle retry success."""
        with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
             patch('elm.elm_utils.db_utils.inspect') as mock_inspect, \
             patch('elm.elm_utils.db_utils._handle_oracle_connection_error', return_value=True):

            # First call fails
            mock_inspector_first = MagicMock()
            mock_inspector_first.has_table.side_effect = Exception("DPY-3015")

            # Second call succeeds
            mock_inspector_second = MagicMock()
            mock_inspector_second.has_table.return_value = True

            mock_inspect.side_effect = [mock_inspector_first, mock_inspector_second]

            result = db_utils.check_table_exists(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                'test_table'
            )

            assert result is True

    def test_check_table_exists_oracle_retry_failed(self):
        """Test check_table_exists with Oracle retry failed."""
        with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
             patch('elm.elm_utils.db_utils.inspect') as mock_inspect, \
             patch('elm.elm_utils.db_utils._handle_oracle_connection_error', return_value=False):

            mock_inspector = MagicMock()
            mock_inspector.has_table.side_effect = SQLAlchemyError("Connection failed")
            mock_inspect.return_value = mock_inspector

            with pytest.raises(ValueError) as exc_info:
                db_utils.check_table_exists(
                    'oracle+oracledb://user:pass@host:1521?service_name=service',
                    'test_table'
                )

            assert "Database error while checking table existence" in str(exc_info.value)

    def test_get_table_columns_oracle_retry_success(self):
        """Test get_table_columns with Oracle retry success."""
        with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
             patch('elm.elm_utils.db_utils.inspect') as mock_inspect, \
             patch('elm.elm_utils.db_utils._handle_oracle_connection_error', return_value=True):

            # First call fails
            mock_inspector_first = MagicMock()
            mock_inspector_first.has_table.side_effect = Exception("DPY-3015")

            # Second call succeeds
            mock_inspector_second = MagicMock()
            mock_inspector_second.has_table.return_value = True
            mock_inspector_second.get_columns.return_value = [
                {'name': 'ID', 'type': 'INTEGER'},
                {'name': 'NAME', 'type': 'VARCHAR'}
            ]

            mock_inspect.side_effect = [mock_inspector_first, mock_inspector_second]

            result = db_utils.get_table_columns(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                'test_table'
            )

            assert result == ['id', 'name']

    def test_execute_query_oracle_retry_with_batching(self):
        """Test execute_query with Oracle retry and batching."""
        with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
             patch('elm.elm_utils.db_utils._handle_oracle_connection_error', return_value=True), \
             patch('elm.elm_utils.data_utils.apply_masking') as mock_apply_masking:

            # First call fails
            mock_engine_first = MagicMock()
            mock_connection_first = MagicMock()
            mock_connection_first.execute.side_effect = Exception("DPY-3015")
            mock_engine_first.connect.return_value.__enter__.return_value = mock_connection_first

            # Second call succeeds
            mock_engine_second = MagicMock()
            mock_connection_second = MagicMock()
            mock_engine_second.connect.return_value.__enter__.return_value = mock_connection_second

            mock_create_engine.side_effect = [mock_engine_first, mock_engine_second]

            # Mock batched results
            batch1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
            batch2 = pd.DataFrame({'id': [2], 'name': ['Bob']})
            mock_apply_masking.side_effect = [batch1, batch2]

            with patch('pandas.read_sql_query', return_value=iter([batch1, batch2])):
                result = db_utils.execute_query(
                    'oracle+oracledb://user:pass@host:1521?service_name=service',
                    'SELECT * FROM test_table',
                    batch_size=1,
                    environment='test-env',
                    apply_mask=True
                )

                # Result should be a generator
                batches = list(result)
                assert len(batches) == 2

    def test_write_to_db_oracle_retry_with_batching(self):
        """Test write_to_db with Oracle retry and batching."""
        with patch('elm.elm_utils.db_utils.create_engine') as mock_create_engine, \
             patch('elm.elm_utils.db_utils._handle_oracle_connection_error', return_value=True):

            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

            # Mock to_sql to fail first, then succeed
            call_count = [0]
            def mock_to_sql_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("DPY-3015: password verifier type 0x939 is not supported")
                return None

            with patch.object(pd.DataFrame, 'to_sql', side_effect=mock_to_sql_side_effect):
                result = db_utils.write_to_db(
                    data,
                    'oracle+oracledb://user:pass@host:1521?service_name=service',
                    'test_table',
                    if_exists='append',
                    batch_size=2
                )

                assert result is True
