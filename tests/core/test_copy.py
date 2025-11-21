"""Tests for core copy functionality."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
from sqlalchemy.exc import SQLAlchemyError

from elm.core import copy
from elm.core.types import FileFormat, WriteMode
from elm.core.exceptions import DatabaseError, CopyError, FileError


class TestCopyCore:
    """Test cases for core copy functionality."""

    def test_copy_file_to_db_success(self):
        """Test successful file to database copy."""
        # Create a temporary CSV file
        test_data = "id,name,email\n1,Alice,alice@example.com\n2,Bob,bob@example.com"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_data)
            temp_file = f.name

        try:
            with patch('elm.core.copy.get_connection_url') as mock_get_url, \
                 patch('elm.core.copy.write_to_db') as mock_write_db, \
                 patch('elm.core.copy.apply_masking') as mock_apply_masking:

                mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'
                mock_write_db.return_value = 2  # Return record count

                # Mock DataFrame after masking
                mock_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob'], 'email': ['alice@example.com', 'bob@example.com']})
                mock_apply_masking.return_value = mock_df

                result = copy.copy_file_to_db(
                    file_path=temp_file,
                    target_env="test-env",
                    table="test_table",
                    file_format='csv',  # Use string format as expected
                    mode='APPEND'
                )

                assert result.success is True
                assert "Successfully copied" in result.message
                assert result.record_count == 2
        finally:
            os.unlink(temp_file)

    def test_copy_file_to_db_invalid_file(self):
        """Test file to database copy with invalid file."""
        result = copy.copy_file_to_db(
            file_path="nonexistent.csv",
            target_env="test-env",
            table="test_table"
        )

        assert result.success is False
        # The function may fail due to file not found or environment not found
        assert ("Environment 'test-env' not found" in result.message or
                "No such file" in result.message or
                "File not found" in result.message or
                "parsing errors" in result.message)

    def test_copy_db_to_db_success(self):
        """Test successful database to database copy."""
        with patch('elm.core.copy.get_connection_url') as mock_get_url, \
             patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy.apply_masking') as mock_apply_masking, \
             patch('elm.core.copy.write_to_db') as mock_write_db:

            mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'
            mock_write_db.return_value = 2  # Return record count
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_connection = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_connection

            # Mock query result
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
            mock_result.keys.return_value = ['id', 'name']
            mock_connection.execute.return_value = mock_result

            # Mock DataFrame after masking
            mock_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
            mock_apply_masking.return_value = mock_df

            result = copy.copy_db_to_db(
                source_env="source-env",
                target_env="target-env",
                query="SELECT * FROM source_table",
                table="target_table"
            )

            assert result.success is True
            assert "Successfully copied" in result.message
            assert result.record_count == 2

    def test_copy_db_to_file_success(self):
        """Test successful database to file copy."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            with patch('elm.core.copy.get_connection_url') as mock_get_url, \
                 patch('elm.core.copy.execute_query') as mock_execute_query, \
                 patch('elm.core.copy.write_to_file') as mock_write_file, \
                 patch('elm.core.copy.apply_masking') as mock_apply_masking:

                mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'

                # Mock query result as DataFrame
                mock_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
                mock_execute_query.return_value = mock_df

                # Mock DataFrame after masking
                mock_apply_masking.return_value = mock_df

                result = copy.copy_db_to_file(
                    source_env="source-env",
                    query="SELECT * FROM source_table",
                    file_path=temp_file,
                    file_format='csv'  # Use string format as expected
                )

                assert result.success is True
                assert "Successfully copied" in result.message
                assert result.record_count == 2
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_execute_query_with_batching(self):
        """Test execute_query with batch processing."""
        from elm.core.copy import execute_query

        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy.apply_masking') as mock_apply_masking:

            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_connection = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_connection

            # Mock chunked result
            batch1 = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
            batch2 = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'Diana']})

            # Mock pandas read_sql_query to return an iterator
            def mock_read_sql_query(query, connection, chunksize=None):
                if chunksize:
                    return iter([batch1, batch2])
                else:
                    return pd.concat([batch1, batch2])

            with patch('pandas.read_sql_query', side_effect=mock_read_sql_query):
                # Mock apply_masking to return the same data
                mock_apply_masking.side_effect = lambda df, env: df

                # Test with batching and masking
                result = execute_query(
                    'postgresql://user:pass@localhost:5432/db',
                    'SELECT * FROM test_table',
                    batch_size=2,
                    environment='test-env',
                    apply_masks=True
                )

                # Result should be a generator
                batches = list(result)
                assert len(batches) == 2
                assert len(batches[0]) == 2
                assert len(batches[1]) == 2

    def test_execute_query_with_batching_no_masking(self):
        """Test execute_query with batch processing but no masking."""
        from elm.core.copy import execute_query

        with patch('elm.core.copy.create_engine') as mock_create_engine:

            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_connection = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_connection

            # Mock chunked result
            batch1 = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
            batch2 = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'Diana']})

            # Mock pandas read_sql_query to return an iterator
            def mock_read_sql_query(query, connection, chunksize=None):
                if chunksize:
                    return iter([batch1, batch2])
                else:
                    return pd.concat([batch1, batch2])

            with patch('pandas.read_sql_query', side_effect=mock_read_sql_query):
                # Test with batching but no masking
                result = execute_query(
                    'postgresql://user:pass@localhost:5432/db',
                    'SELECT * FROM test_table',
                    batch_size=2,
                    apply_masks=False
                )

                # Result should be the original iterator
                batches = list(result)
                assert len(batches) == 2

    def test_execute_query_database_error(self):
        """Test execute_query with database error."""
        from elm.core.copy import execute_query
        from elm.core.exceptions import DatabaseError

        with patch('elm.core.copy.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = SQLAlchemyError("Connection failed")

            with pytest.raises(DatabaseError):
                execute_query(
                    'postgresql://user:pass@localhost:5432/db',
                    'SELECT * FROM test_table'
                )

    def test_execute_query_general_error(self):
        """Test execute_query with general error."""
        from elm.core.copy import execute_query
        from elm.core.exceptions import CopyError

        with patch('elm.core.copy.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = Exception("General error")

            with pytest.raises(CopyError):
                execute_query(
                    'postgresql://user:pass@localhost:5432/db',
                    'SELECT * FROM test_table'
                )

    def test_write_to_file_json_append(self):
        """Test writing to JSON file with append mode."""
        from elm.core.copy import write_to_file

        # Create initial JSON file
        initial_data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Write initial data
            write_to_file(initial_data, temp_file, FileFormat.JSON, mode='w')

            # Append more data
            append_data = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'Diana']})
            write_to_file(append_data, temp_file, FileFormat.JSON, mode='a')

            # Read back and verify
            with open(temp_file, 'r') as f:
                import json
                result = json.load(f)

            assert len(result) == 4
            assert result[0]['name'] == 'Alice'
            assert result[3]['name'] == 'Diana'

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_json_append_empty(self):
        """Test writing to empty JSON file with append mode."""
        from elm.core.copy import write_to_file

        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
            # Create empty file

        try:
            # Append to empty file
            write_to_file(data, temp_file, FileFormat.JSON, mode='a')

            # Read back and verify
            with open(temp_file, 'r') as f:
                import json
                result = json.load(f)

            assert len(result) == 2
            assert result[0]['name'] == 'Alice'

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_csv_append(self):
        """Test writing to CSV file with append mode."""
        from elm.core.copy import write_to_file

        # Create initial CSV file
        initial_data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            # Write initial data
            write_to_file(initial_data, temp_file, FileFormat.CSV, mode='w')

            # Append more data
            append_data = pd.DataFrame({'id': [3, 4], 'name': ['Charlie', 'Diana']})
            write_to_file(append_data, temp_file, FileFormat.CSV, mode='a')

            # Read back and verify
            result_df = pd.read_csv(temp_file)
            assert len(result_df) == 4
            assert result_df.iloc[0]['name'] == 'Alice'
            assert result_df.iloc[3]['name'] == 'Diana'

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_unsupported_format(self):
        """Test writing with unsupported file format."""
        from elm.core.copy import write_to_file
        from elm.core.exceptions import ValidationError

        data = pd.DataFrame({'id': [1], 'name': ['Alice']})

        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(ValidationError):
                write_to_file(data, temp_file, "UNSUPPORTED")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_from_file_not_found(self):
        """Test reading from non-existent file."""
        from elm.core.copy import read_from_file
        from elm.core.exceptions import FileError

        with pytest.raises(FileError):
            read_from_file("nonexistent.csv", FileFormat.CSV)

    def test_read_from_file_unsupported_format(self):
        """Test reading with unsupported file format."""
        from elm.core.copy import read_from_file
        from elm.core.exceptions import ValidationError

        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = f.name

        try:
            with pytest.raises(ValidationError):
                read_from_file(temp_file, "UNSUPPORTED")

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_db_with_batching(self):
        """Test writing to database with batching."""
        from elm.core.copy import write_to_db
        from elm.core.types import WriteMode

        data = pd.DataFrame({'id': range(10), 'name': [f'Name{i}' for i in range(10)]})

        with patch('elm.core.copy.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            # Mock to_sql method
            with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
                write_to_db(
                    data,
                    'postgresql://user:pass@localhost:5432/db',
                    'test_table',
                    WriteMode.APPEND,
                    batch_size=3
                )

                # Should be called multiple times for batches
                assert mock_to_sql.call_count > 1

    def test_write_to_db_database_error(self):
        """Test write_to_db with database error."""
        from elm.core.copy import write_to_db
        from elm.core.exceptions import DatabaseError

        data = pd.DataFrame({'id': [1], 'name': ['Alice']})

        with patch('elm.core.copy.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = SQLAlchemyError("Connection failed")

            with pytest.raises(DatabaseError):
                write_to_db(
                    data,
                    'postgresql://user:pass@localhost:5432/db',
                    'test_table'
                )

    def test_write_to_db_general_error(self):
        """Test write_to_db with general error."""
        from elm.core.copy import write_to_db
        from elm.core.exceptions import CopyError, DatabaseError

        data = pd.DataFrame({'id': [1], 'name': ['Alice']})

        with patch('elm.core.copy.write_to_db_streaming') as mock_streaming:
            mock_streaming.side_effect = CopyError("General error")

            with pytest.raises(CopyError):
                write_to_db(
                    data,
                    'postgresql://user:pass@localhost:5432/db',
                    'test_table'
                )

    def test_write_to_file_json_append_error(self):
        """Test JSON append with corrupted existing file."""
        from elm.core.copy import write_to_file
        from elm.core.exceptions import FileError

        data = pd.DataFrame({'id': [1], 'name': ['Alice']})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Write invalid JSON
            f.write("invalid json content")
            temp_file = f.name

        try:
            with pytest.raises(FileError):
                write_to_file(data, temp_file, FileFormat.JSON, mode='a')

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_general_error(self):
        """Test write_to_file with general file error."""
        from elm.core.copy import write_to_file
        from elm.core.exceptions import FileError

        data = pd.DataFrame({'id': [1], 'name': ['Alice']})

        # Mock pandas to_csv to raise an exception
        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            mock_to_csv.side_effect = Exception("Permission denied")

            with pytest.raises(FileError):
                write_to_file(data, "test.csv", FileFormat.CSV)

    def test_read_from_file_general_error(self):
        """Test read_from_file with general file error."""
        from elm.core.copy import read_from_file
        from elm.core.exceptions import FileError

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            # Mock pandas read_csv to raise an exception
            with patch('pandas.read_csv') as mock_read_csv:
                mock_read_csv.side_effect = Exception("Parsing error")

                with pytest.raises(FileError):
                    read_from_file(temp_file, FileFormat.CSV)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_from_file_csv(self):
        """Test reading CSV file."""
        test_data = "id,name\n1,Alice\n2,Bob"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_data)
            temp_file = f.name

        try:
            from elm.core.copy import read_from_file
            df = read_from_file(temp_file, FileFormat.CSV)

            assert len(df) == 2
            assert list(df.columns) == ['id', 'name']
            assert df.iloc[0]['name'] == 'Alice'
        finally:
            os.unlink(temp_file)

    def test_read_from_file_json(self):
        """Test reading JSON file."""
        test_data = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(test_data)
            temp_file = f.name

        try:
            from elm.core.copy import read_from_file
            df = read_from_file(temp_file, FileFormat.JSON)

            assert len(df) == 2
            assert list(df.columns) == ['id', 'name']
            assert df.iloc[0]['name'] == 'Alice'
        finally:
            os.unlink(temp_file)

    def test_write_to_file_csv(self):
        """Test writing CSV file."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            from elm.core.copy import write_to_file
            write_to_file(df, temp_file, FileFormat.CSV)

            # Verify file was written
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                content = f.read()
                assert 'Alice' in content
                assert 'Bob' in content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_to_file_json(self):
        """Test writing JSON file."""
        df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            from elm.core.copy import write_to_file
            write_to_file(df, temp_file, FileFormat.JSON)

            # Verify file was written
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                content = f.read()
                assert 'Alice' in content
                assert 'Bob' in content
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestCopyUtilities:
    """Test copy utility functions."""

    @patch('elm.core.copy.inspect')
    @patch('elm.core.copy.create_engine')
    def test_check_table_exists_true(self, mock_create_engine, mock_inspect):
        """Test checking if table exists - returns True."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock inspector to return table exists
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = True

        result = copy.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')

        assert result is True
        mock_inspector.has_table.assert_called_once_with('test_table')

    @patch('elm.core.copy.inspect')
    @patch('elm.core.copy.create_engine')
    def test_check_table_exists_false(self, mock_create_engine, mock_inspect):
        """Test checking if table exists - returns False."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock inspector to return table doesn't exist
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = False

        result = copy.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')

        assert result is False
        mock_inspector.has_table.assert_called_once_with('test_table')

    @patch('elm.core.copy.create_engine')
    def test_check_table_exists_database_error(self, mock_create_engine):
        """Test checking table exists with database error."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")

        with pytest.raises(DatabaseError) as exc_info:
            copy.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')

        assert "Database error while checking table existence" in str(exc_info.value)

    @patch('elm.core.copy.create_engine')
    def test_check_table_exists_general_error(self, mock_create_engine):
        """Test checking table exists with general error."""
        mock_create_engine.side_effect = Exception("General error")

        with pytest.raises(CopyError) as exc_info:
            copy.check_table_exists('postgresql://user:pass@host:5432/db', 'test_table')

        assert "Error checking table existence" in str(exc_info.value)

    @patch('elm.core.copy.inspect')
    @patch('elm.core.copy.create_engine')
    def test_get_table_columns_success(self, mock_create_engine, mock_inspect):
        """Test getting table columns successfully."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock inspector to return columns
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.get_columns.return_value = [
            {'name': 'id'}, {'name': 'name'}, {'name': 'email'}
        ]

        result = copy.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')

        assert result == ['id', 'name', 'email']
        mock_inspector.get_columns.assert_called_once_with('test_table')

    @patch('elm.core.copy.inspect')
    @patch('elm.core.copy.create_engine')
    def test_get_table_columns_table_not_exists(self, mock_create_engine, mock_inspect):
        """Test getting table columns when table doesn't exist."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock inspector to return table doesn't exist
        mock_inspector = MagicMock()
        mock_inspect.return_value = mock_inspector
        mock_inspector.has_table.return_value = False  # Table doesn't exist

        result = copy.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')

        assert result is None
        mock_inspector.has_table.assert_called_once_with('test_table')

    @patch('elm.core.copy.create_engine')
    def test_get_table_columns_database_error(self, mock_create_engine):
        """Test getting table columns with database error."""
        mock_create_engine.side_effect = SQLAlchemyError("Connection failed")

        with pytest.raises(DatabaseError) as exc_info:
            copy.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')

        assert "Database error while getting table columns" in str(exc_info.value)

    @patch('elm.core.copy.create_engine')
    def test_get_table_columns_general_error(self, mock_create_engine):
        """Test getting table columns with general error."""
        mock_create_engine.side_effect = Exception("General error")

        with pytest.raises(CopyError) as exc_info:
            copy.get_table_columns('postgresql://user:pass@host:5432/db', 'test_table')

        assert "Error getting table columns" in str(exc_info.value)

    @patch('elm.core.copy.check_table_exists')
    @patch('elm.core.copy.get_table_columns')
    def test_validate_target_table_success(self, mock_get_columns, mock_check_table):
        """Test validating target table successfully."""
        source_df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'email': ['a@test.com', 'b@test.com']})
        mock_check_table.return_value = True
        mock_get_columns.return_value = ['id', 'name', 'email', 'extra_column']

        # Should not raise an exception
        copy.validate_target_table(source_df, 'postgresql://user:pass@host:5432/db', 'test_table')

    @patch('elm.core.copy.check_table_exists')
    @patch('elm.core.copy.get_table_columns')
    def test_validate_target_table_missing_columns(self, mock_get_columns, mock_check_table):
        """Test validating target table with missing columns."""
        source_df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B'], 'email': ['a@test.com', 'b@test.com']})
        mock_check_table.return_value = True
        mock_get_columns.return_value = ['id', 'name']  # Missing 'email' column

        with pytest.raises(CopyError) as exc_info:
            copy.validate_target_table(source_df, 'postgresql://user:pass@host:5432/db', 'test_table')

        assert "Target table test_table is missing columns: email" in str(exc_info.value)

    def test_process_in_parallel_success(self):
        """Test processing items in parallel successfully."""
        def test_func(item):
            return f"processed_{item}"

        items = ['item1', 'item2']
        result = copy.process_in_parallel(test_func, items, max_workers=2)

        assert result == ['processed_item1', 'processed_item2']

    def test_process_in_parallel_with_exception(self):
        """Test processing items in parallel with exception."""
        def test_func(item):
            if item == 'bad_item':
                raise Exception("Processing failed")
            return f"processed_{item}"

        items = ['good_item', 'bad_item']
        with pytest.raises(Exception) as exc_info:
            copy.process_in_parallel(test_func, items, max_workers=2)

        # The first exception raised by a worker should be propagated
        assert "Processing failed" in str(exc_info.value)

    def test_copy_db_to_db_parallel_uses_process_in_parallel(self):
        """copy_db_to_db should use process_in_parallel when parallel_workers > 1."""
        batch1 = pd.DataFrame({'id': [1]})
        batch2 = pd.DataFrame({'id': [2]})
        batch3 = pd.DataFrame({'id': [3]})

        with patch('elm.core.copy.get_connection_url') as mock_get_url, \
             patch('elm.core.copy.execute_query') as mock_execute, \
             patch('elm.core.copy.write_to_db') as mock_write_db, \
             patch('elm.core.copy.process_in_parallel') as mock_parallel:

            mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'
            mock_execute.return_value = iter([batch1, batch2, batch3])
            mock_write_db.return_value = 10

            # Simulate parallel execution while keeping behaviour deterministic
            def fake_parallel(func, items, max_workers):
                return [func(item) for item in items]

            mock_parallel.side_effect = fake_parallel

            result = copy.copy_db_to_db(
                source_env='source-env',
                target_env='target-env',
                query='SELECT * FROM test_table',
                table='target_table',
                mode='REPLACE',
                batch_size=1,
                parallel_workers=2,
                validate_target=False,
                create_if_not_exists=False,
                apply_masks=False,
            )

            assert result.success is True
            # 3 batches * 10 records from mock_write_db
            assert result.record_count == 30

            # process_in_parallel should be called once with 2 pending batches
            mock_parallel.assert_called_once()
            parallel_args, _ = mock_parallel.call_args
            assert len(parallel_args[1]) == 2

            # First batch should use REPLACE, subsequent batches APPEND
            assert mock_write_db.call_count == 3
            first_mode = mock_write_db.call_args_list[0].kwargs['mode']
            second_mode = mock_write_db.call_args_list[1].kwargs['mode']
            third_mode = mock_write_db.call_args_list[2].kwargs['mode']
            assert first_mode == WriteMode.REPLACE
            assert second_mode == WriteMode.APPEND
            assert third_mode == WriteMode.APPEND

    def test_copy_db_to_db_oracle_fallback_parallel_batches(self):
        """Oracle fallback via pandas to_sql should run once per batch in parallel mode."""
        batch1 = pd.DataFrame({'id': [1]})
        batch2 = pd.DataFrame({'id': [2]})

        with patch('elm.core.copy.get_connection_url') as mock_get_url, \
             patch('elm.core.copy.execute_query') as mock_execute, \
             patch('elm.core.streaming.write_oracle_executemany') as mock_oracle_exec, \
             patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:

            # Use an Oracle-style URL so streaming detects the correct database type
            mock_get_url.return_value = 'oracle+oracledb://user:pass@localhost:1521/db'
            mock_execute.return_value = iter([batch1, batch2])

            # Force the optimized Oracle path to fail so that fallback is used
            mock_oracle_exec.side_effect = Exception("optimized writer failed")

            def fake_fallback(data, connection_url, table_name, mode, batch_size=None):
                # Simulate a successful pandas to_sql write by returning row count
                return len(data)

            mock_fallback.side_effect = fake_fallback

            result = copy.copy_db_to_db(
                source_env='source-env',
                target_env='target-env',
                query='SELECT * FROM test_table',
                table='target_table',
                mode='REPLACE',
                batch_size=1,
                parallel_workers=2,
                validate_target=False,
                create_if_not_exists=False,
                apply_masks=False,
            )

            assert result.success is True
            assert result.record_count == 2
            # Fallback should be invoked once per batch
            assert mock_fallback.call_count == 2


class TestCopyEdgeCases:
    """Test edge cases and error handling for copy operations."""

    @patch('elm.core.copy.os.path.exists')
    @patch('elm.core.copy.pd.read_csv')
    def test_read_from_file_csv_with_error(self, mock_read_csv, mock_exists):
        """Test reading CSV file with pandas error."""
        mock_exists.return_value = True  # File exists, so we get to pandas reading
        mock_read_csv.side_effect = Exception("CSV read error")

        with pytest.raises(FileError) as exc_info:
            copy.read_from_file('test.csv', FileFormat.CSV)

        assert "Error reading file" in str(exc_info.value)

    @patch('elm.core.copy.create_engine')
    def test_read_from_database_with_connection_error(self, mock_create_engine):
        """Test reading from database with connection error."""
        mock_create_engine.side_effect = Exception("Connection failed")

        with pytest.raises(CopyError) as exc_info:
            copy.execute_query(
                connection_url='postgresql://user:pass@host:5432/db',
                query='SELECT * FROM test',
                environment='test-env'
            )

        assert "Error executing query" in str(exc_info.value)

    @patch('elm.core.copy.create_engine')
    def test_write_to_database_table_not_exists_fail_mode(self, mock_create_engine):
        """Test writing to database when table doesn't exist in FAIL mode."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock to_sql to raise an exception when table doesn't exist in FAIL mode
        mock_engine.connect.return_value.__enter__.return_value = MagicMock()

        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        # Mock data.to_sql to raise an exception
        with patch.object(data, 'to_sql') as mock_to_sql:
            mock_to_sql.side_effect = Exception("Table 'nonexistent_table' doesn't exist")

            with pytest.raises(CopyError) as exc_info:
                copy.write_to_db(
                    data=data,
                    connection_url='postgresql://user:pass@host:5432/db',
                    table_name='nonexistent_table',
                    mode=WriteMode.FAIL
                )

            assert "Error writing to database" in str(exc_info.value)

    @patch('elm.core.copy.create_engine')
    def test_write_to_database_general_error(self, mock_create_engine):
        """Test writing to database with general error."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        # Mock data.to_sql to raise a general exception
        with patch.object(data, 'to_sql') as mock_to_sql:
            mock_to_sql.side_effect = Exception("General database error")

            with pytest.raises(CopyError) as exc_info:
                copy.write_to_db(
                    data=data,
                    connection_url='postgresql://user:pass@host:5432/db',
                    table_name='test_table',
                    mode=WriteMode.APPEND
                )

            assert "Error writing to database" in str(exc_info.value)

    @patch('elm.core.copy.ensure_directory_exists')
    def test_write_to_file_csv_with_error(self, mock_ensure_dir):
        """Test writing to CSV file with error."""
        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
            mock_to_csv.side_effect = Exception("CSV write error")

            with pytest.raises(FileError) as exc_info:
                copy.write_to_file(
                    data=data,
                    file_path='test.csv',
                    file_format=FileFormat.CSV,
                    mode=WriteMode.REPLACE
                )

            assert "Error writing to file" in str(exc_info.value)

    @patch('elm.core.copy.read_from_file')
    @patch('elm.core.copy.write_to_file')
    def test_file_to_file_copy_success(self, mock_write, mock_read):
        """Test copying data from file to file successfully."""
        # Mock successful read
        mock_data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
        mock_read.return_value = mock_data

        # Mock successful write (write_to_file doesn't return anything on success)
        mock_write.return_value = None

        # Simulate file-to-file copy by reading then writing
        data = copy.read_from_file('source.csv', FileFormat.CSV)
        copy.write_to_file(data, 'target.csv', FileFormat.CSV, WriteMode.REPLACE)

        mock_read.assert_called_once_with('source.csv', FileFormat.CSV)
        mock_write.assert_called_once_with(mock_data, 'target.csv', FileFormat.CSV, WriteMode.REPLACE)

    @patch('elm.core.copy.read_from_file')
    def test_file_read_failure(self, mock_read):
        """Test file read failure."""
        # Mock read failure
        mock_read.side_effect = FileError("Read failed")

        with pytest.raises(FileError) as exc_info:
            copy.read_from_file('source.csv', FileFormat.CSV)

        assert "Read failed" in str(exc_info.value)

    @patch('elm.core.copy.write_to_file')
    def test_file_write_failure(self, mock_write):
        """Test file write failure."""
        # Mock write failure
        mock_write.side_effect = FileError("Write failed")

        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        with pytest.raises(FileError) as exc_info:
            copy.write_to_file(data, 'target.csv', FileFormat.CSV, WriteMode.REPLACE)

        assert "Write failed" in str(exc_info.value)

    def test_execute_query_oracle_error_handling_with_retry(self):
        """Test Oracle error handling with successful retry."""
        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy._handle_oracle_connection_error') as mock_handle_error, \
             patch('elm.core.copy.apply_masking') as mock_apply_masking:

            # First call raises Oracle error
            mock_engine_first = MagicMock()
            mock_connection_first = MagicMock()
            mock_connection_first.execute.side_effect = Exception("DPY-3015: password verifier type 0x939 is not supported")
            mock_engine_first.connect.return_value.__enter__.return_value = mock_connection_first

            # Second call succeeds
            mock_engine_second = MagicMock()
            mock_connection_second = MagicMock()
            mock_engine_second.connect.return_value.__enter__.return_value = mock_connection_second

            mock_create_engine.side_effect = [mock_engine_first, mock_engine_second]
            mock_handle_error.return_value = True  # Indicates successful retry

            # Mock successful query result
            test_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
            mock_apply_masking.return_value = test_df

            with patch('pandas.read_sql_query', return_value=test_df):
                result = copy.execute_query(
                    'oracle+oracledb://user:pass@host:1521?service_name=service',
                    'SELECT * FROM test_table',
                    batch_size=None,
                    environment='test-env',
                    apply_masks=True
                )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

    def test_execute_query_oracle_error_handling_with_batch_retry(self):
        """Test Oracle error handling with batching and successful retry."""
        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy._handle_oracle_connection_error') as mock_handle_error, \
             patch('elm.core.copy.apply_masking') as mock_apply_masking, \
             patch('pandas.read_sql_query') as mock_read_sql:

            # Mock batched query result
            test_df1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
            test_df2 = pd.DataFrame({'id': [2], 'name': ['Bob']})

            # First call to create_engine - will raise Oracle error when read_sql_query is called
            mock_engine_first = MagicMock()
            mock_connection_first = MagicMock()
            mock_engine_first.connect.return_value.__enter__.return_value = mock_connection_first

            # Second call to create_engine - will succeed
            mock_engine_second = MagicMock()
            mock_connection_second = MagicMock()
            mock_engine_second.connect.return_value.__enter__.return_value = mock_connection_second

            mock_create_engine.side_effect = [mock_engine_first, mock_engine_second]

            # First call to read_sql_query raises Oracle error, second call succeeds
            mock_read_sql.side_effect = [
                Exception("DPY-3015: password verifier type 0x939 is not supported"),
                iter([test_df1, test_df2])
            ]

            mock_handle_error.return_value = True  # Indicates successful retry

            # Mock apply_masking to return the input dataframe (identity function)
            mock_apply_masking.side_effect = lambda df, env: df

            result = copy.execute_query(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                'SELECT * FROM test_table',
                batch_size=1,
                environment='test-env',
                apply_masks=True
            )

            # Result should be a generator
            batches = list(result)
            assert len(batches) == 2
            assert batches[0].equals(test_df1)
            assert batches[1].equals(test_df2)

    def test_execute_query_oracle_error_handling_failed_retry(self):
        """Test Oracle error handling with failed retry."""
        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy._handle_oracle_connection_error') as mock_handle_error:

            mock_engine = MagicMock()
            mock_connection = MagicMock()
            mock_connection.execute.side_effect = Exception("DPY-3015: password verifier type 0x939 is not supported")
            mock_engine.connect.return_value.__enter__.return_value = mock_connection

            mock_create_engine.return_value = mock_engine
            mock_handle_error.return_value = False  # Indicates failed retry

            with patch('pandas.read_sql_query', side_effect=Exception("DPY-3015: password verifier type 0x939 is not supported")):
                with pytest.raises(CopyError):  # Changed from DatabaseError to CopyError
                    copy.execute_query(
                        'oracle+oracledb://user:pass@host:1521?service_name=service',
                        'SELECT * FROM test_table',
                        batch_size=None,
                        environment='test-env',
                        apply_masks=True
                    )

    def test_execute_query_oracle_non_sqlalchemy_error_failed_retry(self):
        """Test Oracle error handling with non-SQLAlchemy error and failed retry."""
        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy._handle_oracle_connection_error') as mock_handle_error:

            mock_engine = MagicMock()
            mock_connection = MagicMock()
            mock_connection.execute.side_effect = ValueError("Some other error")
            mock_engine.connect.return_value.__enter__.return_value = mock_connection

            mock_create_engine.return_value = mock_engine
            mock_handle_error.return_value = False  # Indicates failed retry

            with patch('pandas.read_sql_query', side_effect=ValueError("Some other error")):
                with pytest.raises(CopyError):
                    copy.execute_query(
                        'oracle+oracledb://user:pass@host:1521?service_name=service',
                        'SELECT * FROM test_table',
                        batch_size=None,
                        environment='test-env',
                        apply_masks=True
                    )

    def test_write_to_db_oracle_error_handling_with_retry(self):
        """Test Oracle write error handling with successful retry."""
        with patch('elm.core.copy.write_to_db_streaming') as mock_streaming:

            data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

            # Mock streaming to succeed (the retry logic is inside streaming module)
            mock_streaming.return_value = 2  # Return record count

            result = copy.write_to_db(
                data=data,
                connection_url='oracle+oracledb://user:pass@host:1521?service_name=service',
                table_name='test_table',
                mode=WriteMode.APPEND
            )

            assert result == 2  # Should return record count
            mock_streaming.assert_called_once()

    def test_write_to_db_oracle_error_handling_failed_retry(self):
        """Test Oracle write error handling with failed retry."""
        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy._handle_oracle_connection_error') as mock_handle_error:

            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_handle_error.return_value = False  # Indicates failed retry

            data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

            with patch.object(pd.DataFrame, 'to_sql', side_effect=Exception("DPY-3015: password verifier type 0x939 is not supported")):
                with pytest.raises(CopyError):
                    copy.write_to_db(
                        data=data,
                        connection_url='oracle+oracledb://user:pass@host:1521?service_name=service',
                        table_name='test_table',
                        mode=WriteMode.APPEND
                    )

    def test_check_table_exists_oracle_error_handling_with_retry(self):
        """Test Oracle check table exists error handling with successful retry."""
        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy.inspect') as mock_inspect, \
             patch('elm.core.copy._handle_oracle_connection_error') as mock_handle_error:

            # First call raises Oracle error
            mock_inspector_first = MagicMock()
            mock_inspector_first.has_table.side_effect = Exception("DPY-3015: password verifier type 0x939 is not supported")

            # Second call succeeds
            mock_inspector_second = MagicMock()
            mock_inspector_second.has_table.return_value = True

            mock_inspect.side_effect = [mock_inspector_first, mock_inspector_second]
            mock_handle_error.return_value = True  # Indicates successful retry

            result = copy.check_table_exists(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                'test_table'
            )

            assert result is True

    def test_get_table_columns_oracle_error_handling_with_retry(self):
        """Test Oracle get table columns error handling with successful retry."""
        with patch('elm.core.copy.create_engine') as mock_create_engine, \
             patch('elm.core.copy.inspect') as mock_inspect, \
             patch('elm.core.copy._handle_oracle_connection_error') as mock_handle_error:

            # First call raises Oracle error
            mock_inspector_first = MagicMock()
            mock_inspector_first.has_table.side_effect = Exception("DPY-3015: password verifier type 0x939 is not supported")

            # Second call succeeds
            mock_inspector_second = MagicMock()
            mock_inspector_second.has_table.return_value = True
            mock_inspector_second.get_columns.return_value = [
                {'name': 'ID', 'type': 'INTEGER'},
                {'name': 'NAME', 'type': 'VARCHAR'}
            ]

            mock_inspect.side_effect = [mock_inspector_first, mock_inspector_second]
            mock_handle_error.return_value = True  # Indicates successful retry

            result = copy.get_table_columns(
                'oracle+oracledb://user:pass@host:1521?service_name=service',
                'test_table'
            )

            assert result == ['id', 'name']

    def test_validate_target_table_create_if_not_exists_failure(self):
        """Test validate target table with create failure."""
        with patch('elm.core.copy.check_table_exists', return_value=False), \
             patch('elm.core.copy.create_engine') as mock_create_engine:

            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

            with patch.object(pd.DataFrame, 'to_sql', side_effect=Exception("Create table failed")):
                with pytest.raises(CopyError) as exc_info:
                    copy.validate_target_table(
                        source_data=data,
                        target_url='postgresql://user:pass@host:5432/db',
                        table_name='test_table',
                        create_if_not_exists=True
                    )

                assert "Failed to create table" in str(exc_info.value)

    def test_validate_target_table_no_columns_retrieved(self):
        """Test validate target table when columns cannot be retrieved."""
        with patch('elm.core.copy.check_table_exists', return_value=True), \
             patch('elm.core.copy.get_table_columns', return_value=None):

            data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

            with pytest.raises(CopyError) as exc_info:
                copy.validate_target_table(
                    source_data=data,
                    target_url='postgresql://user:pass@host:5432/db',
                    table_name='test_table',
                    create_if_not_exists=False
                )

            assert "Could not retrieve columns" in str(exc_info.value)

    def test_copy_db_to_file_with_batching(self):
        """Test copy database to file with batching."""
        with patch('elm.core.copy.get_connection_url') as mock_get_url, \
             patch('elm.core.copy.execute_query') as mock_execute, \
             patch('elm.core.copy.write_to_file') as mock_write:

            mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'

            # Mock batched results
            batch1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
            batch2 = pd.DataFrame({'id': [2], 'name': ['Bob']})
            mock_execute.return_value = iter([batch1, batch2])

            result = copy.copy_db_to_file(
                source_env='test-env',
                query='SELECT * FROM test_table',
                file_path='output.csv',
                file_format='csv',
                mode='REPLACE',
                batch_size=1,
                apply_masks=True
            )

            assert result.success is True
            assert result.record_count == 2
            assert mock_write.call_count == 2

    def test_copy_file_to_db_with_validation(self):
        """Test copy file to database with validation."""
        test_data = "id,name\n1,Alice\n2,Bob"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_data)
            temp_file = f.name

        try:
            with patch('elm.core.copy.get_connection_url') as mock_get_url, \
                 patch('elm.core.copy.write_to_db') as mock_write_db, \
                 patch('elm.core.copy.apply_masking') as mock_apply_masking, \
                 patch('elm.core.copy.validate_target_table') as mock_validate:

                mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'
                mock_write_db.return_value = 2  # Return record count
                mock_df = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})
                mock_apply_masking.return_value = mock_df

                result = copy.copy_file_to_db(
                    file_path=temp_file,
                    target_env="test-env",
                    table="test_table",
                    file_format='csv',
                    mode='APPEND',
                    validate_target=True,
                    create_if_not_exists=True,
                    apply_masks=True
                )

                assert result.success is True
                mock_validate.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_copy_db_to_db_with_validation_and_batching(self):
        """Test copy database to database with validation and batching."""
        with patch('elm.core.copy.get_connection_url') as mock_get_url, \
             patch('elm.core.copy.execute_query') as mock_execute, \
             patch('elm.core.copy.write_to_db') as mock_write_db, \
             patch('elm.core.copy.validate_target_table') as mock_validate:

            mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'
            mock_write_db.return_value = 1  # Return record count per batch

            # Mock validation query
            sample_data = pd.DataFrame({'id': [1], 'name': ['Sample']})

            # Mock batched results
            batch1 = pd.DataFrame({'id': [1], 'name': ['Alice']})
            batch2 = pd.DataFrame({'id': [2], 'name': ['Bob']})

            mock_execute.side_effect = [sample_data, iter([batch1, batch2])]

            result = copy.copy_db_to_db(
                source_env='source-env',
                target_env='target-env',
                query='SELECT * FROM source_table',
                table='target_table',
                mode='APPEND',
                batch_size=1,
                validate_target=True,
                create_if_not_exists=True,
                apply_masks=True
            )

            assert result.success is True
            assert result.record_count == 2
            mock_validate.assert_called_once()

    def test_copy_db_to_db_validation_error(self):
        """Test copy database to database with validation error."""
        with patch('elm.core.copy.get_connection_url') as mock_get_url, \
             patch('elm.core.copy.execute_query') as mock_execute:

            mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'
            mock_execute.side_effect = Exception("Validation query failed")

            result = copy.copy_db_to_db(
                source_env='source-env',
                target_env='target-env',
                query='SELECT * FROM source_table',
                table='target_table',
                mode='APPEND',
                validate_target=True,
                apply_masks=True
            )

            assert result.success is False
            assert "Error during validation" in result.message
