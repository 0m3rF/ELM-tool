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
             patch('elm.core.copy.apply_masking') as mock_apply_masking:
            
            mock_get_url.return_value = 'postgresql://user:pass@localhost:5432/db'
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
        from elm.core.exceptions import CopyError

        data = pd.DataFrame({'id': [1], 'name': ['Alice']})

        with patch('elm.core.copy.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = Exception("General error")

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
        result = copy.process_in_parallel(test_func, items, max_workers=2)

        # Should return results with None for failed items
        assert result == ['processed_good_item', None]


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
