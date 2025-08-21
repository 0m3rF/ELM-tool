"""Tests for core copy functionality."""

import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os

from elm.core import copy
from elm.core.types import FileFormat


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
