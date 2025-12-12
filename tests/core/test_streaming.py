"""
Tests for streaming data copy operations
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from elm.core.streaming import (
    detect_database_type,
    write_to_db_streaming
)
from elm.core.types import WriteMode
from elm.core.exceptions import DatabaseError, CopyError


class TestDatabaseTypeDetection:
    """Test database type detection from connection URLs"""
    
    def test_detect_postgresql(self):
        """Test PostgreSQL detection"""
        assert detect_database_type("postgresql://user:pass@host:5432/db") == "postgresql"
        assert detect_database_type("postgres://user:pass@host:5432/db") == "postgresql"
    
    def test_detect_oracle(self):
        """Test Oracle detection"""
        assert detect_database_type("oracle+oracledb://user:pass@host:1521/db") == "oracle"
        assert detect_database_type("oracle://user:pass@host:1521/db") == "oracle"
    
    def test_detect_mysql(self):
        """Test MySQL detection"""
        assert detect_database_type("mysql+pymysql://user:pass@host:3306/db") == "mysql"
        assert detect_database_type("mysql://user:pass@host:3306/db") == "mysql"
    
    def test_detect_mssql(self):
        """Test SQL Server detection"""
        assert detect_database_type("mssql+pyodbc://user:pass@host:1433/db") == "mssql"
        assert detect_database_type("sqlserver://user:pass@host:1433/db") == "mssql"
    
    def test_detect_unknown(self):
        """Test unknown database type"""
        assert detect_database_type("unknown://user:pass@host:1234/db") == "unknown"


class TestStreamingWrite:
    """Test streaming write operations"""
    
    def test_write_to_db_streaming_fallback(self):
        """Test fallback to pandas when optimized method not available"""
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        with patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:
            mock_fallback.return_value = 3
            
            result = write_to_db_streaming(
                data=data,
                connection_url="unknown://user:pass@host/db",
                table_name="test_table",
                mode=WriteMode.APPEND,
                use_optimized=True
            )
            
            assert result == 3
            mock_fallback.assert_called_once()
    
    def test_write_to_db_streaming_no_optimization(self):
        """Test streaming with optimization disabled"""
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        with patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:
            mock_fallback.return_value = 3
            
            result = write_to_db_streaming(
                data=data,
                connection_url="postgresql://user:pass@host/db",
                table_name="test_table",
                mode=WriteMode.APPEND,
                use_optimized=False
            )
            
            assert result == 3
            mock_fallback.assert_called_once()


class TestStreamingFallback:
    """Test streaming fallback behavior"""

    def test_fallback_on_optimized_failure(self):
        """Test that streaming falls back to pandas when optimized method fails"""
        data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})

        with patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:
            mock_fallback.return_value = 3

            # Force an error in optimized path by using unknown DB type
            result = write_to_db_streaming(
                data=data,
                connection_url="unknown://user:pass@host/db",
                table_name="test_table",
                mode=WriteMode.APPEND,
                use_optimized=True
            )

            assert result == 3
            mock_fallback.assert_called_once()


class TestPostgreSQLStreaming:
    """Test PostgreSQL-specific streaming operations"""

    def test_postgresql_copy_import_error(self):
        """Test PostgreSQL COPY when psycopg2 not installed"""
        from elm.core.streaming import write_postgresql_copy

        data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

        # Mock the import to raise ImportError
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'psycopg2':
                raise ImportError("No module named 'psycopg2'")
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(CopyError, match="psycopg2 not installed"):
                write_postgresql_copy(
                    data=data,
                    connection_url="postgresql://user:pass@localhost:5432/db",
                    table_name="test_table",
                    mode=WriteMode.APPEND
                )

    def test_postgresql_streaming_integration(self):
        """Test PostgreSQL streaming with fallback"""
        data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

        # Test that it falls back to pandas when psycopg2 not available
        with patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:
            mock_fallback.return_value = 3

            result = write_to_db_streaming(
                data=data,
                connection_url="postgresql://user:pass@localhost:5432/db",
                table_name="test_table",
                mode=WriteMode.APPEND,
                use_optimized=True
            )

            # Should fall back to pandas if psycopg2 not available
            assert result == 3


class TestOracleStreaming:
    """Test Oracle-specific streaming operations"""

    def test_oracle_import_error(self):
        """Test Oracle executemany when oracledb not installed"""
        from elm.core.streaming import write_oracle_executemany

        data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

        # Mock the import to raise ImportError
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'oracledb':
                raise ImportError("No module named 'oracledb'")
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(CopyError, match="oracledb not installed"):
                write_oracle_executemany(
                    data=data,
                    connection_url="oracle+oracledb://user:pass@localhost:1521/db",
                    table_name="test_table",
                    mode=WriteMode.APPEND
                )

    def test_oracle_streaming_integration(self):
        """Test Oracle streaming with fallback"""
        data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

        # Test that it falls back to pandas when oracledb not available
        with patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:
            mock_fallback.return_value = 3

            result = write_to_db_streaming(
                data=data,
                connection_url="oracle+oracledb://user:pass@localhost:1521/db",
                table_name="test_table",
                mode=WriteMode.APPEND,
                use_optimized=True
            )

            # Should fall back to pandas if oracledb not available
            assert result == 3

    def test_oracle_nan_nat_converted_to_none(self):
        """Ensure NaN/NaT values are sent to Oracle as None (NULL)."""
        from elm.core.streaming import write_oracle_executemany
        from elm.core.types import WriteMode

        data = pd.DataFrame({
            'id': [1, 2],
            'created_date': [pd.Timestamp('2025-09-11 15:37:51'), pd.NaT],
            'row_count': [123, float('nan')],
        })

        captured = {}

        class FakeCursor:
            def __init__(self):
                self.executemany_args = None

            def execute(self, sql):
                # TRUNCATE not used in APPEND mode, but keep for completeness
                pass

            def executemany(self, sql, seq, batcherrors=False):
                # Materialize the sequence so we can inspect it later
                self.executemany_args = (sql, list(seq), batcherrors)

            def getbatcherrors(self):
                # Simulate no batch errors
                return []

            def close(self):
                pass

        class FakeConnection:
            def __init__(self):
                self._cursor = FakeCursor()

            def cursor(self):
                captured['cursor'] = self._cursor
                return self._cursor

            def commit(self):
                pass

            def close(self):
                pass

        def fake_connect(user, password, dsn):
            return FakeConnection()

        def fake_makedsn(host, port, service_name=None, sid=None):
            return "DSN"

        import types
        fake_oracledb = types.SimpleNamespace(
            connect=fake_connect,
            makedsn=fake_makedsn,
        )

        # Patch builtins.__import__ so that "import oracledb" inside the
        # streaming module returns our fake module.
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'oracledb':
                return fake_oracledb
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            written = write_oracle_executemany(
                data=data,
                connection_url="oracle+oracledb://user:pass@localhost:1521/db",
                table_name="test_table",
                mode=WriteMode.APPEND,
            )

        # Should report the correct number of written rows
        assert written == len(data)

        # Inspect the values passed to executemany
        sql, seq, batcherrors = captured['cursor'].executemany_args
        # Second row should have None for both the datetime and numeric fields
        second_row = seq[1]
        assert second_row[1] is None  # created_date (NaT) -> None
        assert second_row[2] is None  # row_count (NaN) -> None


    def test_oracle_thick_mode_fallback_uses_executemany(self):
        """When Oracle thick mode is activated, fallback should use executemany.

        This ensures that the oracle init client path uses the same high-
        performance executemany implementation instead of pandas.to_sql.
        """
        from elm.core.streaming import _write_pandas_fallback

        data = pd.DataFrame({'id': [1, 2], 'name': ['Alice', 'Bob']})

        oracle_url = "oracle+oracledb://user:pass@localhost:1521/db"

        # First attempt should go through pandas.to_sql and fail with an
        # Oracle-thin error, triggering _handle_oracle_connection_error.
        # After thick mode activation succeeds (mocked), we expect
        # write_oracle_executemany to be called for the retry.
        with patch('elm.core.streaming.create_engine') as mock_engine, \
             patch.object(pd.DataFrame, 'to_sql', side_effect=Exception("DPY-3015: password verifier type 0x939")) as mock_to_sql, \
             patch('elm.core.streaming._handle_oracle_connection_error') as mock_handle, \
             patch('elm.core.streaming.write_oracle_executemany') as mock_exec:

            mock_engine.return_value = MagicMock()
            mock_handle.return_value = True
            mock_exec.return_value = len(data)

            written = _write_pandas_fallback(
                data=data,
                connection_url=oracle_url,
                table_name="test_table",
                mode=WriteMode.APPEND,
                batch_size=None,
            )

        # pandas.to_sql should have been attempted once and failed
        assert mock_to_sql.call_count >= 1

        # After successful thick-mode handling, executemany-based writer must be used
        mock_exec.assert_called_once_with(data, oracle_url, "test_table", WriteMode.APPEND)
        assert written == len(data)


class TestMySQLStreaming:
    """Test MySQL-specific streaming operations"""

    def test_mysql_import_error(self):
        """Test MySQL executemany when pymysql not installed"""
        from elm.core.streaming import write_mysql_executemany

        data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

        # Mock the import to raise ImportError
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'pymysql':
                raise ImportError("No module named 'pymysql'")
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(CopyError, match="pymysql not installed"):
                write_mysql_executemany(
                    data=data,
                    connection_url="mysql+pymysql://user:pass@localhost:3306/db",
                    table_name="test_table",
                    mode=WriteMode.APPEND
                )


class TestMSSQLStreaming:
    """Test SQL Server-specific streaming operations"""

    def test_mssql_import_error(self):
        """Test SQL Server fast_executemany when pyodbc not installed"""
        from elm.core.streaming import write_mssql_fast_executemany

        data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

        # Mock the import to raise ImportError
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'pyodbc':
                raise ImportError("No module named 'pyodbc'")
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(CopyError, match="pyodbc not installed"):
                write_mssql_fast_executemany(
                    data=data,
                    connection_url="mssql+pyodbc://user:pass@localhost:1433/db",
                    table_name="test_table",
                    mode=WriteMode.APPEND
                )


class TestStreamingErrorHandling:
    """Test error handling in streaming operations"""

    def test_write_to_db_streaming_with_error(self):
        """Test streaming error handling"""
        data = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})

        with patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:
            mock_fallback.side_effect = DatabaseError("Connection failed")

            with pytest.raises(DatabaseError, match="Connection failed"):
                write_to_db_streaming(
                    data=data,
                    connection_url="postgresql://user:pass@localhost:5432/db",
                    table_name="test_table",
                    mode=WriteMode.APPEND,
                    use_optimized=False
                )

    def test_write_to_db_streaming_batch_size(self):
        """Test streaming with custom batch size"""
        data = pd.DataFrame({'id': list(range(100)), 'name': [f'User{i}' for i in range(100)]})

        with patch('elm.core.streaming._write_pandas_fallback') as mock_fallback:
            mock_fallback.return_value = 100

            result = write_to_db_streaming(
                data=data,
                connection_url="postgresql://user:pass@localhost:5432/db",
                table_name="test_table",
                mode=WriteMode.APPEND,
                batch_size=10,
                use_optimized=False
            )

            assert result == 100

