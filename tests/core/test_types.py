"""
Tests for the core types module.

These tests focus on testing the data types and models used
throughout the core modules.
"""
import pytest
import pandas as pd
from enum import Enum

from elm.core.types import (
    OperationResult,
    EnvironmentConfig,
    MaskingConfig,
    MaskingAlgorithm,
    DatabaseType,
    FileFormat,
    WriteMode
)


class TestOperationResult:
    """Test OperationResult data class."""

    def test_operation_result_success(self):
        """Test creating a successful OperationResult."""
        result = OperationResult(
            success=True,
            message="Operation completed successfully",
            data={"key": "value"},
            record_count=10
        )

        assert result.success is True
        assert result.message == "Operation completed successfully"
        assert result.data == {"key": "value"}
        assert result.record_count == 10

    def test_operation_result_failure(self):
        """Test creating a failed OperationResult."""
        result = OperationResult(
            success=False,
            message="Operation failed",
            data=None,
            record_count=0
        )

        assert result.success is False
        assert result.message == "Operation failed"
        assert result.data is None
        assert result.record_count == 0

    def test_operation_result_defaults(self):
        """Test OperationResult with default values."""
        result = OperationResult(success=True, message="Success")

        assert result.success is True
        assert result.message == "Success"
        assert result.data is None
        assert result.record_count is None

    def test_operation_result_with_dataframe(self):
        """Test OperationResult with DataFrame data."""
        df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        result = OperationResult(
            success=True,
            message="Data retrieved",
            data=df,
            record_count=len(df)
        )

        assert result.success is True
        assert isinstance(result.data, pd.DataFrame)
        assert result.record_count == 3

    def test_operation_result_mutable(self):
        """Test that OperationResult is mutable (not frozen dataclass)."""
        result = OperationResult(success=True, message="Test")

        # Should be able to modify the result
        result.success = False
        assert result.success is False

    def test_operation_result_equality(self):
        """Test OperationResult equality comparison."""
        result1 = OperationResult(success=True, message="Test", data={"a": 1})
        result2 = OperationResult(success=True, message="Test", data={"a": 1})
        result3 = OperationResult(success=False, message="Test", data={"a": 1})

        assert result1 == result2
        assert result1 != result3

    def test_operation_result_repr(self):
        """Test OperationResult string representation."""
        result = OperationResult(success=True, message="Test")
        repr_str = repr(result)
        
        assert "OperationResult" in repr_str
        assert "success=True" in repr_str
        assert "message='Test'" in repr_str


class TestEnvironmentConfig:
    """Test EnvironmentConfig data class."""

    def test_environment_config_creation(self):
        """Test creating an EnvironmentConfig."""
        config = EnvironmentConfig(
            name="test-env",
            host="localhost",
            port=5432,
            user="postgres",
            password="secret",
            service="mydb",
            db_type=DatabaseType.POSTGRES,
            is_encrypted=False,
            encryption_key=None
        )

        assert config.name == "test-env"
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "postgres"
        assert config.password == "secret"
        assert config.service == "mydb"
        assert config.db_type == DatabaseType.POSTGRES
        assert config.is_encrypted is False
        assert config.encryption_key is None

    def test_environment_config_with_encryption(self):
        """Test creating an encrypted EnvironmentConfig."""
        config = EnvironmentConfig(
            name="secure-env",
            host="localhost",
            port=5432,
            user="postgres",
            password="encrypted_password",
            service="mydb",
            db_type=DatabaseType.POSTGRES,
            is_encrypted=True,
            encryption_key="some_key"
        )

        assert config.is_encrypted is True
        assert config.encryption_key == "some_key"

    def test_environment_config_defaults(self):
        """Test EnvironmentConfig with default values."""
        config = EnvironmentConfig(
            name="test-env",
            host="localhost",
            port=5432,
            user="postgres",
            password="secret",
            service="mydb",
            db_type=DatabaseType.POSTGRES
        )

        assert config.is_encrypted is False
        assert config.encryption_key is None


class TestMaskingConfig:
    """Test MaskingConfig data class."""

    def test_masking_config_creation(self):
        """Test creating a MaskingConfig."""
        config = MaskingConfig(
            column="password",
            algorithm=MaskingAlgorithm.STAR,
            params={"length": 5},
            environment="prod"
        )

        assert config.column == "password"
        assert config.algorithm == MaskingAlgorithm.STAR
        assert config.params == {"length": 5}
        assert config.environment == "prod"

    def test_masking_config_global(self):
        """Test creating a global MaskingConfig."""
        config = MaskingConfig(
            column="email",
            algorithm=MaskingAlgorithm.STAR_LENGTH,
            params={"length": 4}
        )

        assert config.column == "email"
        assert config.algorithm == MaskingAlgorithm.STAR_LENGTH
        assert config.params == {"length": 4}
        assert config.environment is None

    def test_masking_config_defaults(self):
        """Test MaskingConfig with default values."""
        config = MaskingConfig(
            column="ssn",
            algorithm=MaskingAlgorithm.NULLIFY
        )

        assert config.column == "ssn"
        assert config.algorithm == MaskingAlgorithm.NULLIFY
        assert config.params is None
        assert config.environment is None


# CopyConfig tests removed - type not implemented yet


class TestEnums:
    """Test enum types."""

    def test_masking_algorithm_enum(self):
        """Test MaskingAlgorithm enum."""
        assert MaskingAlgorithm.STAR.value == "star"
        assert MaskingAlgorithm.STAR_LENGTH.value == "star_length"
        assert MaskingAlgorithm.RANDOM.value == "random"
        assert MaskingAlgorithm.NULLIFY.value == "nullify"

        # Test enum membership
        assert "star" in [alg.value for alg in MaskingAlgorithm]
        assert "invalid" not in [alg.value for alg in MaskingAlgorithm]

    def test_database_type_enum(self):
        """Test DatabaseType enum."""
        assert DatabaseType.POSTGRES.value == "POSTGRES"
        assert DatabaseType.MYSQL.value == "MYSQL"
        assert DatabaseType.ORACLE.value == "ORACLE"
        assert DatabaseType.MSSQL.value == "MSSQL"

        # Test enum membership
        assert "POSTGRES" in [db.value for db in DatabaseType]
        assert "invalid" not in [db.value for db in DatabaseType]

    def test_file_format_enum(self):
        """Test FileFormat enum."""
        assert FileFormat.CSV.value == "csv"
        assert FileFormat.JSON.value == "json"

        # Test enum membership
        assert "csv" in [fmt.value for fmt in FileFormat]
        assert "invalid" not in [fmt.value for fmt in FileFormat]

    def test_write_mode_enum(self):
        """Test WriteMode enum."""
        assert WriteMode.APPEND.value == "APPEND"
        assert WriteMode.REPLACE.value == "REPLACE"
        assert WriteMode.FAIL.value == "FAIL"

        # Test enum membership
        assert "APPEND" in [mode.value for mode in WriteMode]
        assert "INVALID" not in [mode.value for mode in WriteMode]

    def test_enum_from_string(self):
        """Test creating enums from string values."""
        # Test MaskingAlgorithm
        assert MaskingAlgorithm("star") == MaskingAlgorithm.STAR
        assert MaskingAlgorithm("star_length") == MaskingAlgorithm.STAR_LENGTH

        # Test DatabaseType
        assert DatabaseType("POSTGRES") == DatabaseType.POSTGRES
        assert DatabaseType("MYSQL") == DatabaseType.MYSQL

        # Test FileFormat
        assert FileFormat("csv") == FileFormat.CSV
        assert FileFormat("json") == FileFormat.JSON

        # Test WriteMode
        assert WriteMode("APPEND") == WriteMode.APPEND
        assert WriteMode("REPLACE") == WriteMode.REPLACE

    def test_enum_invalid_value(self):
        """Test creating enums with invalid values."""
        with pytest.raises(ValueError):
            MaskingAlgorithm("invalid")

        with pytest.raises(ValueError):
            DatabaseType("invalid")

        with pytest.raises(ValueError):
            FileFormat("invalid")

        with pytest.raises(ValueError):
            WriteMode("INVALID")


class TestTypeValidation:
    """Test type validation and conversion utilities."""

    def test_operation_result_type_hints(self):
        """Test that OperationResult accepts correct types."""
        # Test with various data types
        result1 = OperationResult(success=True, message="Test", data=None)
        result2 = OperationResult(success=True, message="Test", data={})
        result3 = OperationResult(success=True, message="Test", data=[])
        result4 = OperationResult(success=True, message="Test", data=pd.DataFrame())

        assert all(isinstance(r, OperationResult) for r in [result1, result2, result3, result4])

    def test_config_type_validation(self):
        """Test that config classes validate types correctly."""
        # This would be more relevant if we had runtime type checking
        # For now, just test that the classes can be instantiated with correct types
        
        env_config = EnvironmentConfig(
            name="test",
            host="localhost",
            port=5432,
            user="user",
            password="pass",
            service="db",
            db_type=DatabaseType.POSTGRES
        )
        
        mask_config = MaskingConfig(
            column="col",
            algorithm=MaskingAlgorithm.STAR
        )
        
        assert isinstance(env_config, EnvironmentConfig)
        assert isinstance(mask_config, MaskingConfig)
