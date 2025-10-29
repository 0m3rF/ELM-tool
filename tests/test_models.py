"""
Tests for the models module (elm/models.py).
"""
import pytest
from elm.models import ContainerState, DatabaseConfig, ContainerResult


class TestContainerState:
    """Test ContainerState enum."""

    def test_container_state_values(self):
        """Test that ContainerState has expected values."""
        assert ContainerState.NOT_EXISTS.value == "not_exists"
        assert ContainerState.STOPPED.value == "stopped"
        assert ContainerState.RUNNING.value == "running"
        assert ContainerState.HEALTHY.value == "healthy"
        assert ContainerState.UNHEALTHY.value == "unhealthy"

    def test_container_state_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {
            'NOT_EXISTS', 'STOPPED', 'RUNNING', 'HEALTHY', 'UNHEALTHY'
        }
        actual_members = set(ContainerState.__members__.keys())
        assert actual_members == expected_members

    def test_container_state_comparison(self):
        """Test ContainerState comparison."""
        assert ContainerState.RUNNING == ContainerState.RUNNING
        assert ContainerState.RUNNING != ContainerState.STOPPED


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_database_config_creation(self):
        """Test creating a DatabaseConfig instance."""
        config = DatabaseConfig(
            name="test-db",
            image="postgres:13",
            target_port=5432,
            env_vars={"POSTGRES_DB": "testdb"},
            health_check_cmd=["pg_isready"],
            default_port=5432,
            wait_time=30,
            startup_priority=1,
            depends_on=["redis"]
        )

        assert config.name == "test-db"
        assert config.image == "postgres:13"
        assert config.target_port == 5432
        assert config.default_port == 5432
        assert config.env_vars == {"POSTGRES_DB": "testdb"}
        assert config.health_check_cmd == ["pg_isready"]
        assert config.wait_time == 30
        assert config.startup_priority == 1
        assert config.depends_on == ["redis"]

    def test_database_config_defaults(self):
        """Test DatabaseConfig with default values."""
        config = DatabaseConfig(
            name="test-db",
            image="postgres:13",
            target_port=5432,
            env_vars={"POSTGRES_DB": "testdb"},
            health_check_cmd=["pg_isready"]
        )

        # Test default values
        assert config.wait_time == 30
        assert config.startup_priority == 0
        assert config.depends_on == []
        assert config.default_port == 5432  # Should default to target_port

    def test_database_config_required_fields(self):
        """Test that required fields are enforced."""
        # This should work
        config = DatabaseConfig(
            name="test-db",
            image="postgres:13",
            target_port=5432,
            env_vars={},
            health_check_cmd=[]
        )
        assert config.name == "test-db"

    def test_database_config_field_types(self):
        """Test that field types are correct."""
        config = DatabaseConfig(
            name="test-db",
            image="postgres:13",
            target_port=5432,
            env_vars={"KEY": "value"},
            health_check_cmd=["cmd"],
            default_port=5432,
            wait_time=60,
            startup_priority=2,
            depends_on=["dep1", "dep2"]
        )

        assert isinstance(config.name, str)
        assert isinstance(config.image, str)
        assert isinstance(config.target_port, int)
        assert isinstance(config.default_port, int)
        assert isinstance(config.env_vars, dict)
        assert isinstance(config.health_check_cmd, list)
        assert isinstance(config.wait_time, int)
        assert isinstance(config.startup_priority, int)
        assert isinstance(config.depends_on, list)

    def test_database_config_port_mapping(self):
        """Test that default_port defaults to target_port when not specified."""
        # Test with explicit default_port
        config1 = DatabaseConfig(
            name="test-db",
            image="postgres:13",
            target_port=5433,
            env_vars={},
            health_check_cmd=[],
            default_port=5432
        )
        assert config1.target_port == 5433
        assert config1.default_port == 5432

        # Test with default_port defaulting to target_port
        config2 = DatabaseConfig(
            name="test-db",
            image="postgres:13",
            target_port=5433,
            env_vars={},
            health_check_cmd=[]
        )
        assert config2.target_port == 5433
        assert config2.default_port == 5433  # Should default to target_port


class TestContainerResult:
    """Test ContainerResult dataclass."""

    def test_container_result_creation(self):
        """Test creating a ContainerResult instance."""
        result = ContainerResult(
            name="test-container",
            success=True,
            state=ContainerState.RUNNING,
            message="Container started successfully",
            startup_time=5.5
        )

        assert result.name == "test-container"
        assert result.success is True
        assert result.state == ContainerState.RUNNING
        assert result.message == "Container started successfully"
        assert result.startup_time == 5.5

    def test_container_result_defaults(self):
        """Test ContainerResult with default values."""
        result = ContainerResult(
            name="test-container",
            success=False,
            state=ContainerState.STOPPED,
            message="Container failed to start"
        )

        # Test default value
        assert result.startup_time == 0.0

    def test_container_result_with_different_states(self):
        """Test ContainerResult with different container states."""
        states_to_test = [
            ContainerState.NOT_EXISTS,
            ContainerState.STOPPED,
            ContainerState.RUNNING,
            ContainerState.HEALTHY,
            ContainerState.UNHEALTHY
        ]

        for state in states_to_test:
            result = ContainerResult(
                name="test-container",
                success=True,
                state=state,
                message=f"Container is {state.value}"
            )
            assert result.state == state
            assert result.message == f"Container is {state.value}"

    def test_container_result_field_types(self):
        """Test that field types are correct."""
        result = ContainerResult(
            name="test-container",
            success=True,
            state=ContainerState.HEALTHY,
            message="All good",
            startup_time=10.25
        )

        assert isinstance(result.name, str)
        assert isinstance(result.success, bool)
        assert isinstance(result.state, ContainerState)
        assert isinstance(result.message, str)
        assert isinstance(result.startup_time, float)


class TestModelsIntegration:
    """Test integration between model classes."""

    def test_database_config_with_container_result(self):
        """Test using DatabaseConfig and ContainerResult together."""
        config = DatabaseConfig(
            name="postgres-db",
            image="postgres:13",
            target_port=5432,
            env_vars={"POSTGRES_DB": "testdb"},
            health_check_cmd=["pg_isready"],
            wait_time=45,
            startup_priority=1
        )

        # Simulate a successful container start
        result = ContainerResult(
            name=config.name,
            success=True,
            state=ContainerState.HEALTHY,
            message=f"Container {config.name} started successfully",
            startup_time=config.wait_time - 15.0
        )

        assert result.name == config.name
        assert result.success is True
        assert result.state == ContainerState.HEALTHY
        assert result.startup_time < config.wait_time

    def test_models_import(self):
        """Test that all models can be imported."""
        from elm.models import ContainerState, DatabaseConfig, ContainerResult

        # Test that classes exist and are callable
        assert ContainerState is not None
        assert DatabaseConfig is not None
        assert ContainerResult is not None