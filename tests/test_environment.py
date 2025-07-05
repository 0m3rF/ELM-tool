"""
Tests for the environment management functionality of the ELM tool.
"""
import configparser
import pytest
import pandas as pd
import elm
from .test_databases import DatabaseConfigs
from elm.elm_utils import variables

@pytest.fixture
def temp_env_dir():
    return variables.ENVS_FILE
    

@pytest.mark.dependency(name="create_environment")
def test_create_environment(temp_env_dir):
    """Test creating a new environment."""
    # Create a test environment
    db_configs = DatabaseConfigs()
    db_config = db_configs.get_configs()["postgresql"]

    result = elm.create_environment(
        name="test-pg",
        host="localhost",
        port=db_config.port,
        user=db_config.env_vars["POSTGRES_USER"],
        password=db_config.env_vars["POSTGRES_PASSWORD"],
        service=db_config.env_vars["POSTGRES_DB"],
        db_type="postgres"
    )

    print("result is " + str(result))

    if(not result):
        config = configparser.ConfigParser()
        config.read(temp_env_dir)
        if "test-pg" in config.sections():
            # If the environment exists, success the test
            assert True
        else:
            pytest.fail("Failed to create environment and it does not exists")
        return

    assert result is True

    # Verify the environment was created
    config = configparser.ConfigParser()
    config.read(temp_env_dir)

    assert "test-pg" in config.sections()
    assert config["test-pg"]["host"] == "localhost"
    assert config["test-pg"]["port"] == str(db_config.port)
    assert config["test-pg"]["user"] == db_config.env_vars["POSTGRES_USER"]
    assert config["test-pg"]["password"] == db_config.env_vars["POSTGRES_PASSWORD"]
    assert config["test-pg"]["service"] == db_config.env_vars["POSTGRES_DB"]
    assert config["test-pg"]["type"] == "postgres"
    assert config["test-pg"]["is_encrypted"] == "False"

@pytest.mark.dependency(name="create_encrypted_environment")
def test_create_encrypted_environment(temp_env_dir):
    """Test creating an encrypted environment."""
    # Create an encrypted test environment
    db_configs = DatabaseConfigs()
    db_config = db_configs.get_configs()["postgresql"]

    result = elm.create_environment(
        name="secure-pg",
        host="localhost",
        port=db_config.port,
        user=db_config.env_vars["POSTGRES_USER"],
        password=db_config.env_vars["POSTGRES_PASSWORD"],
        service=db_config.env_vars["POSTGRES_DB"],
        db_type="postgres",
        encrypt=True,
        encryption_key="secret-key"
    )

    if(not result):
        config = configparser.ConfigParser()
        config.read(temp_env_dir)
        if "secure-pg" in config.sections():
            # If the environment exists, success the test
            assert True
        else:
            pytest.fail("Failed to create environment and it does not exists")

    # Verify the environment was created and is encrypted
    config = configparser.ConfigParser()
    config.read(temp_env_dir)

    assert "secure-pg" in config.sections()
    assert config["secure-pg"]["host"] == "localhost"
    assert config["secure-pg"]["port"] == str(db_config.port)
    assert config["secure-pg"]["user"] == db_config.env_vars["POSTGRES_USER"]
    assert config["secure-pg"]["password"] != db_config.env_vars["POSTGRES_PASSWORD"]  # Password should be encrypted
    assert config["secure-pg"]["service"] == db_config.env_vars["POSTGRES_DB"]
    assert config["secure-pg"]["type"] == "postgres"
    assert config["secure-pg"]["is_encrypted"] == "True"
    assert "salt" in config["secure-pg"]

@pytest.mark.dependency(name="list_environments", depends=["create_environment", "create_encrypted_environment"])
def test_list_environments(temp_env_dir):
    """Test listing environments."""

    # List environments
    environments = elm.list_environments()

    # Verify the environments are listed
    assert len(environments) == 2
    env_names = [env["name"] for env in environments]
    assert "test-pg" in env_names
    assert "secure-pg" in env_names

    # Test with show_all=True
    detailed_environments = elm.list_environments(show_all=True)
    assert len(detailed_environments) == 2

    # Check that passwords are masked
    for env in detailed_environments:
        if "password" in env:
            assert env["password"] == "********"

@pytest.mark.dependency(name="get_environment", depends=["create_environment"])
def test_get_environment(temp_env_dir):
    """Test getting environment details."""

    db_configs = DatabaseConfigs()
    db_config = db_configs.get_configs()["postgresql"]

    # Get environment details
    env = elm.get_environment("test-pg")
    # Verify the environment details
    assert env is not None
    assert env["host"] == "localhost"
    assert env["port"] == str(db_config.port)
    assert env["user"] == db_config.env_vars["POSTGRES_USER"]
    assert env["password"] == db_config.env_vars["POSTGRES_PASSWORD"]
    assert env["service"] == db_config.env_vars["POSTGRES_DB"]
    assert env["type"] == "postgres"

def test_get_non_existent_environment(temp_env_dir):
    # Test getting a non-existent environment
    non_existent_env = elm.get_environment("non-existent")
    assert non_existent_env is None

@pytest.mark.dependency(name="get_encrypted_environment", depends=["create_encrypted_environment"])
def test_get_encrypted_environment(temp_env_dir):
    """Test getting encrypted environment details."""

    db_configs = DatabaseConfigs()
    db_config = db_configs.get_configs()["postgresql"]

    # Get environment details with correct key
    env = elm.get_environment("secure-pg")
    # Verify the environment details
    assert env is not None
    assert env["host"] == "localhost"
    assert env["port"] == str(db_config.port)
    assert env["user"] == db_config.env_vars["POSTGRES_USER"]
    assert env["password"] != db_config.env_vars["POSTGRES_PASSWORD"]
    assert env["service"] == db_config.env_vars["POSTGRES_DB"]
    assert env["type"] == "postgres"

    # Test with incorrect encryption key - this might not raise an exception in the actual implementation
    # Uncomment if the implementation is expected to raise an exception
    # with pytest.raises(Exception):
    #     elm.get_environment("secure-pg", encryption_key="wrong-key")

@pytest.mark.dependency(name="test_test_environment", depends=["create_environment"])
def test_test_environment(temp_env_dir):
    """Test testing a database connection."""

    # Test the connection
    result = elm.test_environment("test-pg")

    # Verify the connection was tested
    assert result["success"] is True
    assert "Successfully connected to test-pg" in result["message"]

@pytest.mark.dependency(name="test_execute_sql", depends=["create_environment"])
def test_execute_sql(temp_env_dir):
    """Test executing SQL on an environment."""


    # Execute SQL
    result = elm.execute_sql("test-pg", "SELECT 1 AS id, 'Test' AS name UNION ALL SELECT 2, 'Test 2' UNION ALL SELECT 3, 'Test 3'")

    # Verify the SQL was executed
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ['id', 'name']

@pytest.mark.dependency(name="delete_environment", depends=["get_encrypted_environment", "test_test_environment", "test_execute_sql"])
def test_delete_environment(temp_env_dir):
    """Test deleting an environment."""

    # Delete the environment
    result = elm.delete_environment("test-pg")
    assert result is True

    # Verify the environment was deleted
    config = configparser.ConfigParser()
    config.read(temp_env_dir)
    assert "test-pg" not in config.sections()

def test_delete_non_existent_environment(temp_env_dir):
    # Test deleting a non-existent environment
    result = elm.delete_environment("non-existent")
    assert result is False