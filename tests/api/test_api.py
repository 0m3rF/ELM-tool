"""
Tests for the API module.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os

import elm


def test_create_environment():
    """Test creating an environment."""
    with patch('configparser.ConfigParser.write') as mock_write:
        # Mock the ConfigParser to avoid actual file operations
        with patch('configparser.ConfigParser.read'), \
             patch('configparser.ConfigParser.sections') as mock_sections, \
             patch('os.makedirs') as mock_makedirs:

            # Mock sections to return an empty list (no existing environments)
            mock_sections.return_value = []

            # Test creating a basic environment
            result = elm.create_environment(
                name="test-env",
                host="localhost",
                port=5432,
                user="postgres",
                password="password",
                service="postgres",
                db_type="postgres"
            )

            assert result is True
            mock_write.assert_called_once()
            mock_makedirs.assert_called_once()

            # Test creating an encrypted environment
            mock_write.reset_mock()
            mock_makedirs.reset_mock()

            # Mock the encryption functions
            with patch('elm.elm_utils.encryption.generate_key_from_password') as mock_gen_key, \
                 patch('elm.elm_utils.encryption.encrypt_data') as mock_encrypt:

                # Set up the mock return values
                mock_gen_key.return_value = (b'test-key', b'test-salt')
                mock_encrypt.return_value = 'encrypted-password'

                result = elm.create_environment(
                    name="secure-env",
                    host="localhost",
                    port=5432,
                    user="postgres",
                    password="password",
                    service="postgres",
                    db_type="postgres",
                    encrypt=True,
                    encryption_key="secret-key"
                )

                assert result is True
                mock_write.assert_called_once()
                mock_gen_key.assert_called_once_with("secret-key")
                mock_encrypt.assert_called_once()


def test_list_environments():
    """Test listing environments."""
    with patch('configparser.ConfigParser.read'), \
         patch('configparser.ConfigParser.sections') as mock_sections, \
         patch('configparser.ConfigParser.__getitem__') as mock_getitem:

        # Mock the sections to include our test environments
        mock_sections.return_value = ['env1', 'env2']

        # Mock the __getitem__ to return our environment configs
        def getitem_side_effect(key):
            if key == 'env1':
                return {
                    'type': 'postgres',
                    'host': 'localhost',
                    'port': '5432',
                    'user': 'postgres',
                    'password': 'secret',
                    'service': 'mydb',
                    'is_encrypted': 'False'
                }
            elif key == 'env2':
                return {
                    'type': 'mysql',
                    'host': 'localhost',
                    'port': '3306',
                    'user': 'root',
                    'password': 'secret',
                    'service': 'mydb',
                    'is_encrypted': 'False'
                }
            else:
                raise KeyError(key)

        mock_getitem.side_effect = getitem_side_effect

        # Test listing environments
        result = elm.list_environments()

        assert len(result) == 2
        assert result[0]["name"] == "env1"
        assert result[1]["name"] == "env2"

        # Test with show_all=True
        result = elm.list_environments(show_all=True)

        assert len(result) == 2
        assert "password" in result[0]
        assert result[0]["password"] == "********"  # Password should be masked


def test_get_environment():
    """Test getting environment details."""
    with patch('configparser.ConfigParser.read'), \
         patch('configparser.ConfigParser.sections') as mock_sections, \
         patch('configparser.ConfigParser.__getitem__') as mock_getitem:

        # Mock the sections to include our test environment
        mock_sections.return_value = ['test-env', 'secure-env']

        # Mock the __getitem__ to return our environment configs
        def getitem_side_effect(key):
            if key == 'test-env':
                return {
                    'type': 'postgres',
                    'host': 'localhost',
                    'port': '5432',
                    'user': 'postgres',
                    'password': 'password',
                    'service': 'postgres',
                    'is_encrypted': 'False'
                }
            elif key == 'secure-env':
                return {
                    'type': 'postgres',
                    'host': 'localhost',
                    'port': '5432',
                    'user': 'postgres',
                    'password': 'encrypted-password',
                    'service': 'postgres',
                    'is_encrypted': 'True',
                    'salt': 'c29tZV9zYWx0'  # base64 encoded 'some_salt'
                }
            else:
                raise KeyError(key)

        mock_getitem.side_effect = getitem_side_effect

        # Test getting environment details
        result = elm.get_environment("test-env")

        assert result is not None
        assert result["name"] == "test-env"
        assert result["type"] == "postgres"

        # Test with encryption key
        with patch('elm.elm_utils.encryption.generate_key_from_password') as mock_gen_key, \
             patch('elm.elm_utils.encryption.decrypt_data') as mock_decrypt:

            # Set up the mock return values
            mock_gen_key.return_value = (b'test-key', b'test-salt')
            mock_decrypt.return_value = 'decrypted-password'

            result = elm.get_environment("secure-env", encryption_key="secret-key")

            assert result is not None
            assert result["name"] == "secure-env"
            assert result["is_encrypted"] == "True"

        # Test with non-existent environment
        result = elm.get_environment("non-existent")
        assert result is None


def test_delete_environment():
    """Test deleting an environment."""
    with patch('configparser.ConfigParser.read'), \
         patch('configparser.ConfigParser.sections') as mock_sections, \
         patch('configparser.ConfigParser.remove_section') as mock_remove_section, \
         patch('configparser.ConfigParser.write') as mock_write:

        # Mock the sections to include our test environment
        mock_sections.return_value = ['test-env']

        # Test deleting an environment
        result = elm.delete_environment("test-env")

        assert result is True
        mock_remove_section.assert_called_with("test-env")
        mock_write.assert_called_once()

        # Test deleting a non-existent environment
        mock_sections.return_value = []
        mock_remove_section.reset_mock()
        mock_write.reset_mock()

        result = elm.delete_environment("non-existent")
        assert result is False
        mock_remove_section.assert_not_called()
        mock_write.assert_not_called()


def test_test_environment():
    """Test testing an environment connection."""
    # Test successful connection
    with patch('elm.elm_utils.db_utils.get_connection_url') as mock_get_url, \
         patch('sqlalchemy.create_engine') as mock_create_engine, \
         patch('sqlalchemy.text') as mock_text:

        # Set up the mocks
        mock_get_url.return_value = 'postgresql://postgres:secret@localhost:5432/mydb'
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_result = MagicMock()
        mock_connection.execute.return_value = mock_result
        mock_text.return_value = "SELECT 1"

        # Test successful connection
        result = elm.test_environment("test-env")

        assert result["success"] is True
        assert "Successfully connected" in result["message"]

    # Test failed connection
    with patch('elm.elm_utils.db_utils.get_connection_url') as mock_get_url:
        # Make the connection fail
        mock_get_url.side_effect = ValueError("Environment 'test-env' not found")

        # Test failed connection
        result = elm.test_environment("test-env")

        assert result["success"] is False
        assert "Failed to connect" in result["message"]


def test_execute_sql():
    """Test executing SQL on an environment."""
    with patch('elm.elm_utils.db_utils.execute_query') as mock_execute:
        # Mock the return value
        mock_df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        mock_execute.return_value = mock_df

        # Test executing SQL
        result = elm.execute_sql("test-env", "SELECT * FROM test_table")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        # Test with apply_masks=True
        mock_execute.reset_mock()
        mock_execute.return_value = mock_df

        result = elm.execute_sql(
            "test-env",
            "SELECT * FROM test_table",
            apply_masks=True
        )

        # Check that the apply_mask parameter was passed correctly
        _, kwargs = mock_execute.call_args
        assert kwargs['apply_mask'] is True


def test_copy_db_to_file():
    """Test copying data from database to file."""
    with patch('elm.elm_commands.copy.db_to_file') as mock_copy:
        # Mock the return value
        mock_copy.return_value = {
            "success": True,
            "message": "Successfully copied 5 records to test_output.csv",
            "record_count": 5
        }

        # Test copying data
        result = elm.copy_db_to_file(
            source_env="test-env",
            query="SELECT * FROM test_table",
            file_path="test_output.csv",
            file_format="csv"
        )

        assert result["success"] is True
        assert result["record_count"] == 5

        # Test with apply_masks=True
        mock_copy.reset_mock()
        result = elm.copy_db_to_file(
            source_env="test-env",
            query="SELECT * FROM test_table",
            file_path="test_output.csv",
            file_format="csv",
            apply_masks=True
        )

        _, kwargs = mock_copy.call_args
        assert kwargs['apply_masks'] is True


def test_copy_file_to_db():
    """Test copying data from file to database."""
    with patch('elm.elm_commands.copy.file_to_db') as mock_copy:
        # Mock the return value
        mock_copy.return_value = {
            "success": True,
            "message": "Successfully copied 5 records to test_table",
            "record_count": 5
        }

        # Test copying data
        result = elm.copy_file_to_db(
            file_path="test_input.csv",
            target_env="test-env",
            table="test_table",
            file_format="csv",
            mode="APPEND"
        )

        assert result["success"] is True
        assert result["record_count"] == 5

        # Test with validate_target=True
        mock_copy.reset_mock()
        result = elm.copy_file_to_db(
            file_path="test_input.csv",
            target_env="test-env",
            table="test_table",
            file_format="csv",
            mode="APPEND",
            validate_target=True
        )

        _, kwargs = mock_copy.call_args
        assert kwargs['validate_target'] is True


def test_copy_db_to_db():
    """Test copying data from database to database."""
    with patch('elm.elm_commands.copy.db_to_db') as mock_copy:
        # Mock the return value
        mock_copy.return_value = {
            "success": True,
            "message": "Successfully copied 5 records from source-env to target-env",
            "record_count": 5
        }

        # Test copying data
        result = elm.copy_db_to_db(
            source_env="source-env",
            target_env="target-env",
            query="SELECT * FROM source_table",
            table="target_table",
            mode="APPEND"
        )

        assert result["success"] is True
        assert result["record_count"] == 5

        # Test with apply_masks=True
        mock_copy.reset_mock()
        result = elm.copy_db_to_db(
            source_env="source-env",
            target_env="target-env",
            query="SELECT * FROM source_table",
            table="target_table",
            mode="APPEND",
            apply_masks=True
        )

        _, kwargs = mock_copy.call_args
        assert kwargs['apply_masks'] is True


def test_add_mask():
    """Test adding a masking rule."""
    with patch('elm.elm_commands.mask.add') as mock_add:
        # Mock the return value
        mock_add.return_value = True

        # Test adding a global masking rule
        result = elm.add_mask(
            column="password",
            algorithm="star"
        )

        assert result is True
        mock_add.assert_called_once()

        # Test adding an environment-specific masking rule
        mock_add.reset_mock()
        result = elm.add_mask(
            column="credit_card",
            algorithm="star_length",
            environment="prod",
            length=4
        )

        assert result is True
        _, kwargs = mock_add.call_args
        assert kwargs['column'] == "credit_card"
        assert kwargs['algorithm'] == "star_length"
        assert kwargs['environment'] == "prod"
        assert kwargs['length'] == 4


def test_remove_mask():
    """Test removing a masking rule."""
    with patch('elm.elm_commands.mask.remove') as mock_remove:
        # Mock the return value
        mock_remove.return_value = True

        # Test removing a global masking rule
        result = elm.remove_mask(
            column="password"
        )

        assert result is True
        mock_remove.assert_called_with("password", None)

        # Test removing an environment-specific masking rule
        mock_remove.reset_mock()
        result = elm.remove_mask(
            column="credit_card",
            environment="prod"
        )

        assert result is True
        mock_remove.assert_called_with("credit_card", "prod")


def test_list_masks():
    """Test listing masking rules."""
    with patch('elm.elm_commands.mask.list') as mock_list:
        # Mock the return value
        mock_list.return_value = {
            'global': {
                'password': {
                    'algorithm': 'star',
                    'params': {}
                }
            },
            'environments': {
                'prod': {
                    'credit_card': {
                        'algorithm': 'star_length',
                        'params': {'length': 4}
                    }
                }
            }
        }

        # Test listing masking rules
        result = elm.list_masks()

        assert 'global' in result
        assert 'environments' in result
        assert 'password' in result['global']
        assert 'prod' in result['environments']
        assert 'credit_card' in result['environments']['prod']


def test_test_mask():
    """Test testing a masking rule."""
    with patch('elm.elm_commands.mask.test') as mock_test:
        # Mock the return value
        mock_test.return_value = {
            'column': 'password',
            'original': 'secret123',
            'masked': '*****',
            'environment': None
        }

        # Test testing a global masking rule
        result = elm.test_mask(
            column="password",
            value="secret123"
        )

        assert result['column'] == 'password'
        assert result['original'] == 'secret123'
        assert result['masked'] == '*****'

        # Test testing an environment-specific masking rule
        mock_test.reset_mock()
        mock_test.return_value = {
            'column': 'credit_card',
            'original': '1234-5678-9012-3456',
            'masked': '1234************',
            'environment': 'prod'
        }

        result = elm.test_mask(
            column="credit_card",
            value="1234-5678-9012-3456",
            environment="prod"
        )

        assert result['column'] == 'credit_card'
        assert result['environment'] == 'prod'


def test_generate_data():
    """Test generating random data."""
    with patch('elm.elm_commands.generate.generate_data') as mock_generate:
        # Mock the return value
        mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Test 1', 'Test 2', 'Test 3'],
            'email': ['test1@example.com', 'test2@example.com', 'test3@example.com']
        })
        mock_generate.return_value = mock_df

        # Test generating data
        result = elm.generate_data(
            num_records=3,
            columns=["id", "name", "email"]
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['id', 'name', 'email']

        # Test with patterns
        mock_generate.reset_mock()
        result = elm.generate_data(
            num_records=3,
            columns=["id", "name", "email"],
            pattern={"email": "email", "name": "name"}
        )

        _, kwargs = mock_generate.call_args
        assert 'pattern' in kwargs
        assert kwargs['pattern']['email'] == 'email'


def test_generate_and_save():
    """Test generating and saving random data."""
    with patch('elm.elm_commands.generate.generate_and_save') as mock_generate_save:
        # Mock the return value
        mock_generate_save.return_value = {
            'success': True,
            'message': 'Successfully wrote 3 records to test_output.csv',
            'record_count': 3,
            'data': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Test 1', 'Test 2', 'Test 3']
            })
        }

        # Test generating and saving to file
        result = elm.generate_and_save(
            num_records=3,
            columns=["id", "name"],
            output="test_output.csv",
            format="csv"
        )

        assert result['success'] is True
        assert result['record_count'] == 3

        # Test generating and saving to database
        mock_generate_save.reset_mock()
        result = elm.generate_and_save(
            num_records=3,
            columns=["id", "name"],
            environment="test-env",
            table="test_table",
            write_to_db=True
        )

        _, kwargs = mock_generate_save.call_args
        assert kwargs['write_to_db'] is True
        assert kwargs['environment'] == 'test-env'
        assert kwargs['table'] == 'test_table'
