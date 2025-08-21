"""
Tests for the encryption utilities.
"""
import pytest
import base64
from unittest.mock import patch, MagicMock

from elm.elm_utils.encryption import (
    generate_key_from_password,
    encrypt_data,
    decrypt_data,
    encrypt_environment,
    decrypt_environment
)


def test_generate_key_from_password():
    """Test generating a key from a password."""
    # Test with a string password and no salt
    key1, salt1 = generate_key_from_password("test_password")
    assert isinstance(key1, bytes)
    assert isinstance(salt1, bytes)
    assert len(salt1) == 16
    
    # Test with a bytes password and no salt
    key2, salt2 = generate_key_from_password(b"test_password")
    assert isinstance(key2, bytes)
    assert isinstance(salt2, bytes)
    
    # Test with a provided salt as bytes
    salt3 = b"0123456789abcdef"
    key3, salt3_returned = generate_key_from_password("test_password", salt3)
    assert salt3 == salt3_returned
    
    # Test with a provided salt as string
    salt4 = "0123456789abcdef"
    key4, salt4_returned = generate_key_from_password("test_password", salt4)
    assert isinstance(salt4_returned, bytes)
    
    # Test that the same password and salt produce the same key
    key5a, _ = generate_key_from_password("same_password", b"same_salt")
    key5b, _ = generate_key_from_password("same_password", b"same_salt")
    assert key5a == key5b
    
    # Test that different passwords produce different keys
    key6a, _ = generate_key_from_password("password1", b"same_salt")
    key6b, _ = generate_key_from_password("password2", b"same_salt")
    assert key6a != key6b
    
    # Test that different salts produce different keys
    key7a, _ = generate_key_from_password("same_password", b"salt1")
    key7b, _ = generate_key_from_password("same_password", b"salt2")
    assert key7a != key7b


def test_encrypt_decrypt_data():
    """Test encrypting and decrypting data."""
    # Generate a key for testing
    key, _ = generate_key_from_password("test_password")
    
    # Test with string data
    original_data = "This is a test string"
    encrypted = encrypt_data(original_data, key)
    assert isinstance(encrypted, str)
    assert encrypted != original_data
    
    decrypted = decrypt_data(encrypted, key)
    assert decrypted == original_data
    
    # Test with bytes data
    original_bytes = b"This is a test bytes object"
    encrypted_bytes = encrypt_data(original_bytes, key)
    assert isinstance(encrypted_bytes, str)
    
    decrypted_bytes = decrypt_data(encrypted_bytes, key)
    assert decrypted_bytes == original_bytes.decode('utf-8')
    
    # Test decrypting with encrypted data as bytes
    encrypted_str = encrypt_data("test", key)
    encrypted_bytes = base64.b64decode(encrypted_str.encode('utf-8'))
    decrypted = decrypt_data(encrypted_bytes, key)
    assert decrypted == "test"


def test_encrypt_environment():
    """Test encrypting environment data."""
    # Create test environment data
    env_data = {
        'host': 'localhost',
        'port': '5432',
        'user': 'postgres',
        'password': 'secret',
        'service': 'postgres',
        'type': 'postgres'
    }
    
    # Encrypt the environment
    encrypted_env = encrypt_environment(env_data, "test_key")
    
    # Verify the encrypted environment
    assert encrypted_env['is_encrypted'] == 'True'
    assert 'salt' in encrypted_env
    
    # Check that all fields are encrypted
    for field in ['host', 'port', 'user', 'password', 'service', 'type']:
        assert field in encrypted_env
        assert encrypted_env[field] != env_data[field]


def test_decrypt_environment():
    """Test decrypting environment data."""
    # Create test environment data
    env_data = {
        'host': 'localhost',
        'port': '5432',
        'user': 'postgres',
        'password': 'secret',
        'service': 'postgres',
        'type': 'postgres'
    }
    
    # Encrypt and then decrypt the environment
    encryption_key = "test_key"
    encrypted_env = encrypt_environment(env_data, encryption_key)
    decrypted_env = decrypt_environment(encrypted_env, encryption_key)
    
    # Verify the decrypted environment
    assert decrypted_env['is_encrypted'] == 'False'
    
    # Check that all fields are correctly decrypted
    for field in ['host', 'port', 'user', 'password', 'service', 'type']:
        assert decrypted_env[field] == env_data[field]
    
    # Test with non-encrypted environment
    non_encrypted = {'is_encrypted': 'False', 'host': 'localhost'}
    result = decrypt_environment(non_encrypted, "any_key")
    assert result == non_encrypted
