import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generate_key_from_password(password, salt=None):
    """
    Generate a Fernet key from a password and optional salt.
    If salt is not provided, a random one will be generated.
    Returns a tuple of (key, salt).
    """
    if salt is None:
        salt = os.urandom(16)
    elif isinstance(salt, str):
        salt = salt.encode('utf-8')
    
    # Convert password to bytes if it's a string
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key, salt

def encrypt_data(data, key):
    """
    Encrypt data using the provided key.
    Returns the encrypted data as a base64 encoded string.
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    f = Fernet(key)
    encrypted_data = f.encrypt(data)
    return base64.b64encode(encrypted_data).decode('utf-8')

def decrypt_data(encrypted_data, key):
    """
    Decrypt data using the provided key.
    Returns the decrypted data as a string.
    """
    if isinstance(encrypted_data, str):
        encrypted_data = base64.b64decode(encrypted_data.encode('utf-8'))
    
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    return decrypted_data.decode('utf-8')

def encrypt_environment(env_data, encryption_key):
    """
    Encrypt environment data using the provided encryption key.
    Returns a dictionary with encrypted values.
    """
    key, salt = generate_key_from_password(encryption_key)
    encrypted_env = {
        'salt': base64.b64encode(salt).decode('utf-8'),
        'is_encrypted': 'True'
    }
    
    # Encrypt each field
    for field in ['host', 'port', 'user', 'password', 'service', 'type']:
        if field in env_data:
            encrypted_env[field] = encrypt_data(env_data[field], key)
    
    return encrypted_env

def decrypt_environment(encrypted_env, encryption_key):
    """
    Decrypt environment data using the provided encryption key.
    Returns a dictionary with decrypted values.
    """
    if encrypted_env.get('is_encrypted') != 'True':
        return encrypted_env
    
    salt = base64.b64decode(encrypted_env['salt'].encode('utf-8'))
    key, _ = generate_key_from_password(encryption_key, salt)
    
    decrypted_env = {'is_encrypted': 'False'}
    
    # Decrypt each field
    for field in ['host', 'port', 'user', 'password', 'service', 'type']:
        if field in encrypted_env:
            decrypted_env[field] = decrypt_data(encrypted_env[field], key)
    
    return decrypted_env
