import random
import string
import pandas as pd

# Masking algorithms
def star_mask(value, **kwargs):
    """Removes all data and puts stars (*)"""
    if value is None or pd.isna(value) or not isinstance(value, str):
        return value
    return '*****'

def star_mask_with_length(value, length=4, **kwargs):
    """Keep the first 'length' characters and replace the rest with stars (*)"""
    if value is None or pd.isna(value) or not isinstance(value, str):
        return value
    if len(value) <= length:
        return value
    return value[:length] + '*' * (len(value) - length)

def random_replace(value, **kwargs):
    """Replace with random characters of the same length"""
    if value is None or pd.isna(value) or not isinstance(value, str):
        return value
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(len(value)))

def nullify(value, **kwargs):
    """Replace with None"""
    return None

# Available masking algorithms
MASKING_ALGORITHMS = {
    'star': star_mask,
    'star_length': star_mask_with_length,
    'random': random_replace,
    'nullify': nullify
}