"""
Tests for the masking algorithms.
"""
import pytest
import pandas as pd
import numpy as np

from elm.elm_utils.mask_algorithms import (
    star_mask,
    star_mask_with_length,
    random_replace,
    nullify,
    MASKING_ALGORITHMS
)


def test_star_mask():
    """Test the star masking algorithm."""
    # Test with a normal string
    assert star_mask("password123") == "*****"

    # Test with an empty string
    assert star_mask("") == "*****"

    # Test with None
    assert star_mask(None) is None

    # Test with NaN
    result = star_mask(float('nan'))
    assert pd.isna(result)  # Use pandas isna to check for NaN

    # Test with a non-string value
    assert star_mask(123) == 123

    # Test with a pandas NA value
    assert star_mask(pd.NA) is pd.NA


def test_star_mask_with_length():
    """Test the star masking with length algorithm."""
    # Test with a normal string and default length
    assert star_mask_with_length("password123") == "pass*******"

    # Test with a normal string and custom length
    # The correct output should have 10 stars (length of "password123" - 2)
    assert star_mask_with_length("password123", length=2) == "pa*********"

    # Test with a string shorter than the length
    assert star_mask_with_length("abc", length=4) == "abc"

    # Test with an empty string
    assert star_mask_with_length("") == ""

    # Test with None
    assert star_mask_with_length(None) is None

    # Test with NaN
    result = star_mask_with_length(float('nan'))
    assert pd.isna(result)  # Use pandas isna to check for NaN

    # Test with a non-string value
    assert star_mask_with_length(123) == 123

    # Test with a pandas NA value
    assert star_mask_with_length(pd.NA) is pd.NA

    # Test with length=0
    assert star_mask_with_length("password123", length=0) == "***********"


def test_random_replace():
    """Test the random replacement algorithm."""
    # Test with a normal string
    result = random_replace("password123")
    assert isinstance(result, str)
    assert len(result) == len("password123")
    assert result != "password123"  # Very unlikely to be the same

    # Test with an empty string
    assert random_replace("") == ""

    # Test with None
    assert random_replace(None) is None

    # Test with NaN
    result = random_replace(float('nan'))
    assert pd.isna(result)  # Use pandas isna to check for NaN

    # Test with a non-string value
    assert random_replace(123) == 123

    # Test with a pandas NA value
    assert random_replace(pd.NA) is pd.NA

    # Test with a fixed seed for reproducibility
    import random
    random.seed(42)
    result1 = random_replace("test")
    random.seed(42)
    result2 = random_replace("test")
    assert result1 == result2


def test_nullify():
    """Test the nullify algorithm."""
    # Test with a normal string
    assert nullify("password123") is None

    # Test with an empty string
    assert nullify("") is None

    # Test with None
    assert nullify(None) is None

    # Test with NaN
    assert nullify(float('nan')) is None

    # Test with a non-string value
    assert nullify(123) is None

    # Test with a pandas NA value
    assert nullify(pd.NA) is None


def test_masking_algorithms_dict():
    """Test the MASKING_ALGORITHMS dictionary."""
    # Check that all algorithms are in the dictionary
    assert 'star' in MASKING_ALGORITHMS
    assert 'star_length' in MASKING_ALGORITHMS
    assert 'random' in MASKING_ALGORITHMS
    assert 'nullify' in MASKING_ALGORITHMS

    # Check that the functions are correctly mapped
    assert MASKING_ALGORITHMS['star'] == star_mask
    assert MASKING_ALGORITHMS['star_length'] == star_mask_with_length
    assert MASKING_ALGORITHMS['random'] == random_replace
    assert MASKING_ALGORITHMS['nullify'] == nullify


def test_masking_algorithms_on_dataframe():
    """Test applying masking algorithms to a DataFrame."""
    # Create a test DataFrame
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com'],
        'password': ['secret1', 'secret2', 'secret3'],
        'ssn': ['123-45-6789', '234-56-7890', '345-67-8901'],
        'notes': ['Note 1', 'Note 2', None]
    })

    # Apply star masking to password column
    df['password_masked'] = df['password'].apply(star_mask)
    assert all(df['password_masked'] == "*****")

    # Apply star_length masking to ssn column
    df['ssn_masked'] = df['ssn'].apply(lambda x: star_mask_with_length(x, length=3))
    assert all(df['ssn_masked'].str.startswith('123') | df['ssn_masked'].str.startswith('234') | df['ssn_masked'].str.startswith('345'))
    assert all(df['ssn_masked'].str.endswith('*******'))

    # Apply random replacement to email column
    df['email_masked'] = df['email'].apply(random_replace)
    assert all(df['email'] != df['email_masked'])
    assert all(df['email'].str.len() == df['email_masked'].str.len())

    # Apply nullify to notes column
    df['notes_masked'] = df['notes'].apply(nullify)
    assert all(df['notes_masked'].isna())
