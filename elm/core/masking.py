"""
ELM Tool Core Data Masking

Unified data masking operations for both CLI and API interfaces.
This module provides consistent masking rule management and data masking functionality.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional, List

from elm.core.types import MaskingConfig, MaskingAlgorithm, OperationResult
from elm.core.exceptions import MaskingError, ValidationError
from elm.core.utils import (
    validate_masking_algorithm, create_success_result, create_error_result,
    handle_exception, validate_required_params
)
from elm.elm_utils import variables
from elm.elm_utils.mask_algorithms import MASKING_ALGORITHMS


# Simple in-process cache for masking definitions to avoid repeated disk I/O
# during large streaming copy operations. The cache is keyed by the masking
# file's modification time so that changes to the mask file are picked up
# automatically on the next access.
_mask_definitions_cache: Optional[Dict[str, Any]] = None
_mask_definitions_mtime: Optional[float] = None


def _invalidate_masking_cache() -> None:
    """Clear the cached masking definitions.

    This is called after we persist new masking definitions so subsequent
    masking operations see the updated rules.
    """
    global _mask_definitions_cache, _mask_definitions_mtime
    _mask_definitions_cache = None
    _mask_definitions_mtime = None


def _get_masking_definitions_cached() -> Dict[str, Any]:
    """Return masking definitions using a lightweight cache.

    For long-running db2db copy operations that process many batches, reading
    the masking file from disk on every batch becomes unnecessary overhead.
    This helper keeps definitions in memory and only reloads them when the
    underlying file's modification time changes.
    """
    global _mask_definitions_cache, _mask_definitions_mtime

    mask_file = variables.MASK_FILE

    try:
        mtime = os.path.getmtime(mask_file)
    except FileNotFoundError:
        # If the file does not exist, behave like load_masking_definitions()
        # which returns an empty definition structure.
        mtime = None

    # Fast path: reuse cached definitions if the file has not changed.
    if _mask_definitions_cache is not None and _mask_definitions_mtime == mtime:
        return _mask_definitions_cache

    # Either first load or the file changed on disk – read fresh definitions.
    definitions = load_masking_definitions()
    _mask_definitions_cache = definitions
    _mask_definitions_mtime = mtime
    return definitions


def load_masking_definitions() -> Dict[str, Any]:
    """Load masking definitions from the masking file with file locking for parallel safety."""
    from elm.elm_utils.file_lock import file_lock

    if not os.path.exists(variables.MASK_FILE):
        return {'global': {}, 'environments': {}}

    try:
        try:
            with file_lock(variables.MASK_FILE, timeout=10.0):
                with open(variables.MASK_FILE, 'r') as f:
                    return json.load(f)
        except TimeoutError:
            # If we can't get the lock, try reading without it
            # This is safe for reads in most cases
            with open(variables.MASK_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        raise MaskingError(f"Error loading masking definitions: {str(e)}")


def save_masking_definitions(definitions: Dict[str, Any]) -> None:
    """Save masking definitions to the masking file with file locking for parallel safety."""
    from elm.elm_utils.file_lock import file_lock

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(variables.MASK_FILE), exist_ok=True)

    try:
        with file_lock(variables.MASK_FILE, timeout=10.0):
            with open(variables.MASK_FILE, 'w') as f:
                json.dump(definitions, f, indent=2)
        # Ensure any cached definitions are invalidated so the next masking
        # operation sees the updated rules.
        _invalidate_masking_cache()
    except Exception as e:
        raise MaskingError(f"Error saving masking definitions: {str(e)}")


def add_mask(
    column: str,
    algorithm: str,
    environment: Optional[str] = None,
    length: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None
) -> OperationResult:
    """
    Add a masking rule for a column.
    
    Args:
        column: Column name
        algorithm: Masking algorithm (star, star_length, random, nullify)
        environment: Environment name (None for global)
        length: Length parameter for algorithms that need it
        params: Additional algorithm parameters
    
    Returns:
        OperationResult with success status and message
    """
    try:
        validate_required_params({'column': column, 'algorithm': algorithm}, 
                                ['column', 'algorithm'])
        
        # Validate algorithm
        algorithm_enum = validate_masking_algorithm(algorithm)
        
        # Load existing definitions
        definitions = load_masking_definitions()
        
        # Prepare the masking configuration
        mask_config = {
            'algorithm': algorithm_enum.value,
            'params': params or {}
        }
        
        # Add algorithm-specific parameters
        if length is not None:
            mask_config['params']['length'] = length
        
        # Add to global or environment-specific definitions
        if environment:
            if 'environments' not in definitions:
                definitions['environments'] = {}
            if environment not in definitions['environments']:
                definitions['environments'][environment] = {}
            definitions['environments'][environment][column] = mask_config
            scope = f"environment '{environment}'"
        else:
            if 'global' not in definitions:
                definitions['global'] = {}
            definitions['global'][column] = mask_config
            scope = "global"
        
        # Save the updated definitions
        save_masking_definitions(definitions)
        
        return create_success_result(
            f"Added {scope} masking for column '{column}' using {algorithm} algorithm"
        )
        
    except Exception as e:
        return handle_exception(e, "mask addition")


def remove_mask(column: str, environment: Optional[str] = None) -> OperationResult:
    """
    Remove a masking rule for a column.
    
    Args:
        column: Column name
        environment: Environment name (None for global)
    
    Returns:
        OperationResult with success status and message
    """
    try:
        validate_required_params({'column': column}, ['column'])
        
        # Load existing definitions
        definitions = load_masking_definitions()
        
        # Remove from global or environment-specific definitions
        removed = False
        if environment:
            if ('environments' in definitions and
                environment in definitions['environments'] and
                column in definitions['environments'][environment]):
                del definitions['environments'][environment][column]
                removed = True
                scope = f"environment '{environment}'"
        else:
            if 'global' in definitions and column in definitions['global']:
                del definitions['global'][column]
                removed = True
                scope = "global"
        
        if not removed:
            scope = f"environment '{environment}'" if environment else "global"
            return create_error_result(f"No {scope} masking found for column '{column}'")
        
        # Save the updated definitions
        save_masking_definitions(definitions)
        
        return create_success_result(f"Removed {scope} masking for column '{column}'")
        
    except Exception as e:
        return handle_exception(e, "mask removal")


def list_masks(environment: Optional[str] = None) -> OperationResult:
    """
    List masking rules.
    
    Args:
        environment: Environment name (None for all)
    
    Returns:
        OperationResult with masking rules
    """
    try:
        # Load existing definitions
        definitions = load_masking_definitions()
        
        if environment:
            # Return only environment-specific rules
            env_rules = definitions.get('environments', {}).get(environment, {})
            return create_success_result(
                f"Found {len(env_rules)} masking rule(s) for environment '{environment}'",
                data={'environment': environment, 'rules': env_rules},
                record_count=len(env_rules)
            )
        else:
            # Return all rules
            global_count = len(definitions.get('global', {}))
            env_count = sum(len(rules) for rules in definitions.get('environments', {}).values())
            total_count = global_count + env_count
            
            return create_success_result(
                f"Found {total_count} total masking rule(s) ({global_count} global, {env_count} environment-specific)",
                data=definitions,
                record_count=total_count
            )
        
    except Exception as e:
        return handle_exception(e, "mask listing")


def test_mask(
    column: str,
    value: str,
    environment: Optional[str] = None
) -> OperationResult:
    """
    Test a masking rule on a value.
    
    Args:
        column: Column name
        value: Value to mask
        environment: Environment name
    
    Returns:
        OperationResult with original and masked values
    """
    try:
        validate_required_params({'column': column, 'value': value}, 
                                ['column', 'value'])
        
        # Load existing definitions
        definitions = load_masking_definitions()
        
        # Find the masking configuration
        mask_config = None
        scope = None
        
        if environment:
            if ('environments' in definitions and
                environment in definitions['environments'] and
                column in definitions['environments'][environment]):
                mask_config = definitions['environments'][environment][column]
                scope = f"environment '{environment}'"
        
        if mask_config is None and 'global' in definitions and column in definitions['global']:
            mask_config = definitions['global'][column]
            scope = "global"
        
        if mask_config is None:
            scope_desc = f"environment '{environment}'" if environment else "global or environment-specific"
            return create_error_result(f"No {scope_desc} masking definition found for column '{column}'")
        
        # Apply masking
        algorithm = mask_config.get('algorithm', 'star')
        params = mask_config.get('params', {})
        
        if algorithm in MASKING_ALGORITHMS:
            masked_value = MASKING_ALGORITHMS[algorithm](value, **params)
            
            result_data = {
                'column': column,
                'original': value,
                'masked': masked_value,
                'algorithm': algorithm,
                'scope': scope,
                'environment': environment
            }
            
            return create_success_result(
                f"Applied {scope} masking for column '{column}' using {algorithm} algorithm",
                data=result_data
            )
        else:
            return create_error_result(f"Unknown masking algorithm: {algorithm}")
        
    except Exception as e:
        return handle_exception(e, "mask testing")


def apply_masking(
    data: pd.DataFrame, 
    environment: Optional[str] = None,
    definitions: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Apply masking to a DataFrame based on masking definitions.
    
    Args:
        data: DataFrame to mask
        environment: Environment name for environment-specific rules
        definitions: Pre-loaded masking definitions (optional)
    
    Returns:
        Masked DataFrame
    """
    # Non-DataFrame inputs are passed through unchanged (existing behaviour).
    if not isinstance(data, pd.DataFrame):
        return data

    # If definitions are not provided, use the cached loader to avoid
    # repeatedly hitting the filesystem for every batch during large copies.
    if definitions is None:
        definitions = _get_masking_definitions_cached()

    # Get global and environment-specific definitions, normalising to dicts.
    global_defs = definitions.get('global', {}) or {}
    env_defs: Dict[str, Any] = {}
    if environment and environment in definitions.get('environments', {}):
        env_defs = definitions['environments'][environment] or {}

    # Fast-path: if there are no masking rules at all, return the original
    # DataFrame without copying or iterating over columns.
    if not global_defs and not env_defs:
        return data

    # Determine which columns in this DataFrame actually need masking.
    # Environment-specific rules take precedence over global rules.
    columns_to_mask: Dict[str, Any] = {}
    for column in data.columns:
        if column in env_defs:
            columns_to_mask[column] = env_defs[column]
        elif column in global_defs:
            columns_to_mask[column] = global_defs[column]

    # If none of the DataFrame's columns are covered by masking rules, we can
    # return the original DataFrame without creating a copy.
    if not columns_to_mask:
        return data

    # Only now do we create a copy – this avoids unnecessary allocations when
    # masking is effectively a no-op for the given data.
    masked_data = data.copy()

    # Apply masking only to the columns that have rules, keeping the logic
    # identical to the previous implementation but avoiding work on
    # unaffected columns.
    for column, mask_config in columns_to_mask.items():
        algorithm = mask_config.get('algorithm', 'star')
        params = mask_config.get('params', {}) or {}

        mask_func = MASKING_ALGORITHMS.get(algorithm)
        if not mask_func:
            # Unknown algorithm – skip rather than failing hard, preserving
            # the previous behaviour of ignoring unknown algorithms here.
            continue

        # Resolve the algorithm function and params once per column and reuse
        # them inside the per-row apply to minimise per-row overhead.
        masked_data[column] = masked_data[column].apply(
            lambda x, func=mask_func, p=params: func(x, **p)
        )

    return masked_data
