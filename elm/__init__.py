"""
ELM Tool - Extract, Load and Mask Tool for Database Operations

This package provides tools for:
- Managing database environments
- Copying data between databases or files
- Masking sensitive data
- Generating test data

It can be used both as a command-line tool and as a Python library.
"""

__version__ = "0.0.9"

# Import main API functions for easy access
from elm.api import (
    # Environment Management
    create_environment,
    list_environments,
    get_environment,
    delete_environment,
    test_environment,
    execute_sql,

    # Data Copy
    copy_db_to_file,
    copy_file_to_db,
    copy_db_to_db,

    # Data Masking
    add_mask,
    remove_mask,
    list_masks,
    test_mask,

    # Data Generation
    generate_data,
    generate_and_save,

    # Configuration Management
    get_config,
    set_config,
    reset_config,
    get_config_info
)

# For backward compatibility
from elm.elm import cli