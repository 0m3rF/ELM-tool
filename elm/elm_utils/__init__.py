"""
ELM Tool Utilities

This package contains utility modules for the ELM Tool.
These utilities are used by both the CLI and API interfaces.
"""

__version__ = "0.0.2"  # Keep in sync with main package version

# Export commonly used utilities
from .command_utils import AliasedGroup

__all__ = ['AliasedGroup']