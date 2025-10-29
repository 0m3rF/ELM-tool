"""
Configuration management for ELM Tool.

This module handles reading, writing, and managing configuration settings
for the ELM Tool, including tool home directory, virtual environment directory,
and other configurable parameters.
"""

import os
import json
import configparser
from typing import Dict, Any, Optional
from platformdirs import user_config_dir

from elm.core.types import OperationResult
from elm.core.utils import create_success_result, create_error_result


class ConfigManager:
    """Manages ELM Tool configuration settings."""
    
    def __init__(self):
        self.app_name = "ELMtool"
        self.config_file_name = "config.json"
        self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        default_home = user_config_dir(".", self.app_name)
        return {
            "ELM_TOOL_HOME": os.getenv("ELM_TOOL_HOME", default_home),
            "VENV_NAME": f"venv_{self.app_name}",
            "APP_NAME": self.app_name,
            "venv_initialized": False
        }
    
    def _get_config_file_path(self) -> str:
        """Get the path to the configuration file."""
        # Use environment variable or default location
        elm_home = os.getenv("ELM_TOOL_HOME", user_config_dir(".", self.app_name))
        return os.path.join(elm_home, self.config_file_name)
    
    def _load_config(self) -> None:
        """Load configuration from file or create default."""
        config_path = self._get_config_file_path()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                # Ensure all default keys exist
                defaults = self._get_default_config()
                for key, value in defaults.items():
                    if key not in self.config:
                        self.config[key] = value
            except Exception:
                # If config file is corrupted, use defaults
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
    
    def _save_config(self) -> None:
        """Save configuration to file."""
        config_path = self._get_config_file_path()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()
    
    def get_config_value(self, key: str) -> Optional[Any]:
        """Get a specific configuration value."""
        return self.config.get(key)
    
    def set_config_value(self, key: str, value: Any) -> OperationResult:
        """Set a configuration value."""
        try:
            self.config[key] = value
            self._save_config()
            return create_success_result(f"Configuration '{key}' updated successfully")
        except Exception as e:
            return create_error_result(f"Failed to update configuration: {str(e)}")
    
    def reset_config(self) -> OperationResult:
        """Reset configuration to defaults."""
        try:
            self.config = self._get_default_config()
            self._save_config()
            return create_success_result("Configuration reset to defaults")
        except Exception as e:
            return create_error_result(f"Failed to reset configuration: {str(e)}")
    
    def get_elm_tool_home(self) -> str:
        """Get the ELM_TOOL_HOME directory."""
        return self.config.get("ELM_TOOL_HOME", user_config_dir(".", self.app_name))
    
    def get_venv_dir(self) -> str:
        """Get the virtual environment directory."""
        elm_home = self.get_elm_tool_home()
        venv_name = self.config.get("VENV_NAME", f"venv_{self.app_name}")
        return os.path.join(elm_home, venv_name)
    
    def get_envs_file(self) -> str:
        """Get the environments file path."""
        return os.path.join(self.get_elm_tool_home(), "environments.ini")
    
    def get_mask_file(self) -> str:
        """Get the masking file path."""
        return os.path.join(self.get_elm_tool_home(), "masking.json")

    def is_venv_initialized(self) -> bool:
        """Check if the virtual environment has been initialized."""
        return self.config.get("venv_initialized", False)

    def mark_venv_initialized(self, initialized: bool = True) -> OperationResult:
        """Mark the virtual environment as initialized or not initialized."""
        try:
            self.config["venv_initialized"] = initialized
            self._save_config()
            status = "initialized" if initialized else "not initialized"
            return create_success_result(f"Virtual environment marked as {status}")
        except Exception as e:
            return create_error_result(f"Failed to update venv status: {str(e)}")

    def check_venv_exists(self) -> bool:
        """Check if the virtual environment directory exists."""
        venv_dir = self.get_venv_dir()
        if not os.path.exists(venv_dir):
            return False

        # Check if it's a valid venv by looking for the Python executable
        if os.name == "nt":  # Windows
            python_path = os.path.join(venv_dir, "Scripts", "python.exe")
        else:  # Unix-like
            python_path = os.path.join(venv_dir, "bin", "python")

        return os.path.exists(python_path)


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> OperationResult:
    """Get current configuration."""
    try:
        config = get_config_manager().get_config()
        return create_success_result("Configuration retrieved successfully", data=config)
    except Exception as e:
        return create_error_result(f"Failed to get configuration: {str(e)}")


def set_config(key: str, value: Any) -> OperationResult:
    """Set a configuration value."""
    return get_config_manager().set_config_value(key, value)


def reset_config() -> OperationResult:
    """Reset configuration to defaults."""
    return get_config_manager().reset_config()


def show_config_info() -> OperationResult:
    """Show configuration information including file paths and venv status."""
    try:
        manager = get_config_manager()
        config = manager.get_config()

        info = {
            "config": config,
            "paths": {
                "config_file": manager._get_config_file_path(),
                "elm_tool_home": manager.get_elm_tool_home(),
                "venv_dir": manager.get_venv_dir(),
                "environments_file": manager.get_envs_file(),
                "masking_file": manager.get_mask_file()
            },
            "venv_status": {
                "initialized": manager.is_venv_initialized(),
                "exists": manager.check_venv_exists()
            }
        }

        return create_success_result("Configuration information retrieved", data=info)
    except Exception as e:
        return create_error_result(f"Failed to get configuration info: {str(e)}")
