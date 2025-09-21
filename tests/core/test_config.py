"""
Tests for the core config module.

This module tests configuration management functionality.
"""
import pytest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from elm.core.config import ConfigManager, get_config_manager, show_config_info, set_config, reset_config
from elm.core.types import OperationResult


class TestConfigManager:
    """Test ConfigManager class functionality."""

    def test_config_manager_init(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        
        assert config_manager.app_name == "ELMtool"
        assert config_manager.config_file_name == "config.json"
        assert hasattr(config_manager, 'config')
        assert isinstance(config_manager.config, dict)

    def test_get_default_config(self):
        """Test getting default configuration."""
        config_manager = ConfigManager()
        default_config = config_manager._get_default_config()
        
        assert isinstance(default_config, dict)
        assert "ELM_TOOL_HOME" in default_config
        assert "VENV_NAME" in default_config
        assert "APP_NAME" in default_config
        assert default_config["APP_NAME"] == "ELMtool"
        assert default_config["VENV_NAME"] == "venv_ELMtool"

    def test_get_config_file_path(self):
        """Test getting config file path."""
        config_manager = ConfigManager()
        config_path = config_manager._get_config_file_path()
        
        assert isinstance(config_path, str)
        assert config_path.endswith("config.json")

    def test_get_config_file_path_with_env_var(self):
        """Test config file path with environment variable."""
        test_home = "/tmp/test_elm_home"
        
        with patch.dict(os.environ, {'ELM_TOOL_HOME': test_home}):
            config_manager = ConfigManager()
            config_path = config_manager._get_config_file_path()
            
            expected_path = os.path.join(test_home, "config.json")
            assert config_path == expected_path

    def test_load_config_file_exists(self):
        """Test loading config when file exists."""
        test_config = {
            "ELM_TOOL_HOME": "/test/home",
            "VENV_NAME": "test_venv",
            "APP_NAME": "TestApp"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_file = f.name
        
        try:
            with patch.object(ConfigManager, '_get_config_file_path', return_value=config_file):
                config_manager = ConfigManager()
                
                assert config_manager.config["ELM_TOOL_HOME"] == "/test/home"
                assert config_manager.config["VENV_NAME"] == "test_venv"
                assert config_manager.config["APP_NAME"] == "TestApp"
        finally:
            os.unlink(config_file)

    def test_load_config_file_not_exists(self):
        """Test loading config when file doesn't exist."""
        non_existent_file = "/tmp/non_existent_config.json"
        
        with patch.object(ConfigManager, '_get_config_file_path', return_value=non_existent_file):
            config_manager = ConfigManager()
            
            # Should use default config
            assert config_manager.config["APP_NAME"] == "ELMtool"
            assert config_manager.config["VENV_NAME"] == "venv_ELMtool"

    def test_load_config_invalid_json(self):
        """Test loading config with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            config_file = f.name
        
        try:
            with patch.object(ConfigManager, '_get_config_file_path', return_value=config_file):
                config_manager = ConfigManager()
                
                # Should fall back to default config
                assert config_manager.config["APP_NAME"] == "ELMtool"
        finally:
            os.unlink(config_file)

    def test_save_config_success(self):
        """Test saving config successfully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            with patch.object(ConfigManager, '_get_config_file_path', return_value=config_file):
                config_manager = ConfigManager()
                config_manager.config["TEST_KEY"] = "test_value"

                # The _save_config method doesn't return anything, it just saves
                config_manager._save_config()

                # Verify file was written
                with open(config_file, 'r') as f:
                    saved_config = json.load(f)
                assert saved_config["TEST_KEY"] == "test_value"
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_save_config_directory_creation(self):
        """Test saving config creates directory if needed."""
        test_dir = tempfile.mkdtemp()
        config_file = os.path.join(test_dir, "subdir", "config.json")

        try:
            with patch.object(ConfigManager, '_get_config_file_path', return_value=config_file):
                config_manager = ConfigManager()

                config_manager._save_config()
                assert os.path.exists(config_file)
        finally:
            import shutil
            shutil.rmtree(test_dir)

    def test_get_config_value(self):
        """Test getting config values."""
        config_manager = ConfigManager()
        
        # Test existing key
        app_name = config_manager.get_config_value("APP_NAME")
        assert app_name == "ELMtool"
        
        # Test non-existing key
        non_existent = config_manager.get_config_value("NON_EXISTENT_KEY")
        assert non_existent is None

    def test_set_config_value(self):
        """Test setting config values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name

        try:
            with patch.object(ConfigManager, '_get_config_file_path', return_value=config_file):
                config_manager = ConfigManager()

                result = config_manager.set_config_value("TEST_KEY", "test_value")
                assert result.success is True
                assert config_manager.config["TEST_KEY"] == "test_value"
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_get_elm_tool_home(self):
        """Test getting ELM tool home directory."""
        config_manager = ConfigManager()
        home_dir = config_manager.get_elm_tool_home()
        
        assert isinstance(home_dir, str)
        assert len(home_dir) > 0

    def test_get_venv_dir(self):
        """Test getting virtual environment directory."""
        config_manager = ConfigManager()
        venv_dir = config_manager.get_venv_dir()
        
        assert isinstance(venv_dir, str)
        assert "venv" in venv_dir.lower()

    def test_get_envs_file(self):
        """Test getting environments file path."""
        config_manager = ConfigManager()
        envs_file = config_manager.get_envs_file()
        
        assert isinstance(envs_file, str)
        assert envs_file.endswith("environments.ini")

    def test_get_mask_file(self):
        """Test getting mask file path."""
        config_manager = ConfigManager()
        mask_file = config_manager.get_mask_file()
        
        assert isinstance(mask_file, str)
        assert mask_file.endswith("masking.json")


class TestConfigFunctions:
    """Test module-level config functions."""

    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns singleton."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, ConfigManager)

    def test_show_config_info_success(self):
        """Test showing config info successfully."""
        result = show_config_info()

        assert isinstance(result, OperationResult)
        assert result.success is True
        assert "config" in result.data
        assert "paths" in result.data

        config_data = result.data["config"]
        assert "ELM_TOOL_HOME" in config_data
        assert "VENV_NAME" in config_data
        assert "APP_NAME" in config_data

        paths_data = result.data["paths"]
        # Check for actual keys that exist in the paths data
        assert "config_file" in paths_data
        assert "elm_tool_home" in paths_data

    def test_set_config_success(self):
        """Test setting config value successfully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            with patch.object(ConfigManager, '_get_config_file_path', return_value=config_file):
                result = set_config("TEST_KEY", "test_value")
                
                assert isinstance(result, OperationResult)
                assert result.success is True
                assert "TEST_KEY" in result.message
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_set_config_failure(self):
        """Test setting config value with failure."""
        with patch.object(ConfigManager, '_save_config', side_effect=Exception("Save failed")):
            result = set_config("TEST_KEY", "test_value")

            assert isinstance(result, OperationResult)
            assert result.success is False
            assert "Failed to update configuration" in result.message

    def test_reset_config_success(self):
        """Test resetting config successfully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            with patch.object(ConfigManager, '_get_config_file_path', return_value=config_file):
                result = reset_config()
                
                assert isinstance(result, OperationResult)
                assert result.success is True
                assert "reset" in result.message.lower()
        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_reset_config_failure(self):
        """Test resetting config with failure."""
        with patch.object(ConfigManager, '_save_config', side_effect=Exception("Save failed")):
            result = reset_config()

            assert isinstance(result, OperationResult)
            assert result.success is False
            assert "Failed to reset configuration" in result.message
