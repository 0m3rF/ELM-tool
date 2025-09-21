import os
from platformdirs import user_config_dir

# Try to import config manager, fall back to defaults if not available
try:
    from elm.core.config import get_config_manager
    _config_manager = get_config_manager()

    APP_NAME = _config_manager.get_config_value("APP_NAME") or "ELMtool"
    VENV_NAME = _config_manager.get_config_value("VENV_NAME") or f"venv_{APP_NAME}"
    ELM_TOOL_HOME = _config_manager.get_elm_tool_home()
    VENV_DIR = _config_manager.get_venv_dir()
    ENVS_FILE = _config_manager.get_envs_file()
    MASK_FILE = _config_manager.get_mask_file()

except ImportError:
    # Fallback to original behavior if config manager is not available
    APP_NAME = "ELMtool"
    VENV_NAME = "venv_" + APP_NAME
    ELM_TOOL_HOME = os.getenv("ELM_TOOL_HOME", user_config_dir(".", APP_NAME))
    VENV_DIR = os.path.join(ELM_TOOL_HOME, VENV_NAME)
    ENVS_FILE = os.path.join(ELM_TOOL_HOME, "environments.ini")
    MASK_FILE = os.path.join(ELM_TOOL_HOME, "masking.json")