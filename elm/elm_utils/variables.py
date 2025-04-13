import os
from platformdirs import user_config_dir

APP_NAME = "ELMtool"
VENV_NAME = "venv_" + APP_NAME
ELM_TOOL_HOME = os.getenv("ELM_TOOL_HOME", user_config_dir(".", APP_NAME))
VENV_DIR = os.path.join(ELM_TOOL_HOME, VENV_NAME)
ENVS_FILE = os.path.join(ELM_TOOL_HOME, "environments.ini")