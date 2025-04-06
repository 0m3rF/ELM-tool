import click, os
from elm_commands import config, mask, copy
from elm_utils import venv
from platformdirs import user_config_dir

APP_NAME = "ELMtool"
VENV_NAME = "venv_" + APP_NAME
ELM_TOOL_HOME = os.getenv("ELM_TOOL_HOME", user_config_dir(".", APP_NAME))
VENV_DIR = os.path.join(ELM_TOOL_HOME, VENV_NAME)
CONFIG_FILE = os.path.join(ELM_TOOL_HOME, "config.json")

def ensure_config_dir():
    """Ensure the config directory exists."""
    if not os.path.exists(CONFIG_FILE):
        os.makedirs(CONFIG_FILE, exist_ok=True)

@click.group()
@click.version_option()
def cli():
    """Extract, Load and Mask Tool for Database Operations"""
    pass

cli.add_command(config.config)
# cli.add_command(mask.mask)
# cli.add_command(copy.copy)

if __name__ == '__main__':
    venv.create_and_activate_venv(VENV_DIR)
    print(ELM_TOOL_HOME)
    print("main")
    cli()