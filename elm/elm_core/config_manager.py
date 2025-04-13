import json
import os
from pathlib import Path

ELM_TOOL_HOME = os.getenv("ELM_TOOL_HOME", os.path.expanduser("~"))
CONFIG_DIR = os.path.join(ELM_TOOL_HOME, ".myprogram")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

def ensure_config_dir():
    """Ensure the config directory exists."""
    config_dir = os.path.dirname(CONFIG_PATH)
    Path(config_dir).mkdir(parents=True, exist_ok=True)

def load_config():
    """Load the configuration from the file."""
    ensure_config_dir()
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(config):
    """Save the configuration to the file."""
    ensure_config_dir()
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

def create_config(key, value):
    """Create or update a configuration key."""
    config = load_config()
    config[key] = value
    save_config(config)

def list_config():
    """List all configurations."""
    config = load_config()
    for key, value in config.items():
        print(f"{key}: {value}")

def update_config(key, value):
    """Update an existing configuration key."""
    config = load_config()
    if key in config:
        config[key] = value
        save_config(config)
    else:
        raise KeyError(f"Key '{key}' does not exist.")

def remove_config(key):
    """Remove a configuration key."""
    config = load_config()
    if key in config:
        del config[key]
        save_config(config)
    else:
        raise KeyError(f"Key '{key}' does not exist.")