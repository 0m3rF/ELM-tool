"""
Configuration management commands for ELM Tool.

This module provides CLI commands for managing ELM Tool configuration,
including setting tool home directory, virtual environment directory,
and other configurable parameters.
"""

import click
import json
from elm.core import config as core_config
from elm.elm_utils.command_utils import AliasedGroup





@click.group(cls=AliasedGroup)
@click.help_option('-h', '--help')
def config():
    """Configuration management commands.
    
    This group contains commands for managing ELM Tool configuration settings.
    Use these commands to view, set, and reset configuration values like
    ELM_TOOL_HOME, virtual environment directory, and other tool settings.
    
    Examples:
    
        Show current configuration:
          elm-tool config show
        
        Set ELM_TOOL_HOME directory:
          elm-tool config set ELM_TOOL_HOME /path/to/elm/home
        
        Reset configuration to defaults:
          elm-tool config reset
    """
    pass


@config.command()
@click.help_option('-h', '--help')
def show():
    """Show current configuration and file paths.
    
    Displays the current configuration values and the paths to important
    files and directories used by ELM Tool.
    
    Examples:
    
        Show all configuration:
          elm-tool config show
        
        Using the info alias:
          elm-tool config info
    """
    result = core_config.show_config_info()
    
    if not result.success:
        raise click.UsageError(result.message)
    
    info = result.data
    config_data = info['config']
    paths = info['paths']
    venv_status = info.get('venv_status', {})

    click.echo("ELM Tool Configuration:")
    click.echo("=" * 50)

    click.echo("\nConfiguration Values:")
    for key, value in config_data.items():
        click.echo(f"  {key}: {value}")

    click.echo("\nVirtual Environment Status:")
    if venv_status:
        initialized = venv_status.get('initialized', False)
        exists = venv_status.get('exists', False)

        init_status = "✓ Initialized" if initialized else "✗ Not initialized"
        exists_status = "✓ Exists" if exists else "✗ Does not exist"

        click.echo(f"  Status: {init_status}")
        click.echo(f"  Directory: {exists_status}")

        if not initialized or not exists:
            click.echo("  ℹ️  Virtual environment will be created on next operation")

    click.echo("\nFile Paths:")
    for key, path in paths.items():
        click.echo(f"  {key}: {path}")


@config.command()
@click.argument('key')
@click.argument('value')
@click.help_option('-h', '--help')
def set(key, value):
    """Set a configuration value.
    
    Sets a configuration parameter to the specified value.
    Common configuration keys include:
    
    - ELM_TOOL_HOME: The home directory for ELM Tool files
    - VENV_NAME: The name of the virtual environment directory
    - APP_NAME: The application name
    
    Examples:
    
        Set ELM_TOOL_HOME:
          elm-tool config set ELM_TOOL_HOME /path/to/elm/home
        
        Set virtual environment name:
          elm-tool config set VENV_NAME my_venv_elm
        
        Using the update alias:
          elm-tool config update ELM_TOOL_HOME /new/path
    """
    # Validate key
    valid_keys = ['ELM_TOOL_HOME', 'VENV_NAME', 'APP_NAME']
    if key not in valid_keys:
        click.echo(f"Warning: '{key}' is not a standard configuration key.")
        click.echo(f"Valid keys are: {', '.join(valid_keys)}")
        if not click.confirm("Do you want to continue?"):
            return
    
    result = core_config.set_config(key, value)
    
    if result.success:
        click.echo(result.message)
        click.echo(f"Note: You may need to restart the tool for some changes to take effect.")
    else:
        raise click.UsageError(result.message)


@config.command()
@click.argument('key')
@click.help_option('-h', '--help')
def get(key):
    """Get a specific configuration value.
    
    Retrieves and displays the value of a specific configuration parameter.
    
    Examples:
    
        Get ELM_TOOL_HOME:
          elm-tool config get ELM_TOOL_HOME
        
        Get virtual environment name:
          elm-tool config get VENV_NAME
    """
    manager = core_config.get_config_manager()
    value = manager.get_config_value(key)
    
    if value is not None:
        click.echo(f"{key}: {value}")
    else:
        click.echo(f"Configuration key '{key}' not found")


@config.command()
@click.confirmation_option(prompt='Are you sure you want to reset all configuration to defaults?')
@click.help_option('-h', '--help')
def reset():
    """Reset configuration to default values.
    
    Resets all configuration values to their defaults. This will:
    - Set ELM_TOOL_HOME to the default user config directory
    - Reset VENV_NAME to the default virtual environment name
    - Reset APP_NAME to the default application name
    
    Examples:
    
        Reset configuration:
          elm-tool config reset
    """
    result = core_config.reset_config()
    
    if result.success:
        click.echo(result.message)
        click.echo("Configuration has been reset to defaults.")
    else:
        raise click.UsageError(result.message)


@config.command()
@click.help_option('-h', '--help')
def paths():
    """Show important file and directory paths.
    
    Displays the paths to important files and directories used by ELM Tool,
    including configuration files, environment definitions, and data files.
    
    Examples:
    
        Show all paths:
          elm-tool config paths
        
        Using the dirs alias:
          elm-tool config dirs
    """
    result = core_config.show_config_info()
    
    if not result.success:
        raise click.UsageError(result.message)
    
    paths = result.data['paths']
    
    click.echo("ELM Tool File Paths:")
    click.echo("=" * 40)
    
    for key, path in paths.items():
        # Check if path exists
        import os
        exists = "✓" if os.path.exists(path) else "✗"
        click.echo(f"  {exists} {key}: {path}")
    
    click.echo("\n✓ = exists, ✗ = does not exist")


# Define aliases for commands
ALIASES = {
    'info': show,
    'display': show,
    'update': set,
    'change': set,
    'dirs': paths,
    'directories': paths,
}
