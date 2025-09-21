import click, os
from elm.elm_commands import environment, mask, copy, generate, config
from elm.elm_utils import venv, variables
from elm.elm_utils.command_utils import AliasedGroup

def ensure_env_dir():
    """Ensure the environment directory exists."""
    if not os.path.exists(variables.ENVS_FILE):
        os.makedirs(variables.ENVS_FILE, exist_ok=True)

@click.group(cls=AliasedGroup)
@click.help_option('-h', '--help')
def cli():
    """Extract, Load and Mask Tool for Database Operations"""
    pass

cli.add_command(environment.environment)
cli.add_command(copy.copy)
cli.add_command(mask.mask)
cli.add_command(generate.generate)
cli.add_command(config.config)

# Define aliases for main commands
ALIASES = {
    'env': environment.environment,
    'cpy': copy.copy,
    'msk': mask.mask,
    'gen': generate.generate,
    'cfg': config.config
}

if __name__ == '__main__':
    venv.create_and_activate_venv(variables.VENV_DIR)
    cli()