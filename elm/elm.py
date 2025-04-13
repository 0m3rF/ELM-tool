import click, os
from elm_commands import environment, mask, copy, generate
from elm_utils import venv, variables

def ensure_env_dir():
    """Ensure the environment directory exists."""
    if not os.path.exists(variables.ENVS_FILE):
        os.makedirs(variables.ENVS_FILE, exist_ok=True)

@click.group()
@click.version_option()
def cli():
    """Extract, Load and Mask Tool for Database Operations"""
    pass

cli.add_command(environment.environment)
cli.add_command(copy.copy)
cli.add_command(mask.mask)
cli.add_command(generate.generate)

if __name__ == '__main__':
    venv.create_and_activate_venv(variables.VENV_DIR)
    cli()