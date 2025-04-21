import click
import os
import json
import pandas as pd
from elm.elm_utils import variables
from elm.elm_utils.mask_algorithms import MASKING_ALGORITHMS


def load_masking_definitions():
    """Load masking definitions from the masking file"""
    if not os.path.exists(variables.MASK_FILE):
        return {'global': {}, 'environments': {}}

    try:
        with open(variables.MASK_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        click.echo(f"Error loading masking definitions: {str(e)}")
        return {'global': {}, 'environments': {}}

def save_masking_definitions(definitions):
    """Save masking definitions to the masking file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(variables.MASK_FILE), exist_ok=True)

    try:
        with open(variables.MASK_FILE, 'w') as f:
            json.dump(definitions, f, indent=2)
        return True
    except Exception as e:
        click.echo(f"Error saving masking definitions: {str(e)}")
        return False

def apply_masking(data, environment=None):
    """Apply masking to a DataFrame based on masking definitions"""
    if not isinstance(data, pd.DataFrame):
        return data

    # Load masking definitions
    definitions = load_masking_definitions()

    # Get global and environment-specific definitions
    global_defs = definitions.get('global', {})
    env_defs = {}
    if environment and environment in definitions.get('environments', {}):
        env_defs = definitions['environments'][environment]

    # Create a copy of the DataFrame to avoid modifying the original
    masked_data = data.copy()

    # Apply masking to each column
    for column in masked_data.columns:
        # Check if column has environment-specific masking
        if column in env_defs:
            mask_config = env_defs[column]
            algorithm = mask_config.get('algorithm', 'star')
            params = mask_config.get('params', {})

            if algorithm in MASKING_ALGORITHMS:
                masked_data[column] = masked_data[column].apply(
                    lambda x: MASKING_ALGORITHMS[algorithm](x, **params)
                )
        # Otherwise, check if column has global masking
        elif column in global_defs:
            mask_config = global_defs[column]
            algorithm = mask_config.get('algorithm', 'star')
            params = mask_config.get('params', {})

            if algorithm in MASKING_ALGORITHMS:
                masked_data[column] = masked_data[column].apply(
                    lambda x: MASKING_ALGORITHMS[algorithm](x, **params)
                )

    return masked_data

class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name].name
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)

@click.group(cls=AliasedGroup)
def mask():
    """Data masking commands for sensitive information"""
    pass

@mask.command()
@click.option("-c", "--column", required=True, help="Column name to mask")
@click.option("-a", "--algorithm", type=click.Choice(['star', 'star_length', 'random', 'nullify'], case_sensitive=False),
              default='star', help="Masking algorithm to use")
@click.option("-e", "--environment", help="Environment name (if not specified, applies globally)")
@click.option("-l", "--length", type=int, default=4, help="Length to keep for star_length algorithm")
def add(column, algorithm, environment, length):
    """Add a masking definition for a column

    Examples:

        Add global masking for a column:
          elm-tool mask add --column password --algorithm star

        Add environment-specific masking:
          elm-tool mask add --column credit_card --algorithm star_length --environment prod --length 6

        Nullify a column in development:
          elm-tool mask add --column ssn --algorithm nullify --environment dev
    """
    # Load existing definitions
    definitions = load_masking_definitions()

    # Prepare the masking configuration
    mask_config = {
        'algorithm': algorithm.lower(),
        'params': {}
    }

    # Add algorithm-specific parameters
    if algorithm.lower() == 'star_length':
        mask_config['params']['length'] = length

    # Add to global or environment-specific definitions
    if environment:
        if 'environments' not in definitions:
            definitions['environments'] = {}
        if environment not in definitions['environments']:
            definitions['environments'][environment] = {}
        definitions['environments'][environment][column] = mask_config
        click.echo(f"Added masking for column '{column}' in environment '{environment}' using {algorithm} algorithm")
    else:
        if 'global' not in definitions:
            definitions['global'] = {}
        definitions['global'][column] = mask_config
        click.echo(f"Added global masking for column '{column}' using {algorithm} algorithm")

    # Save the updated definitions
    if save_masking_definitions(definitions):
        click.echo("Masking definition saved successfully")
    else:
        click.echo("Failed to save masking definition")

@mask.command()
@click.option("-c", "--column", required=True, help="Column name to remove masking for")
@click.option("-e", "--environment", help="Environment name (if not specified, removes from global)")
def remove(column, environment):
    """Remove a masking definition for a column

    Examples:

        Remove global masking for a column:
          elm-tool mask remove --column password

        Remove environment-specific masking:
          elm-tool mask remove --column credit_card --environment prod
    """
    # Load existing definitions
    definitions = load_masking_definitions()

    # Remove from global or environment-specific definitions
    if environment:
        if ('environments' in definitions and
            environment in definitions['environments'] and
            column in definitions['environments'][environment]):
            del definitions['environments'][environment][column]
            click.echo(f"Removed masking for column '{column}' in environment '{environment}'")
        else:
            click.echo(f"No masking found for column '{column}' in environment '{environment}'")
    else:
        if 'global' in definitions and column in definitions['global']:
            del definitions['global'][column]
            click.echo(f"Removed global masking for column '{column}'")
        else:
            click.echo(f"No global masking found for column '{column}'")

    # Save the updated definitions
    if save_masking_definitions(definitions):
        click.echo("Masking definition updated successfully")
    else:
        click.echo("Failed to update masking definition")

@mask.command()
@click.option("-e", "--environment", help="Environment name (if not specified, shows all)")
def list(environment):
    """List masking definitions

    Examples:

        List all masking definitions:
          elm-tool mask list

        List environment-specific masking:
          elm-tool mask list --environment prod
    """
    # Load existing definitions
    definitions = load_masking_definitions()

    if environment:
        # Show environment-specific definitions
        if ('environments' in definitions and
            environment in definitions['environments'] and
            definitions['environments'][environment]):
            click.echo(f"Masking definitions for environment '{environment}':")
            for column, config in definitions['environments'][environment].items():
                algorithm = config.get('algorithm', 'star')
                params_str = ', '.join(f"{k}={v}" for k, v in config.get('params', {}).items())
                if params_str:
                    click.echo(f"  {column}: {algorithm} ({params_str})")
                else:
                    click.echo(f"  {column}: {algorithm}")
        else:
            click.echo(f"No masking definitions found for environment '{environment}'")
    else:
        # Show global definitions
        if 'global' in definitions and definitions['global']:
            click.echo("Global masking definitions:")
            for column, config in definitions['global'].items():
                algorithm = config.get('algorithm', 'star')
                params_str = ', '.join(f"{k}={v}" for k, v in config.get('params', {}).items())
                if params_str:
                    click.echo(f"  {column}: {algorithm} ({params_str})")
                else:
                    click.echo(f"  {column}: {algorithm}")
        else:
            click.echo("No global masking definitions found")

        # Show all environment-specific definitions
        if 'environments' in definitions and definitions['environments']:
            click.echo("\nEnvironment-specific masking definitions:")
            for env, env_defs in definitions['environments'].items():
                if env_defs:
                    click.echo(f"\n  Environment: {env}")
                    for column, config in env_defs.items():
                        algorithm = config.get('algorithm', 'star')
                        params_str = ', '.join(f"{k}={v}" for k, v in config.get('params', {}).items())
                        if params_str:
                            click.echo(f"    {column}: {algorithm} ({params_str})")
                        else:
                            click.echo(f"    {column}: {algorithm}")

@mask.command()
@click.option("-c", "--column", required=True, help="Column name to test masking for")
@click.option("-v", "--value", required=True, help="Value to test masking on")
@click.option("-e", "--environment", help="Environment name (if not specified, uses global)")
def test(column, value, environment):
    """Test masking on a sample value

    Examples:

        Test global masking for a column:
          elm-tool mask test --column password --value "secret123"

        Test environment-specific masking:
          elm-tool mask test --column credit_card --value "4111111111111111" --environment prod
    """
    # Load existing definitions
    definitions = load_masking_definitions()

    # Find the masking configuration
    mask_config = None
    if environment:
        if ('environments' in definitions and
            environment in definitions['environments'] and
            column in definitions['environments'][environment]):
            mask_config = definitions['environments'][environment][column]
            click.echo(f"Using environment '{environment}' masking for column '{column}'")

    if mask_config is None and 'global' in definitions and column in definitions['global']:
        mask_config = definitions['global'][column]
        click.echo(f"Using global masking for column '{column}'")

    if mask_config is None:
        click.echo(f"No masking definition found for column '{column}'")
        return

    # Apply masking
    algorithm = mask_config.get('algorithm', 'star')
    params = mask_config.get('params', {})

    if algorithm in MASKING_ALGORITHMS:
        masked_value = MASKING_ALGORITHMS[algorithm](value, **params)
        click.echo(f"Original value: {value}")
        click.echo(f"Masked value: {masked_value}")
    else:
        click.echo(f"Unknown masking algorithm: {algorithm}")



# Define command aliases
ALIASES = {
    "a": add,
    "rm": remove,
    "ls": list,
    "t": test
}