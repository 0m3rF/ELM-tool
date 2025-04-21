import pandas as pd
from elm.elm_utils.mask_algorithms import MASKING_ALGORITHMS

def apply_masking(data, environment=None, definitions=None):
    """Apply masking to a DataFrame based on masking definitions"""
    if not isinstance(data, pd.DataFrame):
        return data

    # If definitions are not provided, load them
    if definitions is None:
        from elm.elm_commands.mask import load_masking_definitions
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
