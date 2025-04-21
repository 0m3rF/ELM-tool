#!/usr/bin/env python
import sys
import os

# Add the current directory to the path so we can import the elm package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the cli function from the elm package
from elm.elm import cli

if __name__ == '__main__':
    # Ensure the environment directory exists
    from elm.elm_utils import variables
    os.makedirs(os.path.dirname(variables.ENVS_FILE), exist_ok=True)

    # Run the CLI
    cli()
