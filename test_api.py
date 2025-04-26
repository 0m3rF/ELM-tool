#!/usr/bin/env python
"""
Test script for the ELM Tool API
"""

import sys
import os

# Add the current directory to the path so we can import the elm package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the elm package
import elm

def main():
    """Test the ELM Tool API"""
    print("ELM Tool API Test")
    print(f"Version: {elm.__version__}")
    
    # List available functions
    print("\nAvailable API functions:")
    api_functions = [name for name in dir(elm) if not name.startswith('_') and callable(getattr(elm, name))]
    for func in api_functions:
        print(f"- {func}")
    
    # Generate some test data
    print("\nGenerating test data:")
    data = elm.generate_data(
        num_records=5,
        columns=["id", "name", "email"]
    )
    print(data)

    print("\nListing environments:")
    environments = elm.list_environments()
    for env in environments:
        print(env['name'])

if __name__ == "__main__":
    main()
