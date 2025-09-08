#!/usr/bin/env python3
"""
Version management script for ELM Tool
Usage: python version_manager.py [patch|minor|major]
"""

import re
import sys
from pathlib import Path

def get_current_version():
    """Get current version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)
    
    content = pyproject_path.read_text()
    version_match = re.search(r'version = "([^"]+)"', content)
    
    if not version_match:
        print("Error: Version not found in pyproject.toml")
        sys.exit(1)
    
    return version_match.group(1)

def parse_version(version_str):
    """Parse version string into major, minor, patch"""
    parts = version_str.split('.')
    if len(parts) != 3:
        print(f"Error: Invalid version format: {version_str}")
        sys.exit(1)
    
    try:
        return [int(part) for part in parts]
    except ValueError:
        print(f"Error: Invalid version format: {version_str}")
        sys.exit(1)

def bump_version(current_version, bump_type):
    """Bump version based on type"""
    major, minor, patch = parse_version(current_version)
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        print("Error: Invalid bump type. Use 'patch', 'minor', or 'major'")
        sys.exit(1)
    
    return f"{major}.{minor}.{patch}"

def update_pyproject_toml(new_version):
    """Update version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Update version
    content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)
    
    pyproject_path.write_text(content)
    print(f"Updated pyproject.toml with version {new_version}")

def update_init_py(new_version):
    """Update version in __init__.py if it exists"""
    init_path = Path("elm/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        
        # Check if __version__ exists and update it
        if '__version__' in content:
            content = re.sub(
                r'__version__ = "[^"]+"', 
                f'__version__ = "{new_version}"', 
                content
            )
        else:
            # Add __version__ at the top
            content = f'__version__ = "{new_version}"\n\n' + content
        
        init_path.write_text(content)
        print(f"Updated elm/__init__.py with version {new_version}")

def create_git_tag(version):
    """Create git tag for the new version"""
    import subprocess
    try:
        subprocess.run(['git', 'tag', f'v{version}'], check=True)
        print(f"Created git tag v{version}")
        print("Don't forget to push the tag: git push origin --tags")
    except subprocess.CalledProcessError:
        print("Warning: Could not create git tag (git not available or not a git repo)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python version_manager.py [patch|minor|major]")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    
    if bump_type not in ['patch', 'minor', 'major']:
        print("Error: Bump type must be 'patch', 'minor', or 'major'")
        sys.exit(1)
    
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"Current version: {current_version}")
    print(f"New version: {new_version}")
    
    # Confirm the change
    response = input(f"Do you want to bump version to {new_version}? (y/N): ")
    if response.lower() != 'y':
        print("Version bump cancelled")
        sys.exit(0)
    
    # Update files
    update_pyproject_toml(new_version)
    update_init_py(new_version)
    
    # Create git tag
    create_git_tag(new_version)
    
    print(f"\n✅ Version successfully bumped to {new_version}")
    print("\nNext steps:")
    print("1. Review changes: git diff")
    print("2. Commit changes: git add . && git commit -m 'Bump version to {}'".format(new_version))
    print("3. Push changes: git push")
    print("4. Push tags: git push origin --tags")
    print("5. Create GitHub release")
    print("6. Publish to PyPI: python -m build && python -m twine upload dist/*")

if __name__ == "__main__":
    main()