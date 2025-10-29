"""
Virtual Environment Management for ELM Tool.

This module handles virtual environment creation, validation, and dependency
installation with configuration tracking to ensure the venv is properly
initialized before running operations.
"""

import os
import sys
import venv
import subprocess
import configparser
from typing import List, Optional

from elm.core.types import OperationResult
from elm.core.utils import create_success_result, create_error_result


# Core dependencies that are always required
CORE_PACKAGES = [
    "click",
    "platformdirs",
    "configparser",
    "sqlalchemy",
    "pandas",
    "cryptography",
    "faker"
]

# Database-specific dependencies
DB_PACKAGES = {
    "ORACLE": ["oracledb"],
    "MYSQL": ["pymysql"],
    "MSSQL": ["pyodbc"],
    "POSTGRES": ["psycopg2-binary"]
}


def ensure_venv_ready(venv_dir: str, config_manager) -> OperationResult:
    """
    Ensure virtual environment is ready for use.
    
    This function checks if the venv exists and is initialized. If not,
    it creates the venv, installs dependencies, and updates the config.
    
    Args:
        venv_dir: Path to the virtual environment directory
        config_manager: ConfigManager instance for tracking venv status
    
    Returns:
        OperationResult indicating success or failure
    """
    try:
        # Check if venv is already initialized according to config
        if config_manager.is_venv_initialized() and config_manager.check_venv_exists():
            # Venv is marked as initialized and exists, just verify dependencies
            missing = get_missing_packages(venv_dir)
            if missing:
                print(f"📦 Installing {len(missing)} missing package(s)...")
                result = install_packages(venv_dir, missing)
                if not result.success:
                    return result
            return create_success_result("Virtual environment is ready")
        
        # Venv needs to be created or re-initialized
        if not config_manager.check_venv_exists():
            print(f"\n🔧 Creating virtual environment in {venv_dir}")
            print("   This is a one-time setup process...")
            result = create_venv(venv_dir)
            if not result.success:
                return result
        
        # Install all required dependencies
        print("\n📦 Installing required dependencies...")
        print("   This may take a few minutes on first run...")
        
        all_packages = get_all_required_packages(config_manager)
        result = install_packages(venv_dir, all_packages)
        if not result.success:
            return result
        
        # Mark venv as initialized in config
        config_manager.mark_venv_initialized(True)
        
        print("\n✅ Virtual environment setup complete!")
        return create_success_result("Virtual environment created and initialized successfully")
        
    except Exception as e:
        return create_error_result(f"Failed to ensure venv ready: {str(e)}")


def create_venv(venv_dir: str) -> OperationResult:
    """
    Create a new virtual environment.
    
    Args:
        venv_dir: Path where the virtual environment should be created
    
    Returns:
        OperationResult indicating success or failure
    """
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(venv_dir), exist_ok=True)
        
        # Create the virtual environment
        venv.create(venv_dir, with_pip=True)
        
        return create_success_result(f"Virtual environment created at {venv_dir}")
    except Exception as e:
        return create_error_result(f"Failed to create virtual environment: {str(e)}")


def get_all_required_packages(config_manager) -> List[str]:
    """
    Get all required packages including core and database-specific packages.
    
    Args:
        config_manager: ConfigManager instance to access environment file
    
    Returns:
        List of package names to install
    """
    all_packages = CORE_PACKAGES.copy()
    
    # Add database-specific packages based on configured environments
    db_packages = get_required_db_packages(config_manager)
    all_packages.extend(db_packages)
    
    return all_packages


def get_required_db_packages(config_manager) -> List[str]:
    """
    Determine which database packages are needed based on configured environments.
    
    Args:
        config_manager: ConfigManager instance to access environment file
    
    Returns:
        List of database-specific package names
    """
    required_db_packages = []
    
    try:
        envs_file = config_manager.get_envs_file()
        
        if os.path.exists(envs_file):
            config = configparser.ConfigParser()
            config.read(envs_file)
            
            # Check each environment's database type
            for section in config.sections():
                if "type" in config[section]:
                    db_type = config[section]["type"].upper()
                    if db_type in DB_PACKAGES:
                        for pkg in DB_PACKAGES[db_type]:
                            if pkg not in required_db_packages:
                                required_db_packages.append(pkg)
    except Exception as e:
        print(f"⚠️  Warning: Could not read environment configuration: {str(e)}")
        # If we can't read the config, include all database packages as a fallback
        for packages in DB_PACKAGES.values():
            for pkg in packages:
                if pkg not in required_db_packages:
                    required_db_packages.append(pkg)
    
    # If no database packages were found, include all as a fallback
    if not required_db_packages:
        print("ℹ️  No database environments found. Installing all database drivers.")
        for packages in DB_PACKAGES.values():
            required_db_packages.extend(packages)
    
    return required_db_packages


def install_packages(venv_dir: str, packages: List[str]) -> OperationResult:
    """
    Install packages in the virtual environment.
    
    Args:
        venv_dir: Path to the virtual environment
        packages: List of package names to install
    
    Returns:
        OperationResult indicating success or failure
    """
    if not packages:
        return create_success_result("No packages to install")
    
    try:
        # Get the path to the Python executable in the venv
        if os.name == "nt":  # Windows
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:  # Unix-like
            venv_python = os.path.join(venv_dir, "bin", "python")
        
        if not os.path.exists(venv_python):
            return create_error_result(f"Virtual environment Python not found at {venv_python}")
        
        # Install packages
        print(f"   Installing: {', '.join(packages)}")
        subprocess.check_call(
            [venv_python, "-m", "pip", "install", "--quiet"] + packages,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        return create_success_result(f"Successfully installed {len(packages)} package(s)")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return create_error_result(f"Failed to install packages: {error_msg}")
    except Exception as e:
        return create_error_result(f"Error installing packages: {str(e)}")


def get_missing_packages(venv_dir: str) -> List[str]:
    """
    Get list of packages that are not installed in the venv.
    
    Args:
        venv_dir: Path to the virtual environment
    
    Returns:
        List of missing package names
    """
    # For now, we'll do a simple check - in production you might want more sophisticated checking
    # This is a lightweight check that doesn't require importing from the venv
    try:
        if os.name == "nt":  # Windows
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:  # Unix-like
            venv_python = os.path.join(venv_dir, "bin", "python")
        
        if not os.path.exists(venv_python):
            return []
        
        # Get list of installed packages
        result = subprocess.run(
            [venv_python, "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        
        installed = set()
        for line in result.stdout.split('\n'):
            if '==' in line:
                pkg_name = line.split('==')[0].lower()
                installed.add(pkg_name)
        
        # Check which core packages are missing
        missing = []
        for pkg in CORE_PACKAGES:
            if pkg.lower() not in installed:
                missing.append(pkg)
        
        return missing
    except Exception:
        # If we can't check, assume nothing is missing to avoid unnecessary reinstalls
        return []


def is_package_installed(venv_dir: str, package_name: str) -> bool:
    """
    Check if a specific package is installed in the virtual environment.
    
    Args:
        venv_dir: Path to the virtual environment
        package_name: Name of the package to check
    
    Returns:
        True if package is installed, False otherwise
    """
    try:
        if os.name == "nt":  # Windows
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:  # Unix-like
            venv_python = os.path.join(venv_dir, "bin", "python")
        
        if not os.path.exists(venv_python):
            return False
        
        # Use pip to check if the package is installed
        result = subprocess.run(
            [venv_python, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0
    except Exception:
        return False

