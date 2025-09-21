import os, sys, subprocess, venv
import configparser

# Core dependencies that are always required
CORE_PACKAGES = ["click", "platformdirs", "configparser", "sqlalchemy", "pandas", "cryptography"]

# Database-specific dependencies
DB_PACKAGES = {
    "ORACLE": ["oracledb"],
    "MYSQL": ["pymysql"],
    "MSSQL": ["pyodbc"],
    "POSTGRES": ["psycopg2-binary"]
}

def is_venv_active():
    return sys.prefix != sys.base_prefix

def create_and_activate_venv(VENV_DIR):
    """Create and activate a virtual environment, installing required dependencies"""
    # Check if virtual environment exists
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)

    # Install missing dependencies
    install_missing_dependencies(VENV_DIR)

def install_dependency(library_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])

def get_required_db_packages():
    """Determine which database packages are needed based on configured environments"""
    required_db_packages = []

    try:
        # Import variables here to avoid circular imports
        from elm_utils import variables

        # Read the environment configuration file
        config = configparser.ConfigParser()
        if os.path.exists(variables.ENVS_FILE):
            config.read(variables.ENVS_FILE)

            # Check each environment's database type
            for section in config.sections():
                if "type" in config[section]:
                    db_type = config[section]["type"].upper()
                    if db_type in DB_PACKAGES and DB_PACKAGES[db_type][0] not in required_db_packages:
                        required_db_packages.extend(DB_PACKAGES[db_type])
    except Exception as e:
        print(f"Warning: Could not read environment configuration: {str(e)}")
        # If we can't read the config, include all database packages as a fallback
        for packages in DB_PACKAGES.values():
            required_db_packages.extend(packages)

    # If no database packages were found, include all as a fallback
    if not required_db_packages:
        print("No database environments found. Installing all database drivers as a fallback.")
        for packages in DB_PACKAGES.values():
            required_db_packages.extend(packages)

    return required_db_packages

def install_missing_dependencies(VENV_DIR):
    """Install missing dependencies based on core requirements and configured database types"""
    # Get all required packages
    all_required_packages = CORE_PACKAGES.copy()
    all_required_packages.extend(get_required_db_packages())

    # Find missing packages
    missing_packages = [pkg for pkg in all_required_packages if not is_package_installed_in_venv(VENV_DIR, pkg)]
    venv_python = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")

    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}...")
        subprocess.check_call([venv_python, "-m", "pip", "install"] + missing_packages)

def is_package_installed_in_venv(venv_path, package_name):
    """Check if a package is installed in a specific virtual environment."""
    # Get the path to the Python executable in the virtual environment
    venv_python = os.path.join(venv_path, "Scripts" if os.name == "nt" else "bin", "python")

    # Use pip to check if the package is installed
    try:
        # Run pip list and check if the package is in the output
        result = subprocess.run(
            [venv_python, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        # Check if the package name is in the output
        return package_name.lower() in result.stdout.lower()
    except subprocess.CalledProcessError:
        # If pip list fails, fall back to the directory check method
        site_packages = os.path.join(venv_path, "Lib", "site-packages") if os.name == "nt" else \
                        os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")

        if not os.path.exists(site_packages):
            return False

        # Check for common package naming patterns
        for item in os.listdir(site_packages):
            item_lower = item.lower()
            if item_lower == package_name.lower() or \
               item_lower.startswith(f"{package_name.lower()}-") or \
               item_lower.startswith(f"{package_name.lower()}_"):
                return True

        return False