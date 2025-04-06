import os, sys, subprocess, venv
import importlib.util

REQUIRED_PACKAGES = ["click", "platformdirs"]  # Forced dependencies

def is_venv_active():
    return sys.prefix != sys.base_prefix

def create_and_activate_venv(VENV_DIR):
    #print(f"Checking virtual environment in {VENV_DIR}")
    venv_python = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}")
        venv.create(VENV_DIR, with_pip=True)
    if not is_venv_active():
        print(f"Re-running script inside virtual environment: {venv_python}")
    
    install_missing_dependencies(VENV_DIR)
    #print(venv_python)
    #os.execv(venv_python, [venv_python] + sys.argv)

def install_dependency(library_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", library_name])

def install_missing_dependencies(VENV_DIR):
    missing_packages = [pkg for pkg in REQUIRED_PACKAGES if not is_package_installed_in_venv(VENV_DIR ,pkg)]
    venv_python = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}...")
        subprocess.check_call([venv_python, "-m", "pip", "install"] + missing_packages)

def is_package_installed_in_venv(venv_path, package_name):
    """Check if a package is installed in a specific virtual environment."""
    site_packages = os.path.join(venv_path, "Lib", "site-packages") if os.name == "nt" else \
                    os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    
    package_installed = any(pkg.startswith(package_name) for pkg in os.listdir(site_packages)) if os.path.exists(site_packages) else False
    return package_installed