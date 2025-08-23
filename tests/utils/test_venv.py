"""
Tests for the virtual environment utilities.
"""
import pytest
import os
import sys
import configparser
import subprocess
from unittest.mock import patch, MagicMock

from elm.elm_utils.venv import (
    is_venv_active,
    create_and_activate_venv,
    install_dependency,
    get_required_db_packages,
    install_missing_dependencies,
    is_package_installed_in_venv,
    CORE_PACKAGES,
    DB_PACKAGES
)


def test_is_venv_active():
    """Test checking if a virtual environment is active."""
    # Mock sys.prefix and sys.base_prefix
    with patch('sys.prefix', 'venv/path'), \
         patch('sys.base_prefix', 'base/path'):
        # When prefix != base_prefix, venv is active
        assert is_venv_active() is True

    with patch('sys.prefix', 'same/path'), \
         patch('sys.base_prefix', 'same/path'):
        # When prefix == base_prefix, venv is not active
        assert is_venv_active() is False


def test_create_and_activate_venv():
    """Test creating and activating a virtual environment."""
    with patch('os.path.exists') as mock_exists, \
         patch('venv.create') as mock_venv_create, \
         patch('elm.elm_utils.venv.install_missing_dependencies') as mock_install:

        # Test when venv doesn't exist
        mock_exists.return_value = False

        create_and_activate_venv('/path/to/venv')

        # Verify venv was created and dependencies were installed
        mock_venv_create.assert_called_once_with('/path/to/venv', with_pip=True)
        mock_install.assert_called_once_with('/path/to/venv')

        # Test when venv already exists
        mock_exists.return_value = True
        mock_venv_create.reset_mock()
        mock_install.reset_mock()

        create_and_activate_venv('/path/to/venv')

        # Verify venv was not created but dependencies were still installed
        mock_venv_create.assert_not_called()
        mock_install.assert_called_once_with('/path/to/venv')


def test_install_dependency():
    """Test installing a dependency."""
    with patch('subprocess.check_call') as mock_check_call, \
         patch('sys.executable', '/path/to/python'):

        install_dependency('test-package')

        # Verify pip install was called with the correct arguments
        mock_check_call.assert_called_once_with([
            '/path/to/python', '-m', 'pip', 'install', 'test-package'
        ])


def test_get_required_db_packages():
    """Test getting required database packages."""
    # Test with existing environments
    with patch('os.path.exists') as mock_exists, \
         patch('configparser.ConfigParser.read') as mock_read, \
         patch('configparser.ConfigParser.sections') as mock_sections, \
         patch('configparser.ConfigParser.__getitem__') as mock_getitem:

        mock_exists.return_value = True
        mock_sections.return_value = ['env1', 'env2']

        # Mock the __getitem__ to return environment configs
        def getitem_side_effect(key):
            if key == 'env1':
                return {'type': 'postgres'}
            elif key == 'env2':
                return {'type': 'oracle'}
            else:
                raise KeyError(key)

        mock_getitem.side_effect = getitem_side_effect

        # Get required packages
        packages = get_required_db_packages()

        # Verify the correct packages were returned
        assert 'psycopg2-binary' in packages
        assert 'cx_oracle' in packages

    # Test with no environments (should include all DB packages)
    with patch('os.path.exists') as mock_exists, \
         patch('configparser.ConfigParser.read') as mock_read, \
         patch('configparser.ConfigParser.sections') as mock_sections:

        mock_exists.return_value = True
        mock_sections.return_value = []

        # Get required packages
        packages = get_required_db_packages()

        # Verify all DB packages were included
        for db_packages in DB_PACKAGES.values():
            for pkg in db_packages:
                assert pkg in packages

    # Test with exception during config reading
    with patch('os.path.exists') as mock_exists, \
         patch('configparser.ConfigParser.read') as mock_read:

        mock_exists.return_value = True
        mock_read.side_effect = Exception("Config error")

        # Get required packages
        packages = get_required_db_packages()

        # Verify all DB packages were included as fallback
        for db_packages in DB_PACKAGES.values():
            for pkg in db_packages:
                assert pkg in packages


def test_install_missing_dependencies():
    """Test installing missing dependencies."""
    with patch('elm.elm_utils.venv.get_required_db_packages') as mock_get_db_packages, \
         patch('elm.elm_utils.venv.is_package_installed_in_venv') as mock_is_installed, \
         patch('subprocess.check_call') as mock_check_call, \
         patch('os.path.join') as mock_join, \
         patch('os.name', 'nt'):  # Simulate Windows

        # Mock required packages
        mock_get_db_packages.return_value = ['psycopg2-binary']

        # Mock package installation status
        def is_installed_side_effect(venv_path, package):
            # Simulate some packages are installed, others are not
            return package not in ['click', 'psycopg2-binary']

        mock_is_installed.side_effect = is_installed_side_effect

        # Mock venv Python path
        mock_join.return_value = '/path/to/venv/Scripts/python'

        # Install missing dependencies
        install_missing_dependencies('/path/to/venv')

        # Verify check_call was called with the correct arguments
        mock_check_call.assert_called_once()
        args = mock_check_call.call_args[0][0]
        assert args[0] == '/path/to/venv/Scripts/python'
        assert args[1:4] == ['-m', 'pip', 'install']
        assert 'click' in args[4:]
        assert 'psycopg2-binary' in args[4:]


def test_is_package_installed_in_venv():
    """Test checking if a package is installed in a virtual environment."""
    # Test with pip list success
    with patch('subprocess.run') as mock_run, \
         patch('os.path.join') as mock_join, \
         patch('os.name', 'nt'):  # Simulate Windows

        # Mock subprocess.run result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "Package    Version\n---------  -------\nclick      8.1.3\npandas     1.5.3"
        mock_run.return_value = mock_process

        # Mock venv Python path
        mock_join.return_value = '/path/to/venv/Scripts/python'

        # Check if package is installed
        result = is_package_installed_in_venv('/path/to/venv', 'click')
        assert result is True

        # Check if package is not installed
        result = is_package_installed_in_venv('/path/to/venv', 'flask')
        assert result is False


def test_get_required_db_packages_file_not_exists():
    """Test getting required database packages when config file doesn't exist."""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False

        packages = get_required_db_packages()

        # Should include all DB packages as fallback
        for db_packages in DB_PACKAGES.values():
            for pkg in db_packages:
                assert pkg in packages


def test_get_required_db_packages_with_duplicate_types():
    """Test getting required database packages with duplicate database types."""
    with patch('os.path.exists') as mock_exists, \
         patch('configparser.ConfigParser.read') as mock_read, \
         patch('configparser.ConfigParser.sections') as mock_sections, \
         patch('configparser.ConfigParser.__getitem__') as mock_getitem:

        mock_exists.return_value = True
        mock_sections.return_value = ['env1', 'env2', 'env3']

        # Mock the __getitem__ to return environment configs with duplicates
        def getitem_side_effect(key):
            if key == 'env1':
                return {'type': 'postgres'}
            elif key == 'env2':
                return {'type': 'postgres'}  # Duplicate
            elif key == 'env3':
                return {'type': 'mysql'}
            else:
                raise KeyError(key)

        mock_getitem.side_effect = getitem_side_effect

        # Get required packages
        packages = get_required_db_packages()

        # Verify packages are not duplicated
        assert packages.count('psycopg2-binary') == 1
        assert 'pymysql' in packages


def test_get_required_db_packages_environment_without_type():
    """Test getting required database packages when environment has no type."""
    with patch('os.path.exists') as mock_exists, \
         patch('configparser.ConfigParser.read') as mock_read, \
         patch('configparser.ConfigParser.sections') as mock_sections, \
         patch('configparser.ConfigParser.__getitem__') as mock_getitem:

        mock_exists.return_value = True
        mock_sections.return_value = ['env1', 'env2']

        # Mock the __getitem__ to return environment configs
        def getitem_side_effect(key):
            if key == 'env1':
                return {'host': 'localhost'}  # No type field
            elif key == 'env2':
                return {'type': 'postgres'}
            else:
                raise KeyError(key)

        mock_getitem.side_effect = getitem_side_effect

        # Get required packages
        packages = get_required_db_packages()

        # Should only include packages for env2
        assert 'psycopg2-binary' in packages


def test_get_required_db_packages_unknown_db_type():
    """Test getting required database packages with unknown database type."""
    with patch('os.path.exists') as mock_exists, \
         patch('configparser.ConfigParser.read') as mock_read, \
         patch('configparser.ConfigParser.sections') as mock_sections, \
         patch('configparser.ConfigParser.__getitem__') as mock_getitem:

        mock_exists.return_value = True
        mock_sections.return_value = ['env1']

        # Mock the __getitem__ to return environment configs
        def getitem_side_effect(key):
            if key == 'env1':
                return {'type': 'UNKNOWN_DB'}  # Unknown type
            else:
                raise KeyError(key)

        mock_getitem.side_effect = getitem_side_effect

        # Get required packages
        packages = get_required_db_packages()

        # Should include all DB packages as fallback since no valid types found
        for db_packages in DB_PACKAGES.values():
            for pkg in db_packages:
                assert pkg in packages


def test_install_missing_dependencies_no_missing_packages():
    """Test installing missing dependencies when all packages are already installed."""
    with patch('elm.elm_utils.venv.get_required_db_packages') as mock_get_db_packages, \
         patch('elm.elm_utils.venv.is_package_installed_in_venv') as mock_is_installed, \
         patch('subprocess.check_call') as mock_check_call:

        # Mock required packages
        mock_get_db_packages.return_value = ['psycopg2-binary']

        # Mock all packages as installed
        mock_is_installed.return_value = True

        # Install missing dependencies
        install_missing_dependencies('/path/to/venv')

        # Verify check_call was not called since no packages are missing
        mock_check_call.assert_not_called()


def test_is_package_installed_in_venv_pip_failure_fallback():
    """Test package installation check with pip failure fallback."""
    with patch('subprocess.run') as mock_run, \
         patch('os.path.join') as mock_join, \
         patch('os.path.exists') as mock_exists, \
         patch('os.listdir') as mock_listdir, \
         patch('os.name', 'nt'), \
         patch('sys.version_info', MagicMock(major=3, minor=12)):

        # Mock subprocess.run to raise CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(1, 'pip list')

        # Mock venv Python path
        mock_join.side_effect = lambda *args: '/'.join(args)

        # Mock site-packages directory exists
        mock_exists.return_value = True

        # Mock directory listing
        mock_listdir.return_value = ['click-8.1.3.dist-info', 'pandas', 'other_package-1.0']

        # Check if package is installed (should find click)
        result = is_package_installed_in_venv('/path/to/venv', 'click')
        assert result is True

        # Check if package is not installed
        result = is_package_installed_in_venv('/path/to/venv', 'flask')
        assert result is False


def test_is_package_installed_in_venv_no_site_packages():
    """Test package installation check when site-packages doesn't exist."""
    with patch('subprocess.run') as mock_run, \
         patch('os.path.join') as mock_join, \
         patch('os.path.exists') as mock_exists, \
         patch('os.name', 'nt'):

        # Mock subprocess.run to raise CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(1, 'pip list')

        # Mock venv Python path
        mock_join.side_effect = lambda *args: '/'.join(args)

        # Mock site-packages directory doesn't exist
        mock_exists.return_value = False

        # Check if package is installed
        result = is_package_installed_in_venv('/path/to/venv', 'click')
        assert result is False


def test_is_package_installed_in_venv_unix_paths():
    """Test package installation check on Unix systems."""
    with patch('subprocess.run') as mock_run, \
         patch('os.path.join') as mock_join, \
         patch('os.path.exists') as mock_exists, \
         patch('os.listdir') as mock_listdir, \
         patch('os.name', 'posix'), \
         patch('sys.version_info', MagicMock(major=3, minor=12)):

        # Mock subprocess.run to raise CalledProcessError
        mock_run.side_effect = subprocess.CalledProcessError(1, 'pip list')

        # Mock venv Python path
        mock_join.side_effect = lambda *args: '/'.join(args)

        # Mock site-packages directory exists
        mock_exists.return_value = True

        # Mock directory listing with underscore package name
        mock_listdir.return_value = ['click_package', 'pandas', 'other_package-1.0']

        # Check if package is installed (should find click_package)
        result = is_package_installed_in_venv('/path/to/venv', 'click_package')
        assert result is True


def test_install_dependency_failure():
    """Test installing a dependency with failure."""
    with patch('subprocess.check_call') as mock_check_call, \
         patch('sys.executable', '/path/to/python'):

        # Mock subprocess failure
        mock_check_call.side_effect = subprocess.CalledProcessError(1, 'pip install')

        # Should raise the exception
        with pytest.raises(subprocess.CalledProcessError):
            install_dependency('test-package')


def test_create_and_activate_venv_with_print():
    """Test creating virtual environment with print output."""
    with patch('os.path.exists') as mock_exists, \
         patch('venv.create') as mock_venv_create, \
         patch('elm.elm_utils.venv.install_missing_dependencies') as mock_install, \
         patch('builtins.print') as mock_print:

        # Test when venv doesn't exist
        mock_exists.return_value = False

        create_and_activate_venv('/path/to/venv')

        # Verify print was called
        mock_print.assert_called_once_with("Creating virtual environment in /path/to/venv")


def test_install_missing_dependencies_with_print():
    """Test installing missing dependencies with print output."""
    with patch('elm.elm_utils.venv.get_required_db_packages') as mock_get_db_packages, \
         patch('elm.elm_utils.venv.is_package_installed_in_venv') as mock_is_installed, \
         patch('subprocess.check_call') as mock_check_call, \
         patch('os.path.join') as mock_join, \
         patch('os.name', 'posix'), \
         patch('builtins.print') as mock_print:

        # Mock required packages
        mock_get_db_packages.return_value = ['psycopg2-binary']

        # Mock package installation status
        mock_is_installed.return_value = False

        # Mock venv Python path
        mock_join.return_value = '/path/to/venv/bin/python'

        # Install missing dependencies
        install_missing_dependencies('/path/to/venv')

        # Verify print was called
        mock_print.assert_called_once()
        print_call_args = mock_print.call_args[0][0]
        assert "Installing missing packages:" in print_call_args
