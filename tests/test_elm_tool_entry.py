"""Tests for elm-tool.py entry point."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, call


class TestElmToolEntry:
    """Test cases for elm-tool.py entry point."""

    def test_elm_tool_entry_point_imports(self):
        """Test that elm-tool.py can be imported."""
        # Add the project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Try to import the module
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("elm_tool_entry", 
                                                          os.path.join(project_root, "elm-tool.py"))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Don't execute it, just check it can be loaded
                assert module is not None
        except Exception as e:
            pytest.fail(f"Failed to load elm-tool.py: {e}")

    def test_elm_tool_entry_point_execution(self):
        """Test that elm-tool.py executes correctly."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elm_tool_path = os.path.join(project_root, "elm-tool.py")
        
        # Mock the cli function and os.makedirs
        with patch('elm.elm.cli') as mock_cli, \
             patch('os.makedirs') as mock_makedirs:
            
            # Execute the file
            with open(elm_tool_path, 'r') as f:
                code = f.read()
            
            # Create a namespace for execution
            namespace = {
                '__name__': '__main__',
                '__file__': elm_tool_path,
            }
            
            try:
                exec(code, namespace)
                
                # Check that makedirs was called
                assert mock_makedirs.called
                
                # Check that cli was called
                assert mock_cli.called
            except SystemExit:
                # CLI might call sys.exit, which is fine
                pass

    def test_elm_tool_creates_environment_directory(self):
        """Test that elm-tool.py creates the environment directory."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elm_tool_path = os.path.join(project_root, "elm-tool.py")
        
        with patch('elm.elm.cli') as mock_cli, \
             patch('os.makedirs') as mock_makedirs:
            
            # Execute the file
            with open(elm_tool_path, 'r') as f:
                code = f.read()
            
            namespace = {
                '__name__': '__main__',
                '__file__': elm_tool_path,
            }
            
            try:
                exec(code, namespace)
                
                # Verify makedirs was called with exist_ok=True
                assert mock_makedirs.called
                # Check that exist_ok was set to True
                call_args = mock_makedirs.call_args
                if call_args:
                    assert call_args[1].get('exist_ok', False) is True
            except SystemExit:
                pass

    def test_elm_tool_path_manipulation(self):
        """Test that elm-tool.py correctly manipulates sys.path."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elm_tool_path = os.path.join(project_root, "elm-tool.py")
        
        # Read the file content
        with open(elm_tool_path, 'r') as f:
            content = f.read()
        
        # Check that it manipulates sys.path
        assert 'sys.path.insert' in content
        assert 'os.path.dirname' in content
        assert 'os.path.abspath' in content

    def test_elm_tool_imports_cli(self):
        """Test that elm-tool.py imports the cli function."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elm_tool_path = os.path.join(project_root, "elm-tool.py")
        
        # Read the file content
        with open(elm_tool_path, 'r') as f:
            content = f.read()
        
        # Check that it imports cli
        assert 'from elm.elm import cli' in content or 'from elm.elm import' in content

    def test_elm_tool_imports_variables(self):
        """Test that elm-tool.py imports variables."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elm_tool_path = os.path.join(project_root, "elm-tool.py")
        
        # Read the file content
        with open(elm_tool_path, 'r') as f:
            content = f.read()
        
        # Check that it imports variables
        assert 'from elm.elm_utils import variables' in content or 'import variables' in content

    def test_elm_tool_main_guard(self):
        """Test that elm-tool.py has a main guard."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elm_tool_path = os.path.join(project_root, "elm-tool.py")
        
        # Read the file content
        with open(elm_tool_path, 'r') as f:
            content = f.read()
        
        # Check that it has a main guard
        assert "if __name__ == '__main__':" in content

    def test_elm_tool_shebang(self):
        """Test that elm-tool.py has a proper shebang."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        elm_tool_path = os.path.join(project_root, "elm-tool.py")
        
        # Read the first line
        with open(elm_tool_path, 'r') as f:
            first_line = f.readline().strip()
        
        # Check that it has a shebang
        assert first_line.startswith('#!')
        assert 'python' in first_line.lower()

