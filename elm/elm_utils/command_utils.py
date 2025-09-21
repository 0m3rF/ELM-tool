"""
Command utilities for ELM Tool CLI.

This module provides shared utilities for Click command groups and commands,
including the AliasedGroup class that supports command aliases.
"""

import click


class AliasedGroup(click.Group):
    """Click group that supports command aliases.
    
    This class extends click.Group to support command aliases by looking up
    command names in an ALIASES dictionary. If an alias is found, it resolves
    to the actual command name before calling the parent get_command method.
    
    Usage:
        @click.group(cls=AliasedGroup)
        def my_command():
            pass
        
        # Define aliases at the module level
        ALIASES = {
            'ls': list_command,
            'rm': remove_command,
        }
    """
    
    def get_command(self, ctx, cmd_name):
        """Get command by name, resolving aliases if present.

        Args:
            ctx: Click context
            cmd_name: Command name or alias to resolve

        Returns:
            The resolved command object, or None if not found
        """
        # Get the module where this group was defined by looking at the callback
        import inspect

        # Try to get ALIASES from the module where this group's callback is defined
        try:
            if hasattr(self, 'callback') and self.callback:
                callback_module = inspect.getmodule(self.callback)
                if callback_module and hasattr(callback_module, 'ALIASES'):
                    aliases = callback_module.ALIASES
                    try:
                        cmd_name = aliases[cmd_name].name
                    except (KeyError, AttributeError):
                        pass
        except Exception:
            # If anything goes wrong, just continue with the original cmd_name
            pass

        # cmd_name should be a string at this point, but add safety check
        if cmd_name is None:
            return None

        return super().get_command(ctx, cmd_name)
