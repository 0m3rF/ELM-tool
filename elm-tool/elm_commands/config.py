import click

class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name].name
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)
    
@click.group(cls=AliasedGroup)
def config():
    """Configuration management commands"""
    print("config command")
    pass


@config.command()
def create():
    """Create a new system configuration"""
    # Implementation here
    print("create")

@config.command()
@click.option('--show-sensitive', is_flag=True, help='Show sensitive information')
def list(show_sensitive):
    """List all system configurations"""
    # Implementation here
    print("list")
    print(show_sensitive)


@config.command()
def delete():
    """Remove a system configuration"""
    # Implementation here
    print("delete")

@config.command()
def show():
    """Show a system configuration"""
    # Implementation here
    print("show")

@config.command()
def update():
    """Update a system configuration"""
    # Implementation here
    print("update")

@config.command()
def test():
    """Test a system configuration"""
    # Implementation here
    print("test")

ALIASES = {
    "new": create,
    "ls": list,
    "rm": delete,
    "remove": delete,
    "inspect": show,
    "edit": update,
    "validate": test
}
