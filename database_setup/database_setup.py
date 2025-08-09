#!/usr/bin/env python3
"""
Database Setup Tool - Standalone script for setting up database containers for development and testing.
"""

import click
import sys
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from TestDatabaseConfigs import DatabaseConfigs
from elm.models import ContainerResult, ContainerState, DatabaseConfig
from DockerManager import DockerManager

class DatabaseOrchestrator:
    """Orchestrates database container startup with proper dependency management."""

    def __init__(self, docker_manager: DockerManager):
        self.docker_manager = docker_manager
        self.results = {}

    def setup_databases(self, configs: Dict[str, DatabaseConfig], parallel: bool = True) -> Dict[str, ContainerResult]:
        """Setup all databases with proper ordering and dependency management."""
        click.echo("\n" + "="*60)
        click.echo("🚀 STARTING DATABASE CONTAINER ORCHESTRATION")
        click.echo("="*60)

        # Sort configs by startup priority
        sorted_configs = sorted(configs.items(), key=lambda x: x[1].startup_priority)

        if parallel:
            return self._setup_parallel(sorted_configs)
        else:
            return self._setup_sequential(sorted_configs)

    def _setup_sequential(self, sorted_configs: List[Tuple[str, DatabaseConfig]]) -> Dict[str, ContainerResult]:
        """Setup databases sequentially based on priority."""
        results = {}

        for db_name, config in sorted_configs:
            click.echo(f"\n📋 Processing {db_name} (priority: {config.startup_priority})")
            result = self.docker_manager.create_container(config)
            results[db_name] = result

            if not result.success:
                click.echo(f"❌ {db_name} setup failed: {result.message}")
                # Continue with other databases even if one fails
            else:
                click.echo(f"✅ {db_name} setup completed in {result.startup_time:.2f}s")

        return results

    def _setup_parallel(self, sorted_configs: List[Tuple[str, DatabaseConfig]]) -> Dict[str, ContainerResult]:
        """Setup databases in parallel, respecting priorities and dependencies."""
        results = {}

        # Group by priority
        priority_groups = {}
        for db_name, config in sorted_configs:
            priority = config.startup_priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append((db_name, config))

        # Process each priority group
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]
            click.echo(f"\n🔄 Starting priority group {priority} ({len(group)} databases)")

            # Start all databases in this priority group in parallel
            with ThreadPoolExecutor(max_workers=len(group)) as executor:
                future_to_db = {
                    executor.submit(self.docker_manager.create_container, config): db_name
                    for db_name, config in group
                }

                for future in as_completed(future_to_db):
                    db_name = future_to_db[future]
                    try:
                        result = future.result()
                        results[db_name] = result

                        if result.success:
                            click.echo(f"✅ {db_name} completed in {result.startup_time:.2f}s")
                        else:
                            click.echo(f"❌ {db_name} failed: {result.message}")
                    except Exception as e:
                        click.echo(f"❌ {db_name} exception: {str(e)}")
                        results[db_name] = ContainerResult(
                            name=db_name,
                            success=False,
                            state=ContainerState.UNHEALTHY,
                            message=f"Exception: {str(e)}"
                        )

        return results

# Utility functions
def print_connection_strings():
    """Print connection strings for all databases."""
    click.echo("\n" + "="*60)
    click.echo("DATABASE CONNECTION STRINGS")
    click.echo("="*60)

    click.echo("\nPostgreSQL:")
    click.echo("  postgresql://ELM_TOOL_user:ELM_TOOL_password@localhost:5433/ELM_TOOL_db")

    click.echo("\nMySQL:")
    click.echo("  mysql://ELM_TOOL_user:ELM_TOOL_password@localhost:3307/ELM_TOOL_db")

    click.echo("\nOracle:")
    click.echo("  oracle://ELM_TOOL_user:ELM_TOOL_password@localhost:1522/XE")

    click.echo("\nMSSQL:")
    click.echo("  mssql://sa:ELM_TOOL_Password123!@localhost:1434")
    click.echo("  Note: Use TrustServerCertificate=true in connection string")
    click.echo("  Example: Server=localhost,1434;Database=master;User Id=sa;Password=ELM_TOOL_Password123!;TrustServerCertificate=true;")

def print_summary(results: Dict[str, ContainerResult]):
    """Print setup summary."""
    click.echo("\n" + "="*60)
    click.echo("📊 DATABASE SETUP SUMMARY")
    click.echo("="*60)

    successful = 0
    failed = 0

    for db_name, result in results.items():
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        click.echo(f"{db_name:12} | {status:10} | {result.startup_time:6.2f}s | {result.message}")

        if result.success:
            successful += 1
        else:
            failed += 1

    click.echo(f"\nTotal: {successful} successful, {failed} failed")
    return successful, failed



# CLI Commands
@click.group()
@click.option('--timeout', default=600, help='Default timeout for Docker operations in seconds')
@click.pass_context
def cli(ctx, timeout):
    """Database Setup Tool - Manage database containers for development."""
    ctx.ensure_object(dict)
    ctx.obj['timeout'] = timeout


@cli.command()
@click.pass_context
def check_docker(ctx):
    """Check if Docker is installed and running."""
    docker_manager = DockerManager(timeout=ctx.obj['timeout'])

    click.echo("🔍 Checking Docker installation...")
    is_available, message = docker_manager.check_docker_installation()

    if is_available:
        click.echo(f"✅ {message}")

        # Show Docker version info
        success, stdout, _ = docker_manager.run_command(["docker", "--version"], timeout=5)
        if success:
            click.echo(f"📋 {stdout.strip()}")

        # Show Docker info
        success, stdout, _ = docker_manager.run_command(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=10)
        if success:
            click.echo(f"📋 Docker Engine version: {stdout.strip()}")
    else:
        click.echo(f"❌ {message}")
        sys.exit(1)


@cli.command()
@click.option('--databases', '-d', multiple=True,
              type=click.Choice(['postgresql', 'mysql', 'mssql', 'oracle'], case_sensitive=False),
              help='Specific databases to setup (default: all)')
@click.option('--parallel/--sequential', default=True,
              help='Run database setup in parallel or sequential mode')
@click.option('--pull/--no-pull', default=False,
              help='Pull latest images before starting containers')
@click.option('--remove-existing/--keep-existing', default=False,
              help='Remove existing containers before creating new ones')
@click.pass_context
def setup(ctx, databases, parallel, pull, remove_existing):
    """Setup database containers."""
    docker_manager = DockerManager(timeout=ctx.obj['timeout'])

    # Check Docker first
    click.echo("🔍 Checking Docker availability...")
    is_available, message = docker_manager.check_docker_installation()
    if not is_available:
        click.echo(f"❌ {message}")
        sys.exit(1)

    click.echo(f"✅ {message}")

    # Get configurations
    all_configs = DatabaseConfigs.get_configs()

    # Filter databases if specified
    if databases:
        configs = {db: all_configs[db] for db in databases if db in all_configs}
        if not configs:
            click.echo("❌ No valid databases specified")
            sys.exit(1)
    else:
        configs = all_configs

    click.echo(f"\n📋 Selected databases: {', '.join(configs.keys())}")

    # Remove existing containers if requested
    if remove_existing:
        remove_container(docker_manager, configs)

    # Pull images if requested
    if pull:
        pull_container(docker_manager, configs)

    # Setup databases
    orchestrator = DatabaseOrchestrator(docker_manager)
    results = orchestrator.setup_databases(configs, parallel=parallel)

    # Print summary
    successful, failed = print_summary(results)

    if successful > 0:
        print_connection_strings()

    if failed > 0:
        sys.exit(1)

@cli.command()
@click.option('--databases', '-d', multiple=True,
              type=click.Choice(['postgresql', 'mysql', 'mssql', 'oracle'], case_sensitive=False),
              help='Specific databases to check (default: all)')
@click.pass_context
def status(ctx, databases):
    """Check status of database containers."""
    docker_manager = DockerManager(timeout=ctx.obj['timeout'])

    # Get configurations
    all_configs = DatabaseConfigs.get_configs()

    # Filter databases if specified
    if databases:
        configs = {db: all_configs[db] for db in databases if db in all_configs}
    else:
        configs = all_configs

    click.echo("\n📊 DATABASE CONTAINER STATUS")
    click.echo("="*60)

    for db_name, config in configs.items():
        state = docker_manager.get_container_state(config.name)

        if state == ContainerState.NOT_EXISTS:
            status_icon = "❌"
            status_text = "NOT EXISTS"
        elif state == ContainerState.STOPPED:
            status_icon = "⏸️ "
            status_text = "STOPPED"
        elif state == ContainerState.RUNNING:
            status_icon = "✅"
            status_text = "RUNNING"
        else:
            status_icon = "⚠️ "
            status_text = str(state.value).upper()

        click.echo(f"{db_name:12} | {status_icon} {status_text:12} | {config.name}")


@cli.command()
@click.option('--databases', '-d', multiple=True,
              type=click.Choice(['postgresql', 'mysql', 'mssql', 'oracle'], case_sensitive=False),
              help='Specific databases to remove (default: all)')
@click.option('--force', '-f', is_flag=True, help='Force removal without confirmation')
@click.pass_context
def remove(ctx, databases, force):
    """Remove database containers."""
    docker_manager = DockerManager(timeout=ctx.obj['timeout'])

    # Get configurations
    all_configs = DatabaseConfigs.get_configs()

    # Filter databases if specified
    if databases:
        configs = {db: all_configs[db] for db in databases if db in all_configs}
    else:
        configs = all_configs

    if not force:
        click.echo("⚠️  This will remove the following containers:")
        for db_name, config in configs.items():
            click.echo(f"  - {config.name}")

        if not click.confirm("Are you sure you want to continue?"):
            click.echo("Operation cancelled.")
            return

    click.echo("\n🗑️  Removing containers...")
    for db_name, config in configs.items():
        state = docker_manager.get_container_state(config.name)
        if state != ContainerState.NOT_EXISTS:
            click.echo(f"🗑️  Removing {config.name}")
            if docker_manager.remove_container(config.name):
                click.echo(f"✅ Successfully removed {config.name}")
            else:
                click.echo(f"❌ Failed to remove {config.name}")
        else:
            click.echo(f"ℹ️  {config.name} does not exist")


@cli.command()
def connections():
    """Show database connection strings."""
    print_connection_strings()


@cli.command()
@click.argument('database', type=click.Choice(['mssql'], case_sensitive=False))
def debug(database):
    """Debug database connection issues."""
    if database.lower() == 'mssql':
        debug_mssql_connection()


def debug_mssql_connection():
    """Debug MSSQL connection issues."""
    click.echo("\n" + "="*60)
    click.echo("🔍 MSSQL CONNECTION DEBUGGING")
    click.echo("="*60)

    docker_manager = DockerManager()
    container_name = "ELM_TOOL_mssql"

    if docker_manager.get_container_state(container_name) != ContainerState.RUNNING:
        click.echo(f"❌ {container_name} is not running")
        return

    click.echo(f"✅ {container_name} is running")

    # Check container logs
    success, logs, _ = docker_manager.run_command(
        ["docker", "logs", "--tail", "30", container_name], timeout=10
    )
    if success:
        click.echo(f"\n📋 Container logs:")
        click.echo(logs)

    # Try various connection methods
    connection_tests = [
        {
            "name": "Basic sqlcmd with trust certificate",
            "cmd": ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-Q", "SELECT 1"]
        },
        {
            "name": "sqlcmd with no encryption",
            "cmd": ["/opt/mssql-tools18/bin/sqlcmd", "-S", "localhost", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-N", "-Q", "SELECT 1"]
        },
        {
            "name": "sqlcmd with IP address",
            "cmd": ["/opt/mssql-tools18/bin/sqlcmd", "-S", "127.0.0.1", "-U", "sa", "-P", "ELM_TOOL_Password123!", "-C", "-Q", "SELECT 1"]
        },
        {
            "name": "Process check",
            "cmd": ["pgrep", "-f", "sqlservr"]
        }
    ]

    for test in connection_tests:
        click.echo(f"\n🔍 Testing: {test['name']}")
        cmd = ["docker", "exec", container_name] + test["cmd"]
        success, stdout, stderr = docker_manager.run_command(cmd, timeout=15)

        if success:
            click.echo(f"✅ Success: {stdout[:100]}...")
        else:
            click.echo(f"❌ Failed: {stderr[:200]}...")

    # Check if we can connect from host
    click.echo(f"\n🔍 Testing host connectivity")
    success, stdout, stderr = docker_manager.run_command(
        ["docker", "exec", container_name, "netstat", "-tln"], timeout=10
    )
    if success:
        click.echo(f"📋 Network ports:")
        click.echo(stdout)

def pull_container(docker_manager, configs):
    click.echo("\n📥 Pulling Docker images...")
    for db_name, config in configs.items():
        if not docker_manager.pull_image(config.image):
            click.echo(f"⚠️  Failed to pull {config.image}, continuing anyway...")

def remove_container(docker_manager, configs):
    click.echo("\n🗑️  Removing existing containers...")
    for db_name, config in configs.items():
        if docker_manager.get_container_state(config.name) != ContainerState.NOT_EXISTS:
            click.echo(f"🗑️  Removing {config.name}")
            docker_manager.remove_container(config.name)


if __name__ == "__main__":
    cli()
