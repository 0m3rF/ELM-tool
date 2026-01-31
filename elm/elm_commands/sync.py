import click

from elm.core import metadata_sync as core_metadata_sync
from elm.elm_utils.command_utils import AliasedGroup


@click.group(cls=AliasedGroup)
@click.help_option('-h', '--help')
def sync():
    """Synchronization commands (metadata-only)."""
    pass


@sync.command("metadata")
@click.option("-s", "--source", required=True, help="Source environment name")
@click.option("-t", "--target", required=True, help="Target environment name")
@click.option("--schema", required=False, help="Schema name to synchronize (optional)")
@click.option(
    "--table",
    "tables",
    multiple=True,
    required=False,
    help="Limit to table(s). Can be provided multiple times: --table users --table orders",
)
@click.option(
    "--dry-run/--apply",
    default=True,
    show_default=True,
    help="Preview DDL only (default) or apply generated DDL to target.",
)
@click.option("--output-ddl-file", required=False, help="Write generated DDL statements to this file")
@click.option("--audit-log-file", required=False, help="Append audit log lines to this file (only when applying)")
@click.option("-sk", "--source-key", required=False, help="Encryption key for source environment")
@click.option("-tk", "--target-key", required=False, help="Encryption key for target environment")
@click.option(
    "--show-ddl/--no-show-ddl",
    default=True,
    show_default=True,
    help="Print generated DDL statements to stdout.",
)
def metadata(
    source,
    target,
    schema,
    tables,
    dry_run,
    output_ddl_file,
    audit_log_file,
    source_key,
    target_key,
    show_ddl,
):
    """Synchronize database object definitions (DDL/metadata only).

    Notes:
    - No data copy (no DML).
    - No destructive DDL (no DROP) is generated.
    """

    table_list = list(tables) if tables else None
    apply_changes = not dry_run

    result = core_metadata_sync.synchronize_metadata(
        source_env=source,
        target_env=target,
        schema=schema,
        tables=table_list,
        dry_run=dry_run,
        apply=apply_changes,
        source_encryption_key=source_key,
        target_encryption_key=target_key,
        output_ddl_file=output_ddl_file,
        audit_log_file=audit_log_file,
    )

    if not result.success:
        raise click.UsageError(result.message)

    click.echo(result.message)

    data = result.data or {}
    warnings = data.get("warnings") or []
    ddls = data.get("ddl") or []

    if warnings:
        click.echo("\nWarnings:")
        for w in warnings:
            click.echo(f"  - {w}")

    if show_ddl and ddls:
        click.echo("\nDDL statements:")
        for i, stmt in enumerate(ddls, start=1):
            click.echo(f"\n-- [{i}] ------------------------------")
            click.echo(stmt.rstrip() + ";")


# Subcommand aliases
ALIASES = {
    "meta": metadata,
    "ddl": metadata,
    "plan": metadata,
}
