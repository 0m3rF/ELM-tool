import click
from elm.core import copy as core_copy
from elm.elm_utils.command_utils import AliasedGroup

@click.group(cls=AliasedGroup)
def copy():
    """Data copy commands for database operations"""
    pass

@copy.command()
@click.option("-s", "--source", required=True, help="Source environment name")
@click.option("-q", "--query", required=True, help="SQL query to execute")
@click.option("-f", "--file", required=True, help="Output file path")
@click.option("-t", "--format", type=click.Choice(['CSV', 'JSON'], case_sensitive=False), default='CSV', help="Output file format")
@click.option("-m", "--mode", type=click.Choice(['OVERWRITE', 'APPEND'], case_sensitive=False), default='OVERWRITE', help="File write mode")
@click.option("-b", "--batch-size", type=int, default=None, help="Batch size for processing large datasets")
@click.option("-p", "--parallel", type=int, default=1, help="Number of parallel processes")
@click.option("-k", "--encryption-key", required=False, help="Encryption key for encrypted environments")
@click.option("--no-mask", is_flag=True, help="Disable data masking")
@click.option(
    "--verbose-batch-logs/--no-verbose-batch-logs",
    default=True,
    help="Enable or disable per-batch timing logs (summary is always shown).",
    show_default=True,
)
def db2file(source, query, file, format, mode, batch_size, parallel, encryption_key, no_mask, verbose_batch_logs):
    """Copy data from database to file"""
    # Use core module for the operation
    result = core_copy.copy_db_to_file(
        source_env=source,
        query=query,
        file_path=file,
        file_format=format.lower(),
        mode=mode,
        batch_size=batch_size,
        parallel_workers=parallel,
        source_encryption_key=encryption_key,
        apply_masks=not no_mask,
        verbose_batch_logs=verbose_batch_logs,
    )
    
    if result.success:
        click.echo(result.message)
        if result.record_count:
            click.echo(f"Records processed: {result.record_count}")
    else:
        raise click.UsageError(result.message)

@copy.command()
@click.option("-s", "--source", required=True, help="Source file path")
@click.option("-t", "--target", required=True, help="Target environment name")
@click.option("-T", "--table", required=True, help="Target table name")
@click.option("-f", "--format", type=click.Choice(['CSV', 'JSON'], case_sensitive=False), default='CSV', help="Input file format")
@click.option("-m", "--mode", type=click.Choice(['APPEND', 'OVERWRITE', 'FAIL'], case_sensitive=False), default='APPEND', help="Database write mode")
@click.option("-b", "--batch-size", type=int, default=1000, help="Batch size for processing large datasets")
@click.option("-p", "--parallel", type=int, default=1, help="Number of parallel processes")
@click.option("-k", "--encryption-key", required=False, help="Encryption key for encrypted environments")
@click.option("--validate-target", is_flag=True, help="Validate that target table exists and has all required columns")
@click.option("--create-if-not-exists", is_flag=True, help="Create target table if it doesn't exist")
@click.option("--no-mask", is_flag=True, help="Disable data masking")
@click.option(
    "--verbose-batch-logs/--no-verbose-batch-logs",
    default=True,
    help="Enable or disable per-batch timing logs (summary is always shown).",
    show_default=True,
)
def file2db(source, target, table, format, mode, batch_size, parallel, encryption_key, validate_target, create_if_not_exists, no_mask, verbose_batch_logs):
    """Copy data from file to database"""
    # Use core module for the operation
    result = core_copy.copy_file_to_db(
        file_path=source,
        target_env=target,
        table=table,
        file_format=format.lower(),
        mode=mode,
        batch_size=batch_size,
        parallel_workers=parallel,
        target_encryption_key=encryption_key,
        validate_target=validate_target,
        create_if_not_exists=create_if_not_exists,
        apply_masks=not no_mask,
        verbose_batch_logs=verbose_batch_logs,
    )
    
    if result.success:
        click.echo(result.message)
        if result.record_count:
            click.echo(f"Records processed: {result.record_count}")
    else:
        raise click.UsageError(result.message)

@copy.command()
@click.option("-s", "--source", required=True, help="Source environment name")
@click.option("-t", "--target", required=True, help="Target environment name")
@click.option("-q", "--query", required=True, help="SQL query to execute on source")
@click.option("-T", "--table", required=True, help="Target table name")
@click.option("-m", "--mode", type=click.Choice(['APPEND', 'OVERWRITE', 'FAIL'], case_sensitive=False), default='APPEND', help="Database write mode")
@click.option("-b", "--batch-size", type=int, default=1000, help="Batch size for processing large datasets")
@click.option("-p", "--parallel", type=int, default=1, help="Number of parallel processes")
@click.option("-sk", "--source-key", required=False, help="Encryption key for source environment")
@click.option("-tk", "--target-key", required=False, help="Encryption key for target environment")
@click.option("--validate-target", is_flag=True, help="Validate that target table exists and has all required columns")
@click.option("--create-if-not-exists", is_flag=True, help="Create target table if it doesn't exist")
@click.option("--no-mask", is_flag=True, help="Disable data masking")
@click.option(
    "--verbose-batch-logs/--no-verbose-batch-logs",
    default=True,
    help="Enable or disable per-batch timing logs (summary is always shown).",
    show_default=True,
)
def db2db(source, target, query, table, mode, batch_size, parallel, source_key, target_key, validate_target, create_if_not_exists, no_mask, verbose_batch_logs):
    """Copy data from one database to another"""
    # Use core module for the operation
    result = core_copy.copy_db_to_db(
        source_env=source,
        target_env=target,
        query=query,
        table=table,
        mode=mode,
        batch_size=batch_size,
        parallel_workers=parallel,
        source_encryption_key=source_key,
        target_encryption_key=target_key,
        validate_target=validate_target,
        create_if_not_exists=create_if_not_exists,
        apply_masks=not no_mask,
        verbose_batch_logs=verbose_batch_logs,
    )
    
    if result.success:
        click.echo(result.message)
        if result.record_count:
            click.echo(f"Records processed: {result.record_count}")
    else:
        raise click.UsageError(result.message)

# Define aliases for commands
ALIASES = {
    'database2file': db2file,
    'db-to-file': db2file,
    'file2database': file2db,
    'file-to-db': file2db,
    'database2database': db2db,
    'db-to-db': db2db,
}
