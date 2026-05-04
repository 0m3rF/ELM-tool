import json
from datetime import datetime

import click
from elm.core import copy as core_copy
from elm.core.history import HistoryRecorder, HistoryRecord
from elm.core.types import OperationResult
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

@copy.command(name="list")
@click.option("--status", help="Filter by status (success/failure)")
@click.option("--source", help="Filter by source environment or file")
@click.option("--target", help="Filter by target environment or file")
@click.option("--operation-type", help="Filter by operation type (db2db/db2file/file2db)")
@click.option("--table", help="Filter by table name")
@click.option("--limit", type=int, default=10, help="Max results (0 = unlimited)")
@click.option("--sort", type=click.Choice(["asc", "desc"]), default="desc", help="Sort order by date")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table", help="Output format")
def list_history(status, source, target, operation_type, table, limit, sort, output_format):
    """List previous copy operations"""
    recorder = HistoryRecorder()
    records = recorder.read_records()

    if not records:
        click.echo("No history records found.")
        return

    # Apply filters
    filters = {
        "status": status,
        "source": source,
        "target": target,
        "operation_type": operation_type,
        "table": table,
    }
    for field, value in filters.items():
        if value is not None:
            value_lower = value.lower()
            records = [
                r for r in records
                if getattr(r, field) is not None and getattr(r, field).lower() == value_lower
            ]

    # Sort
    records = sorted(records, key=lambda r: r.date, reverse=(sort == "desc"))

    # Apply limit
    if limit > 0:
        records = records[:limit]

    if output_format == "json":
        from dataclasses import asdict
        click.echo(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))
    else:
        # Table format
        headers = ["ID", "Date", "Type", "Source", "Target", "Table", "Status"]
        col_widths = [5, 20, 10, 15, 15, 15, 10]

        # Header row
        header_line = "  ".join(
            h.ljust(w) for h, w in zip(headers, col_widths)
        )
        click.echo(header_line)
        click.echo("-" * len(header_line))

        for r in records:
            row = [
                str(r.id),
                (r.date or "")[:20],
                (r.operation_type or "")[:10],
                (r.source or "")[:15],
                (r.target or "")[:15],
                (r.table or "")[:15],
                (r.status or "")[:10],
            ]
            click.echo("  ".join(
                cell.ljust(w) for cell, w in zip(row, col_widths)
            ))


@copy.command(name="re-run")
@click.argument("id", type=int)
def re_run(id):
    """Re-run a previous copy operation"""
    recorder = HistoryRecorder()
    record = recorder.get_record(id)

    if record is None:
        click.echo(f"Error: History record {id} not found.", err=True)
        raise click.ClickException("Record not found")

    result = None
    _history_params = None

    if record.operation_type == "db2db":
        _history_params = {
            "operation_type": "db2db",
            "source_env": record.source,
            "target_env": record.target,
            "query": record.query,
            "table": record.table,
            "mode": record.mode or "APPEND",
            "batch_size": record.batch_size,
            "parallel_workers": record.parallel_workers or 1,
        }
        result = core_copy.copy_db_to_db(
            source_env=record.source,
            target_env=record.target,
            query=record.query,
            table=record.table,
            mode=record.mode or "APPEND",
            batch_size=record.batch_size,
            parallel_workers=record.parallel_workers or 1,
            apply_masks=True,
            validate_target=False,
            create_if_not_exists=False,
            verbose_batch_logs=True,
        )
    elif record.operation_type == "db2file":
        _history_params = {
            "operation_type": "db2file",
            "source_env": record.source,
            "file_path": record.target,
            "query": record.query,
            "mode": record.mode or "REPLACE",
            "batch_size": record.batch_size,
            "parallel_workers": record.parallel_workers or 1,
        }
        result = core_copy.copy_db_to_file(
            source_env=record.source,
            query=record.query,
            file_path=record.target,
            file_format="csv",
            mode=record.mode or "REPLACE",
            batch_size=record.batch_size,
            parallel_workers=record.parallel_workers or 1,
            apply_masks=True,
            verbose_batch_logs=True,
        )
    elif record.operation_type == "file2db":
        _history_params = {
            "operation_type": "file2db",
            "file_path": record.source,
            "target_env": record.target,
            "table": record.table,
            "mode": record.mode or "APPEND",
            "batch_size": record.batch_size,
            "parallel_workers": record.parallel_workers or 1,
        }
        result = core_copy.copy_file_to_db(
            file_path=record.source,
            target_env=record.target,
            table=record.table,
            file_format="csv",
            mode=record.mode or "APPEND",
            batch_size=record.batch_size,
            parallel_workers=record.parallel_workers or 1,
            apply_masks=True,
            validate_target=False,
            create_if_not_exists=False,
            verbose_batch_logs=True,
        )
    else:
        click.echo(f"Error: Unknown operation type '{record.operation_type}'", err=True)
        raise click.ClickException("Unknown operation type")

    if result.success:
        recorder.update_record(
            id,
            last_run_date=datetime.now().isoformat(),
            status="success",
            record_count=result.record_count,
            end_time=datetime.now().isoformat(),
            error_message=None,
        )
        click.echo(result.message)
        if result.record_count:
            click.echo(f"Records processed: {result.record_count}")
    else:
        # Check if failure is due to missing environment/file
        if result.message and ("not found" in result.message.lower() or "does not exist" in result.message.lower()):
            failure_result = OperationResult(
                success=False,
                message=result.message,
                record_count=result.record_count,
            )
            recorder.record(failure_result, _history_params)
        click.echo(result.message, err=True)
        raise click.UsageError(result.message)


@copy.command(name="edit")
@click.argument("id", type=int)
@click.option("-s", "--source", help="Override source environment or file path")
@click.option("-t", "--target", help="Override target environment or file path")
@click.option("-q", "--query", help="Override SQL query")
@click.option("-T", "--table", help="Override target table name")
@click.option("-f", "--format", type=click.Choice(["CSV", "JSON"]), help="Override file format (for db2file/file2db)")
@click.option("-m", "--mode", type=click.Choice(["APPEND", "OVERWRITE", "FAIL"]), help="Override write mode")
@click.option("-b", "--batch-size", type=int, help="Override batch size")
@click.option("-p", "--parallel", type=int, help="Override parallel workers")
@click.option("-k", "--encryption-key", help="Override encryption key")
@click.option("--no-mask", is_flag=True, help="Disable data masking")
@click.option("--validate-target", is_flag=True, help="Validate target table exists")
@click.option("--create-if-not-exists", is_flag=True, help="Create target table if missing")
def edit_history(id, source, target, query, table, format, mode, batch_size, parallel, encryption_key, no_mask, validate_target, create_if_not_exists):
    """Edit and re-run a previous copy operation"""
    recorder = HistoryRecorder()
    record = recorder.get_record(id)

    if record is None:
        click.echo(f"Error: History record {id} not found.", err=True)
        raise click.ClickException("Record not found")

    # Resolve overrides
    resolved_source = source if source is not None else record.source
    resolved_target = target if target is not None else record.target
    resolved_query = query if query is not None else record.query
    resolved_table = table if table is not None else record.table
    resolved_mode = mode if mode is not None else (record.mode or "APPEND")
    resolved_batch_size = batch_size if batch_size is not None else record.batch_size
    resolved_parallel = parallel if parallel is not None else (record.parallel_workers or 1)
    resolved_apply_masks = not no_mask

    # Preview
    click.echo("Preview of edited operation:")
    click.echo("─────────────────────────")
    click.echo(f"Operation type: {record.operation_type}")
    click.echo(f"Source: {resolved_source}")
    click.echo(f"Target: {resolved_target}")
    click.echo(f"Query: {resolved_query}")
    click.echo(f"Table: {resolved_table}")
    click.echo(f"Mode: {resolved_mode}")
    click.echo(f"Batch size: {resolved_batch_size}")
    click.echo(f"Parallel workers: {resolved_parallel}")
    click.echo("─────────────────────────")

    result = None
    _history_params = None

    if record.operation_type == "db2db":
        _history_params = {
            "operation_type": "db2db",
            "source_env": resolved_source,
            "target_env": resolved_target,
            "query": resolved_query,
            "table": resolved_table,
            "mode": resolved_mode,
            "batch_size": resolved_batch_size,
            "parallel_workers": resolved_parallel,
        }
        result = core_copy.copy_db_to_db(
            source_env=resolved_source,
            target_env=resolved_target,
            query=resolved_query,
            table=resolved_table,
            mode=resolved_mode,
            batch_size=resolved_batch_size,
            parallel_workers=resolved_parallel,
            apply_masks=resolved_apply_masks,
            validate_target=validate_target,
            create_if_not_exists=create_if_not_exists,
            verbose_batch_logs=True,
        )
    elif record.operation_type == "db2file":
        _history_params = {
            "operation_type": "db2file",
            "source_env": resolved_source,
            "file_path": resolved_target,
            "query": resolved_query,
            "mode": resolved_mode,
            "batch_size": resolved_batch_size,
            "parallel_workers": resolved_parallel,
        }
        result = core_copy.copy_db_to_file(
            source_env=resolved_source,
            query=resolved_query,
            file_path=resolved_target,
            file_format=(format or "CSV").lower(),
            mode=resolved_mode,
            batch_size=resolved_batch_size,
            parallel_workers=resolved_parallel,
            source_encryption_key=encryption_key,
            apply_masks=resolved_apply_masks,
            verbose_batch_logs=True,
        )
    elif record.operation_type == "file2db":
        _history_params = {
            "operation_type": "file2db",
            "file_path": resolved_source,
            "target_env": resolved_target,
            "table": resolved_table,
            "mode": resolved_mode,
            "batch_size": resolved_batch_size,
            "parallel_workers": resolved_parallel,
        }
        result = core_copy.copy_file_to_db(
            file_path=resolved_source,
            target_env=resolved_target,
            table=resolved_table,
            file_format=(format or "CSV").lower(),
            mode=resolved_mode,
            batch_size=resolved_batch_size,
            parallel_workers=resolved_parallel,
            target_encryption_key=encryption_key,
            validate_target=validate_target,
            create_if_not_exists=create_if_not_exists,
            apply_masks=resolved_apply_masks,
            verbose_batch_logs=True,
        )
    else:
        click.echo(f"Error: Unknown operation type '{record.operation_type}'", err=True)
        raise click.ClickException("Unknown operation type")

    # Always create a new history entry
    recorder.record(result, _history_params)

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
    'history': list_history,
    'ls': list_history,
    'rerun': re_run,
}
