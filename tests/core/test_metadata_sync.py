from unittest.mock import MagicMock, patch

import pytest


from elm.core.metadata_sync import (
    ColumnDef,
    TableDef,
    canonicalize_type,
    generate_sync_plan,
    synchronize_metadata,
)


def _col(name: str, type_str: str, nullable: bool = True, default=None) -> ColumnDef:
    return ColumnDef(
        name=name,
        type_str=type_str,
        canonical_type=canonicalize_type(type_str),
        nullable=nullable,
        default=default,
    )


def test_generate_sync_plan_creates_missing_table_and_adds_missing_column():
    src_tables = {
        ("dbo", "users"): TableDef(
            schema="dbo",
            name="users",
            columns=(
                _col("id", "INT", nullable=False),
                _col("name", "VARCHAR(50)", nullable=True),
            ),
            primary_key=("id",),
        ),
        ("dbo", "orders"): TableDef(
            schema="dbo",
            name="orders",
            columns=(
                _col("id", "INT", nullable=False),
                _col("amount", "DECIMAL(10,2)", nullable=True),
            ),
            primary_key=("id",),
        ),
    }

    tgt_tables = {
        ("dbo", "orders"): TableDef(
            schema="dbo",
            name="orders",
            columns=(
                _col("id", "INT", nullable=False),
            ),
            primary_key=("id",),
        ),
    }

    src_engine = MagicMock()
    tgt_engine = MagicMock()

    with patch("elm.core.metadata_sync.detect_database_type", side_effect=["postgresql", "mssql"]), patch(
        "elm.core.metadata_sync._extract_tables", side_effect=[src_tables, tgt_tables]
    ), patch("elm.core.metadata_sync._schema_exists", return_value=True):
        plan = generate_sync_plan(
            source_engine=src_engine,
            target_engine=tgt_engine,
            schema="dbo",
            tables=None,
        )

    # users table should be created
    assert any("CREATE TABLE" in ddl and "dbo.users" in ddl for ddl in plan.ddl_statements)
    # orders.amount should be added
    assert any("ALTER TABLE" in ddl and "orders" in ddl and "amount" in ddl for ddl in plan.ddl_statements)

    # metadata-only: no destructive operations
    for ddl in plan.ddl_statements:
        assert "DROP " not in ddl.upper()
        assert "INSERT " not in ddl.upper()

    # placeholder warning about unsupported object types
    assert any("not auto-synchronized yet" in w for w in plan.warnings)


@pytest.mark.file_io
def test_synchronize_metadata_writes_output_ddl_file(tmp_path):
    from elm.core.metadata_sync import SyncPlan, SyncAction

    output_file = tmp_path / "plan.sql"
    fake_plan = SyncPlan(
        source_db_type="postgresql",
        target_db_type="mssql",
        schema="dbo",
        actions=(SyncAction(action_type="CREATE_TABLE", object_type="TABLE", object_name="dbo.users", ddl="CREATE TABLE dbo.users (id INT)"),),
        ddl_statements=("CREATE TABLE dbo.users (id INT)",),
        warnings=(),
    )

    with patch("elm.core.metadata_sync.core_env.get_connection_url", side_effect=["postgresql://x", "mssql://y"]), patch(
        "elm.core.metadata_sync.create_engine", return_value=MagicMock()
    ), patch("elm.core.metadata_sync.generate_sync_plan", return_value=fake_plan):
        res = synchronize_metadata(
            source_env="src",
            target_env="tgt",
            dry_run=True,
            apply=False,
            output_ddl_file=str(output_file),
        )

    assert res.success is True
    assert output_file.exists()
    contents = output_file.read_text(encoding="utf-8")
    assert "CREATE TABLE dbo.users" in contents
    assert ";" in contents


def test_synchronize_metadata_returns_error_result_on_exception():
    with patch("elm.core.metadata_sync.core_env.get_connection_url", side_effect=Exception("boom")):
        res = synchronize_metadata(source_env="src", target_env="tgt")

    assert res.success is False
    assert "metadata synchronize" in res.message.lower()
