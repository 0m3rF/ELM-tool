"""elm.core.metadata_sync

Metadata-only synchronization (DDL) between two database environments.

This module intentionally does *not* move/copy data (no DML). It extracts
basic metadata from source/target, computes a diff, generates a DDL plan,
and optionally applies it to the target.

Supported today (best-effort, cross-platform):
- Schemas (PostgreSQL + SQL Server only)
- Tables: create missing tables, add missing columns

Reported (no automatic changes yet):
- Column type/nullable/default mismatches
- Views / routines / partitions (partial reporting)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import create_engine, inspect, text

from elm.core.streaming import detect_database_type
from elm.core.utils import (
    create_error_result,
    create_success_result,
    ensure_directory_exists,
    handle_exception,
)
from elm.core import environment as core_env
from elm.elm_utils import variables


@dataclass(frozen=True)
class CanonicalType:
    kind: str
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    raw: Optional[str] = None


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type_str: str
    canonical_type: CanonicalType
    nullable: bool
    default: Optional[str] = None


@dataclass(frozen=True)
class TableDef:
    schema: Optional[str]
    name: str
    columns: Tuple[ColumnDef, ...]
    primary_key: Tuple[str, ...] = ()
    is_partitioned: Optional[bool] = None

    @property
    def key(self) -> Tuple[Optional[str], str]:
        return (self.schema, self.name)


@dataclass(frozen=True)
class SyncAction:
    action_type: str  # CREATE_SCHEMA | CREATE_TABLE | ADD_COLUMN | WARNING
    object_type: str  # SCHEMA | TABLE | COLUMN | VIEW | ROUTINE
    object_name: str
    ddl: Optional[str] = None
    details: Optional[str] = None


@dataclass(frozen=True)
class SyncPlan:
    source_db_type: str
    target_db_type: str
    schema: Optional[str]
    actions: Tuple[SyncAction, ...]
    ddl_statements: Tuple[str, ...]
    warnings: Tuple[str, ...]


_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _default_schema_for(db_type: str) -> Optional[str]:
    if db_type == "postgresql":
        return "public"
    if db_type == "mssql":
        return "dbo"
    # Oracle schema is the user; SQLAlchemy inspector may set default_schema_name.
    return None


def _needs_quoting(identifier: str) -> bool:
    if not identifier:
        return True
    if not _SAFE_IDENT_RE.match(identifier):
        return True
    # minimal reserved words (cross-db-ish)
    if identifier.upper() in {"SELECT", "FROM", "WHERE", "TABLE", "VIEW", "USER", "GROUP", "ORDER"}:
        return True
    return False


def _quote_ident(db_type: str, identifier: str) -> str:
    if not _needs_quoting(identifier):
        return identifier

    if db_type == "mysql":
        return "`" + identifier.replace("`", "``") + "`"
    if db_type == "mssql":
        return "[" + identifier.replace("]", "]]" ) + "]"
    # postgres + oracle default
    return '"' + identifier.replace('"', '""') + '"'


def _qualify(db_type: str, schema: Optional[str], name: str) -> str:
    if schema:
        return f"{_quote_ident(db_type, schema)}.{_quote_ident(db_type, name)}"
    return _quote_ident(db_type, name)


def _parse_first_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def canonicalize_type(type_str: str) -> CanonicalType:
    """Convert a vendor type string into a small canonical set.

    This is intentionally heuristic; it is used for cross-platform mapping.
    """
    raw = (type_str or "").strip()
    t = raw.upper()

    # VARCHAR / CHAR
    m = re.match(r"^(N?VAR)?CHAR\s*\(([^\)]+)\)", t)
    if m:
        length = _parse_first_int(m.group(2).split(",")[0].strip())
        return CanonicalType(kind="VARCHAR", length=length, raw=raw)

    m = re.match(r"^(VARCHAR2|NVARCHAR2)\s*\(([^\)]+)\)", t)
    if m:
        length = _parse_first_int(m.group(2).split(",")[0].strip())
        return CanonicalType(kind="VARCHAR", length=length, raw=raw)

    if any(x in t for x in ["TEXT", "CLOB"]):
        return CanonicalType(kind="TEXT", raw=raw)

    # INTEGER-ish
    if t in {"INT", "INTEGER", "SMALLINT", "TINYINT"}:
        return CanonicalType(kind="INTEGER", raw=raw)
    if t in {"BIGINT"}:
        return CanonicalType(kind="BIGINT", raw=raw)

    # NUMERIC/DECIMAL/NUMBER
    m = re.match(r"^(DECIMAL|NUMERIC)\s*\(([^\)]+)\)", t)
    if m:
        parts = [p.strip() for p in m.group(2).split(",")]
        precision = _parse_first_int(parts[0]) if parts else None
        scale = _parse_first_int(parts[1]) if len(parts) > 1 else None
        return CanonicalType(kind="DECIMAL", precision=precision, scale=scale, raw=raw)

    m = re.match(r"^NUMBER\s*(\(([^\)]+)\))?", t)
    if m:
        if m.group(2):
            parts = [p.strip() for p in m.group(2).split(",")]
            precision = _parse_first_int(parts[0]) if parts else None
            scale = _parse_first_int(parts[1]) if len(parts) > 1 else 0
            if scale and scale > 0:
                return CanonicalType(kind="DECIMAL", precision=precision, scale=scale, raw=raw)
            if precision is not None:
                if precision <= 9:
                    return CanonicalType(kind="INTEGER", raw=raw)
                if precision <= 18:
                    return CanonicalType(kind="BIGINT", raw=raw)
            return CanonicalType(kind="DECIMAL", precision=precision, scale=0, raw=raw)
        return CanonicalType(kind="DECIMAL", raw=raw)

    # FLOAT/REAL
    if any(x in t for x in ["FLOAT", "REAL", "DOUBLE"]):
        return CanonicalType(kind="FLOAT", raw=raw)

    # DATE/TIME
    if t.startswith("TIMESTAMP") or "DATETIME" in t:
        return CanonicalType(kind="TIMESTAMP", raw=raw)
    if t == "DATE":
        return CanonicalType(kind="DATE", raw=raw)

    # BOOLEAN
    if t in {"BOOLEAN", "BOOL", "BIT"}:
        return CanonicalType(kind="BOOLEAN", raw=raw)

    # BINARY
    if any(x in t for x in ["BLOB", "BYTEA", "VARBINARY", "BINARY"]):
        return CanonicalType(kind="BLOB", raw=raw)

    return CanonicalType(kind="UNKNOWN", raw=raw)


def map_type_to_target(target_db_type: str, ct: CanonicalType) -> str:
    """Map a CanonicalType to a target database type string."""
    kind = ct.kind

    if kind == "VARCHAR":
        length = ct.length or 255
        if target_db_type == "oracle":
            return f"VARCHAR2({length})"
        if target_db_type == "mssql":
            return f"NVARCHAR({length})"
        return f"VARCHAR({length})"

    if kind == "TEXT":
        if target_db_type == "oracle":
            return "CLOB"
        if target_db_type == "mssql":
            return "NVARCHAR(MAX)"
        return "TEXT"

    if kind == "INTEGER":
        if target_db_type == "oracle":
            return "NUMBER(10,0)"
        return "INTEGER" if target_db_type == "postgresql" else "INT"

    if kind == "BIGINT":
        if target_db_type == "oracle":
            return "NUMBER(19,0)"
        return "BIGINT"

    if kind == "DECIMAL":
        p = ct.precision or 38
        s = ct.scale or 0
        if target_db_type == "oracle":
            return f"NUMBER({p},{s})"
        return f"DECIMAL({p},{s})"

    if kind == "FLOAT":
        return "FLOAT"

    if kind == "BOOLEAN":
        if target_db_type == "oracle":
            # Oracle has no native BOOLEAN in SQL (pre-23c). Use NUMBER(1).
            return "NUMBER(1,0)"
        if target_db_type == "mssql":
            return "BIT"
        return "BOOLEAN"

    if kind == "DATE":
        if target_db_type == "mssql":
            return "DATE"
        return "DATE"

    if kind == "TIMESTAMP":
        if target_db_type == "mssql":
            return "DATETIME2"
        if target_db_type == "oracle":
            return "TIMESTAMP"
        return "TIMESTAMP"

    if kind == "BLOB":
        if target_db_type == "postgresql":
            return "BYTEA"
        if target_db_type == "oracle":
            return "BLOB"
        if target_db_type == "mssql":
            return "VARBINARY(MAX)"
        return "BLOB"

    # Fallback: be conservative
    if target_db_type == "oracle":
        return "CLOB"
    return "TEXT"


def _render_column_def(
    target_db_type: str,
    col: ColumnDef,
    *,
    include_default: bool,
) -> str:
    col_name = _quote_ident(target_db_type, col.name)
    col_type = map_type_to_target(target_db_type, col.canonical_type)
    parts = [col_name, col_type]

    if include_default and col.default:
        parts.append(f"DEFAULT {col.default}")

    parts.append("NULL" if col.nullable else "NOT NULL")
    return " ".join(parts)


def _render_create_schema(target_db_type: str, schema: str) -> Optional[str]:
    if target_db_type in {"postgresql", "mssql"}:
        return f"CREATE SCHEMA {_quote_ident(target_db_type, schema)}"
    return None


def _render_create_table(
    target_db_type: str,
    table: TableDef,
    *,
    include_defaults: bool,
) -> str:
    fq = _qualify(target_db_type, table.schema, table.name)

    col_lines = [
        _render_column_def(target_db_type, c, include_default=include_defaults)
        for c in table.columns
    ]

    if table.primary_key:
        pk_cols = ", ".join(_quote_ident(target_db_type, c) for c in table.primary_key)
        col_lines.append(f"PRIMARY KEY ({pk_cols})")

    cols_sql = ",\n  ".join(col_lines)
    return f"CREATE TABLE {fq} (\n  {cols_sql}\n)"


def _render_add_column(target_db_type: str, table: TableDef, col: ColumnDef, *, include_default: bool) -> str:
    fq = _qualify(target_db_type, table.schema, table.name)
    col_sql = _render_column_def(target_db_type, col, include_default=include_default)

    if target_db_type == "oracle":
        return f"ALTER TABLE {fq} ADD ({col_sql})"
    if target_db_type == "postgresql":
        return f"ALTER TABLE {fq} ADD COLUMN {col_sql}"
    return f"ALTER TABLE {fq} ADD {col_sql}"


def _extract_tables(
    engine,
    db_type: str,
    schema: Optional[str],
    table_filter: Optional[Sequence[str]],
) -> Dict[Tuple[Optional[str], str], TableDef]:
    insp = inspect(engine)
    eff_schema = schema or getattr(insp, "default_schema_name", None) or _default_schema_for(db_type)

    tables: Dict[Tuple[Optional[str], str], TableDef] = {}
    for tname in insp.get_table_names(schema=eff_schema):
        if table_filter and tname not in table_filter:
            continue
        cols_raw = insp.get_columns(tname, schema=eff_schema)
        cols: List[ColumnDef] = []
        for c in cols_raw:
            type_str = str(c.get("type"))
            cols.append(
                ColumnDef(
                    name=c.get("name"),
                    type_str=type_str,
                    canonical_type=canonicalize_type(type_str),
                    nullable=bool(c.get("nullable", True)),
                    default=c.get("default"),
                )
            )

        pk = insp.get_pk_constraint(tname, schema=eff_schema) or {}
        pk_cols = tuple(pk.get("constrained_columns") or [])

        tables[(eff_schema, tname)] = TableDef(
            schema=eff_schema,
            name=tname,
            columns=tuple(cols),
            primary_key=pk_cols,
        )
    return tables


def _schema_exists(engine, schema: str) -> bool:
    insp = inspect(engine)
    try:
        return schema in (insp.get_schema_names() or [])
    except Exception:
        # Some dialects may not implement this well.
        return False


def generate_sync_plan(
    *,
    source_engine,
    target_engine,
    schema: Optional[str],
    tables: Optional[Sequence[str]],
    include_defaults_same_db_only: bool = True,
) -> SyncPlan:
    source_db_type = detect_database_type(str(source_engine.url))
    target_db_type = detect_database_type(str(target_engine.url))

    src_tables = _extract_tables(source_engine, source_db_type, schema, tables)
    tgt_tables = _extract_tables(target_engine, target_db_type, schema, tables)

    eff_schema = schema or _default_schema_for(source_db_type) or _default_schema_for(target_db_type)

    actions: List[SyncAction] = []
    ddls: List[str] = []
    warnings: List[str] = []

    # Schema creation (where applicable)
    if eff_schema and not _schema_exists(target_engine, eff_schema):
        ddl_schema = _render_create_schema(target_db_type, eff_schema)
        if ddl_schema:
            actions.append(
                SyncAction(
                    action_type="CREATE_SCHEMA",
                    object_type="SCHEMA",
                    object_name=eff_schema,
                    ddl=ddl_schema,
                )
            )
            ddls.append(ddl_schema)
        else:
            warnings.append(f"Target database type '{target_db_type}' does not support CREATE SCHEMA via this tool.")

    include_defaults = (source_db_type == target_db_type) and include_defaults_same_db_only

    # Table create / alter
    for key, src_table in sorted(src_tables.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        tgt_table = tgt_tables.get(key)
        if not tgt_table:
            ddl = _render_create_table(target_db_type, src_table, include_defaults=include_defaults)
            actions.append(
                SyncAction(
                    action_type="CREATE_TABLE",
                    object_type="TABLE",
                    object_name=_qualify(target_db_type, src_table.schema, src_table.name),
                    ddl=ddl,
                )
            )
            ddls.append(ddl)
            continue

        tgt_cols_by_name = {c.name.lower(): c for c in tgt_table.columns}
        for src_col in src_table.columns:
            if src_col.name.lower() not in tgt_cols_by_name:
                ddl = _render_add_column(target_db_type, tgt_table, src_col, include_default=include_defaults)
                actions.append(
                    SyncAction(
                        action_type="ADD_COLUMN",
                        object_type="COLUMN",
                        object_name=f"{_qualify(target_db_type, src_table.schema, src_table.name)}.{src_col.name}",
                        ddl=ddl,
                    )
                )
                ddls.append(ddl)
            else:
                tgt_col = tgt_cols_by_name[src_col.name.lower()]
                # Report differences (no auto-alter yet)
                if src_col.canonical_type.kind != tgt_col.canonical_type.kind:
                    warnings.append(
                        f"Type mismatch for {src_table.schema}.{src_table.name}.{src_col.name}: "
                        f"source={src_col.type_str} target={tgt_col.type_str} (no automatic ALTER generated)"
                    )
                if bool(src_col.nullable) != bool(tgt_col.nullable):
                    warnings.append(
                        f"Nullability mismatch for {src_table.schema}.{src_table.name}.{src_col.name}: "
                        f"source={'NULL' if src_col.nullable else 'NOT NULL'} target={'NULL' if tgt_col.nullable else 'NOT NULL'} "
                        "(no automatic ALTER generated)"
                    )

    # Warn about extra tables in target (no drops)
    extra = [k for k in tgt_tables.keys() if k not in src_tables]
    if extra:
        warnings.append(
            f"Target has {len(extra)} extra table(s) not present in source. "
            "This tool does not DROP objects automatically."
        )

    # Surface placeholder capabilities for other object types
    warnings.append("Views/routines/packages/partition definitions are not auto-synchronized yet (tables/columns only).")

    return SyncPlan(
        source_db_type=source_db_type,
        target_db_type=target_db_type,
        schema=eff_schema,
        actions=tuple(actions),
        ddl_statements=tuple(ddls),
        warnings=tuple(warnings),
    )


def _write_audit_line(path: str, line: str) -> None:
    ensure_directory_exists(path)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line}\n")


def synchronize_metadata(
    *,
    source_env: str,
    target_env: str,
    schema: Optional[str] = None,
    tables: Optional[Sequence[str]] = None,
    dry_run: bool = True,
    apply: bool = False,
    source_encryption_key: Optional[str] = None,
    target_encryption_key: Optional[str] = None,
    output_ddl_file: Optional[str] = None,
    audit_log_file: Optional[str] = None,
) -> "elm.core.types.OperationResult":
    """Synchronize metadata (DDL) from source to target.

    Notes:
    - Metadata-only: only DDL statements are generated/executed.
    - If apply=False, no changes are made.
    """
    try:
        if apply:
            dry_run = False

        audit_path = audit_log_file or os.path.join(variables.ELM_TOOL_HOME, "metadata_sync_audit.log")

        src_url = core_env.get_connection_url(source_env, source_encryption_key)
        tgt_url = core_env.get_connection_url(target_env, target_encryption_key)

        src_engine = create_engine(src_url)
        tgt_engine = create_engine(tgt_url)

        plan = generate_sync_plan(
            source_engine=src_engine,
            target_engine=tgt_engine,
            schema=schema,
            tables=tables,
        )

        if output_ddl_file:
            ensure_directory_exists(output_ddl_file)
            with open(output_ddl_file, "w", encoding="utf-8") as f:
                for stmt in plan.ddl_statements:
                    f.write(stmt.rstrip() + ";\n\n")

        applied = 0
        errors: List[str] = []

        if apply:
            _write_audit_line(audit_path, f"START metadata sync source={source_env} target={target_env} schema={plan.schema}")
            _write_audit_line(audit_path, f"DB types: source={plan.source_db_type} target={plan.target_db_type}")

            # Apply in order. Use an explicit transaction when possible.
            with tgt_engine.begin() as conn:
                for stmt in plan.ddl_statements:
                    try:
                        _write_audit_line(audit_path, f"EXECUTE: {stmt}")
                        conn.execute(text(stmt))
                        applied += 1
                    except Exception as e:
                        msg = f"Failed statement: {stmt} | Error: {e}"
                        errors.append(msg)
                        _write_audit_line(audit_path, f"ERROR: {msg}")
                        raise

            _write_audit_line(audit_path, f"DONE metadata sync applied_statements={applied}")

        summary = {
            "source_db_type": plan.source_db_type,
            "target_db_type": plan.target_db_type,
            "schema": plan.schema,
            "ddl": list(plan.ddl_statements),
            "actions": [a.__dict__ for a in plan.actions],
            "warnings": list(plan.warnings),
            "applied_statements": applied,
            "dry_run": dry_run,
        }

        msg = (
            f"Metadata synchronize plan generated: {len(plan.ddl_statements)} DDL statement(s), "
            f"{len(plan.warnings)} warning(s)."
        )
        if apply:
            msg = f"Metadata synchronized successfully: applied {applied} DDL statement(s)."

        return create_success_result(msg, data=summary, record_count=len(plan.ddl_statements))
    except Exception as e:
        return handle_exception(e, "metadata synchronize")
