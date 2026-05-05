# Phase 05: CLI History Commands - Research

**Researched:** 2026-05-04
**Domain:** Click CLI patterns, table formatting, JSON read-update-write, pytest CLI testing
**Confidence:** HIGH

## Summary

Phase 5 adds three Click subcommands (`list`, `re-run`, `edit`) that interact with the existing `history.json` from Phase 4. The domain is well-understood: Click CLI patterns are already established in the project, `history.json` is a flat JSON array, and the `HistoryRecorder` from Phase 4 provides the foundation. No new external dependencies are needed.

Key constraints from CONTEXT.md:
- `list` must support human-readable table and JSON output
- `re-run` must dispatch to the correct core copy function based on `operation_type`
- `re-run` updates the original record on success (not a new entry)
- `edit` always creates a new history entry after showing a preview
- All commands must handle missing IDs gracefully (non-zero exit, stderr)
- Filtering and sorting must be implemented in-memory (flat JSON, small dataset)

**Primary recommendation:** Extend `HistoryRecorder` with `read_records()`, `get_record()`, and `update_record()` methods, then add three Click subcommands to `elm/elm_commands/copy.py` using existing patterns.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| History reading/filtering | Core (`elm/core/history.py`) | — | Reuses `HistoryRecorder` config and locking |
| CLI command definitions | CLI (`elm/elm_commands/copy.py`) | — | Follows existing Click group pattern |
| Table formatting | CLI (`elm/elm_commands/copy.py`) | — | Manual string formatting avoids new deps |
| Core dispatch | CLI (`elm/elm_commands/copy.py`) | `elm/core/copy.py` | Calls existing public copy functions |
| Tests | Tests (`tests/test_history_cli.py`) | — | Click CliRunner + mock core functions |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `json` (stdlib) | 3.12 | JSON deserialization for list | Already used in `history.py` |
| `click` (existing) | project | CLI commands, arguments, options | Already used throughout project |
| `datetime` (stdlib) | 3.12 | Timestamps for updates | Already used in `history.py` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `file_lock` (project) | existing | Thread-safe history reads/updates | All reads and updates to `history.json` |
| `HistoryRecorder` (project) | Phase 4 | Read, filter, update records | `list`, `re-run`, `edit` commands |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual table formatting | `tabulate` | Adds dependency; manual formatting is 20 lines and sufficient for CLI |
| Manual table formatting | `rich.table` | Adds heavy dependency; overkill for 7-column table |
| In-memory filtering | SQLite query | Overkill; history.json is bounded to 100 entries |

## Architecture Patterns

### Pattern 1: HistoryReader Extension
**What:** Add `read_records()`, `get_record(id)`, `update_record(id, **kwargs)` to existing `HistoryRecorder`.
**When to use:** All three CLI commands need to read and optionally update history.
**Example:**
```python
class HistoryRecorder:
    # ... existing record() method ...
    
    def read_records(self) -> List[HistoryRecord]:
        """Read all records from history file."""
        return self._read_records(self.config.get_history_file())
    
    def get_record(self, record_id: int) -> Optional[HistoryRecord]:
        """Get single record by ID."""
        for r in self.read_records():
            if r.id == record_id:
                return r
        return None
    
    def update_record(self, record_id: int, **kwargs) -> bool:
        """Update fields of an existing record under file_lock."""
        history_file = self.config.get_history_file()
        with file_lock(history_file):
            records = self._read_records(history_file)
            for r in records:
                if r.id == record_id:
                    for k, v in kwargs.items():
                        if hasattr(r, k):
                            setattr(r, k, v)
                    self._write_records(history_file, records)
                    return True
            return False
```

### Pattern 2: In-Memory Filter/Sort/Limit
**What:** Read all records, apply Python list comprehensions for filtering, `sorted()` for sorting, slice for limit.
**When to use:** `list` command filtering (`--status`, `--source`, `--target`, `--operation-type`, `--table`).
**Example:**
```python
records = recorder.read_records()

# Filter
if status:
    records = [r for r in records if r.status == status]
if source:
    records = [r for r in records if r.source and r.source.lower() == source.lower()]
# ... etc for each filter ...

# Sort
records = sorted(records, key=lambda r: r.date, reverse=(sort == "desc"))

# Limit
if limit > 0:
    records = records[:limit]
```

### Pattern 3: Manual Table Formatting
**What:** Use `str.ljust`/`rjust` with computed column widths for CLI table output.
**When to use:** `list --format table` (default).
**Example:**
```python
columns = [("ID", 4), ("Date", 20), ("Type", 10), ("Source", 15), ("Target", 15), ("Table", 15), ("Status", 8)]
# Build header
header = " | ".join(name.ljust(width) for name, width in columns)
click.echo(header)
click.echo("-" * len(header))
# Build rows
for r in records:
    row = " | ".join(str(val).ljust(width) for val, (name, width) in zip([r.id, r.date[:19], r.operation_type, r.source or "", r.target or "", r.table or "", r.status], columns))
    click.echo(row)
```

### Pattern 4: Core Function Dispatch
**What:** Switch on `record.operation_type` to call the correct `core_copy.*` function with stored parameters.
**When to use:** `re-run` and `edit` commands.
**Example:**
```python
if record.operation_type == "db2db":
    result = core_copy.copy_db_to_db(
        source_env=record.source,
        target_env=record.target,
        query=record.query,
        table=record.table,
        mode=record.mode or "APPEND",
        batch_size=record.batch_size,
        parallel_workers=record.parallel_workers or 1,
        apply_masks=True,
        verbose_batch_logs=True,
    )
elif record.operation_type == "db2file":
    result = core_copy.copy_db_to_file(
        source_env=record.source,
        query=record.query,
        file_path=record.target,
        mode=record.mode or "REPLACE",
        batch_size=record.batch_size,
        parallel_workers=record.parallel_workers or 1,
        apply_masks=True,
        verbose_batch_logs=True,
    )
# ... file2db ...
```

### Pattern 5: Parameter Override with Preview
**What:** For `edit`, merge record values with CLI flags (flag wins if provided), then echo resolved parameters before execution.
**When to use:** `edit` command before calling core function.
**Example:**
```python
source = override_source or record.source
target = override_target or record.target
query = override_query or record.query
# ... etc ...

click.echo("Preview of edited operation:")
click.echo(f"  Operation type: {record.operation_type}")
click.echo(f"  Source: {source}")
click.echo(f"  Target: {target}")
# ... etc ...
```

## Anti-Patterns to Avoid
- **Loading entire history.json for every record lookup:** Acceptable here because file is bounded to 100 entries (~10KB). For larger datasets, build an index.
- **Updating JSON without file_lock:** Always acquire `file_lock` around read-modify-write cycles.
- **Using Click `confirmation_prompt` for re-run:** D-04 says execute immediately with no prompt.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON deserialization | custom parser | `json.load` + `HistoryRecord(**item)` | Already used in `history.py` |
| Command dispatch | `if/elif` chain on string | `if/elif` on `operation_type` | Only 3 operation types; dynamic dispatch is overkill |
| Table alignment | manual spaces | `str.ljust`/`rjust` | Simple, no dependencies, sufficient for CLI |
| Record update | rewrite entire file | `_write_records` via `HistoryRecorder` | Reuses existing atomic write pattern |

## Common Pitfalls

### Pitfall 1: Missing ID Handling
**What goes wrong:** `get_record(id)` returns `None`, command proceeds with `None` values causing cryptic downstream errors.
**How to avoid:** Check `if record is None` immediately after lookup, echo to `err=True`, raise `click.ClickException` for non-zero exit.

### Pitfall 2: Record Update Race Condition
**What goes wrong:** Two `re-run` processes on same ID both read, modify, and write; last write wins, losing one update.
**How to avoid:** `update_record` must acquire `file_lock` around the entire read-modify-write cycle (already in Pattern 1).

### Pitfall 3: Unstored Parameters on Re-run
**What goes wrong:** `file_format`, `encryption_key`, `validate_target`, `create_if_not_exists`, `apply_masks` are not stored in `HistoryRecord`. Re-run uses defaults that may differ from original run.
**How to avoid:** Document in help text that re-run uses defaults for unstored parameters; user can use `edit` to override.

### Pitfall 4: Table Output Breaks with Long Values
**What goes wrong:** A 200-character query string breaks fixed-width table formatting.
**How to avoid:** Truncate long fields (e.g., query to 40 chars) or use variable-width columns computed from actual data.

## Code Examples

### Filtered List Output
```python
records = recorder.read_records()
if status:
    records = [r for r in records if r.status == status]
records = sorted(records, key=lambda r: r.date, reverse=True)
if limit > 0:
    records = records[:limit]
```

### Re-run with Update on Success
```python
result = core_copy.copy_db_to_db(...)
if result.success:
    recorder.update_record(
        id,
        last_run_date=datetime.now().isoformat(),
        status="success",
        record_count=result.record_count,
        end_time=datetime.now().isoformat(),
        error_message=None,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| — | Click groups with subcommands | v1.0 | Already established in project |

**Deprecated/outdated:**
- None applicable for this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `history.json` never exceeds 100 entries (FIFO) | Architecture Patterns | Low — bounded by Phase 4 design |
| A2 | `file_lock` works correctly for read+update sequences | Common Pitfalls | Low — same pattern as Phase 4 writes |
| A3 | Manual string formatting is acceptable for table output | Alternatives Considered | Low — CLI users expect simple tables |

## Open Questions

1. **Table column width for queries**
   - What we know: Queries can be very long
   - What's unclear: Exact truncation length or wrapping strategy
   - Recommendation: Truncate query to 40 characters in table view; full query available via `--format json`

## Environment Availability

> Step 2.6: SKIPPED (no external dependencies identified — phase uses Click + stdlib + existing project utilities)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pytest.ini` or `pyproject.toml` |
| Quick run command | `pytest tests/test_history_cli.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CLIC-01 | `copy list` shows formatted table | unit | `pytest tests/test_history_cli.py::test_list_default_output -x` | ❌ Wave 0 |
| CLIC-01 | `copy list --format json` outputs JSON | unit | `pytest tests/test_history_cli.py::test_list_json_format -x` | ❌ Wave 0 |
| CLIC-02 | `copy re-run <id>` executes stored operation | unit | `pytest tests/test_history_cli.py::test_re_run_db2db_success -x` | ❌ Wave 0 |
| CLIC-02 | Re-run updates original record on success | unit | `pytest tests/test_history_cli.py::test_re_run_updates_original -x` | ❌ Wave 0 |
| CLIC-03 | `copy edit <id>` shows preview | unit | `pytest tests/test_history_cli.py::test_edit_preview_shown -x` | ❌ Wave 0 |
| CLIC-03 | Edit creates new history entry | unit | `pytest tests/test_history_cli.py::test_edit_creates_new_record -x` | ❌ Wave 0 |
| CLIC-04 | `copy list --status success --limit 5` filters | unit | `pytest tests/test_history_cli.py::test_list_filter_status -x` | ❌ Wave 0 |

### Wave 0 Gaps
- [ ] `tests/test_history_cli.py` — covers CLIC-01 through CLIC-04

## Security Domain

> Required when `security_enforcement` is enabled (absent = enabled). Omit only if explicitly `false` in config.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Integer ID validation via Click `type=int`; filter values case-insensitive compared |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Missing ID causes downstream errors | Denial of Service | Validate ID exists before dispatch; return non-zero exit |
| Concurrent re-run on same ID | Tampering | `file_lock` around update_record read-modify-write |
| History file read without lock | Information Disclosure/Tampering | `file_lock` on reads if another process may be writing |

## Sources

### Primary (HIGH confidence)
- Click documentation — verified via training knowledge, stable API
- `elm/elm_commands/copy.py` — verified via file read (existing command patterns)
- `elm/core/history.py` — verified via file read (existing HistoryRecorder)

### Secondary (MEDIUM confidence)
- Python `str.ljust`/`rjust` table formatting — stdlib, stable

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Click + stdlib only
- Architecture: HIGH — extends existing project patterns
- Pitfalls: MEDIUM-HIGH — mostly about edge cases in CLI UX

**Research date:** 2026-05-04
**Valid until:** 90 days (stdlib + Click, stable)
