# Phase 04: Storage Layer & Recording - Research

**Researched:** 2026-05-04
**Domain:** Python file persistence, JSON storage, thread-safe file locking
**Confidence:** HIGH

## Summary

Phase 4 requires implementing persistent storage that records every copy operation with full metadata. The domain is straightforward: Python standard library (`json`, `datetime`, `os`) with existing project patterns. No external dependencies needed.

Key constraints from CONTEXT.md:
- Co-located JSON file (`history.json`) in `ELM_TOOL_HOME`
- Thread-safe via existing `file_lock` context manager
- FIFO bounded queue (default 100 entries)
- Backup-before-write with verification
- Must never block the copy operation on history failure

**Primary recommendation:** Implement a thin `HistoryRecorder` class in a new `elm/core/history.py` module, hooking into existing `copy.py` functions after `OperationResult` is constructed.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| History recording | Core (`elm/core/history.py`) | — | Data persistence belongs in core |
| JSON file I/O | Core (`elm/core/history.py`) | — | Simple standard library usage |
| Thread-safe access | Core (`elm/core/history.py`) | `elm_utils/file_lock.py` | Reuse existing lock pattern |
| Config path resolution | Core (`elm/core/config.py`) | — | `ConfigManager` already owns paths |
| Copy operation hooks | Core (`elm/core/copy.py`) | — | Recording at end of each copy function |
| Result enrichment | Core (`elm/core/types.py`) | — | Add `history_saved` to `OperationResult` |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `json` (stdlib) | 3.12 | JSON serialization | Human-readable, no deps, project already uses JSON |
| `datetime` (stdlib) | 3.12 | ISO-8601 timestamps | Already used throughout `copy.py` |
| `os` (stdlib) | 3.12 | File existence, backup | Standard Python file operations |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `file_lock` (project) | existing | Cross-process lock | All reads/writes to `history.json` |
| `ConfigManager` (project) | existing | Path resolution | `get_history_file()`, `history.max_entries` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Flat JSON array | SQLite | Overkill for 100-entry FIFO; adds schema migration burden |
| Flat JSON array | pickle | Not human-readable, not inspectable by users |
| `json` | `orjson` | Faster but adds dependency; 100-entry writes are trivial |

## Architecture Patterns

### Pattern 1: HistoryRecorder Class
**What:** A stateless recorder class with a single public method `record(operation_result, copy_params)`.
**When to use:** All copy operations (`copy_db_to_db`, `copy_db_to_file`, `copy_file_to_db`).
**Example:**
```python
# Source: elm/core/history.py (new)
class HistoryRecorder:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or get_config_manager()
    
    def record(self, result: OperationResult, params: Dict[str, Any]) -> bool:
        """Append a history record. Returns True on success."""
        try:
            history_file = self.config.get_history_file()
            max_entries = self.config.get_config_value("history.max_entries") or 100
            # ... read, append, trim, write under lock
            return True
        except Exception:
            return False
```

### Pattern 2: Backup-Before-Write
**What:** Create `history.json.bak`, write new file, verify parse, delete backup.
**When to use:** Every write to `history.json`.
**Example:**
```python
# Source: project-specific pattern (new)
def _atomic_write_json(file_path: str, data: List[Dict]) -> None:
    backup_path = f"{file_path}.bak"
    # Create backup if file exists
    if os.path.exists(file_path):
        shutil.copy2(file_path, backup_path)
    # Write new content
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Verify
    with open(file_path, 'r', encoding='utf-8') as f:
        json.load(f)  # Will raise if corrupt
    # Remove backup on success
    if os.path.exists(backup_path):
        os.remove(backup_path)
```

### Pattern 3: Non-Blocking History Hook
**What:** Wrap history recording in `try/except`; never re-raise.
**When to use:** End of every public copy function.
**Example:**
```python
# Source: elm/core/copy.py
result = create_success_result(...)
# Hook:
try:
    recorder = HistoryRecorder()
    history_ok = recorder.record(result, locals_dict)
    result.history_saved = history_ok
except Exception:
    result.history_saved = False
return result
```

### Anti-Patterns to Avoid
- **Reading JSON without lock:** Even reads need `file_lock` if another process may be writing.
- **Growing unbounded list:** Must slice `records[-max_entries:]` before write.
- **Letting history failure bubble up:** Copy operation must succeed even if history fails.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON serialization | custom text format | `json` stdlib | Already used in project; human-readable |
| File locking | `threading.Lock` | `file_lock` context manager | Existing project pattern; cross-process safe |
| Timestamp formatting | manual string concat | `datetime.now().isoformat()` | ISO-8601 standard, one-liner |
| Config path resolution | hardcoded paths | `ConfigManager.get_*_file()` | Existing project pattern; respects `ELM_TOOL_HOME` |

## Common Pitfalls

### Pitfall 1: Concurrent Write Corruption
**What goes wrong:** Two parallel `elm copy` processes write to `history.json` simultaneously; one overwrites the other's changes.
**Why it happens:** No exclusive lock during read-modify-write cycle.
**How to avoid:** Use `file_lock` around the entire read→parse→append→trim→write sequence.
**Warning signs:** Missing history entries after parallel copy operations.

### Pitfall 2: Unbounded File Growth
**What goes wrong:** `history.json` grows indefinitely, slowing reads and consuming disk space.
**Why it happens:** Forgetting to slice records before write.
**How to avoid:** After appending, `records = records[-max_entries:]`.
**Warning signs:** `history.json` exceeds 1MB.

### Pitfall 3: Backup File Left Behind
**What goes wrong:** Crash between write and backup deletion leaves stale `.bak` files accumulating.
**Why it happens:** Exception or process kill before cleanup.
**How to avoid:** On startup, clean up any `.bak` files older than 24h. Also clean before each write.
**Warning signs:** Multiple `.bak` files in `ELM_TOOL_HOME`.

### Pitfall 4: Duplicate IDs Under Lock Timeout
**What goes wrong:** Lock timeout causes fallback; two processes both read max id=5, both write id=6.
**Why it happens:** `file_lock` raises `TimeoutError` on lock failure; if caught and operation continues without lock, race condition occurs.
**How to avoid:** Let `TimeoutError` propagate (fail the history write), do NOT proceed without lock. Return `history_saved=False`.
**Warning signs:** Two history records with same `id`.

## Code Examples

### History Record Schema
```python
# Source: project-specific (from CONTEXT.md D-06)
{
    "id": 1,
    "date": "2026-05-04T14:30:00",
    "operation_type": "db2db",
    "source": "prod",
    "target": "staging",
    "query": "SELECT * FROM users",
    "table": "users",
    "mode": "APPEND",
    "batch_size": 1000,
    "parallel_workers": 1,
    "start_time": "2026-05-04T14:30:00",
    "end_time": "2026-05-04T14:31:23",
    "status": "success",
    "record_count": 15000,
    "error_message": None
}
```

### ConfigManager Extension
```python
# Source: elm/core/config.py (existing pattern)
def get_history_file(self) -> str:
    """Get the history file path."""
    return os.path.join(self.get_elm_tool_home(), "history.json")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| — | Standard library JSON | N/A (new) | No dependencies, human-readable |

**Deprecated/outdated:**
- None applicable for this phase.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `json.dump(..., indent=2)` is sufficient performance for 100-entry array | Standard Stack | Low — writes are <1ms |
| A2 | `file_lock` (O_CREAT\|O_EXCL) works correctly on Windows + Python 3.12 | Architecture Patterns | Medium — test on CI; if fails, use `portalocker` or `fasteners` |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

## Open Questions

1. **Backup cleanup policy**
   - What we know: CONTEXT.md says create `.bak`, verify, delete on success
   - What's unclear: Should stale `.bak` files be cleaned on module import?
   - Recommendation: Add `_cleanup_old_backups()` called once per `HistoryRecorder` init; delete `.bak` files older than 24 hours

## Environment Availability

> Step 2.6: SKIPPED (no external dependencies identified — phase uses stdlib + existing project utilities)

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | `pytest.ini` or `pyproject.toml` |
| Quick run command | `pytest tests/test_history.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STOR-01 | Copy operation creates history record | unit | `pytest tests/test_history.py::test_record_created -x` | ❌ Wave 0 |
| STOR-02 | History file is valid JSON in ELM_TOOL_HOME | unit | `pytest tests/test_history.py::test_file_location -x` | ❌ Wave 0 |
| STOR-03 | Each record has unique incrementing id | unit | `pytest tests/test_history.py::test_unique_ids -x` | ❌ Wave 0 |
| STOR-04 | Record contains all required fields | unit | `pytest tests/test_history.py::test_record_schema -x` | ❌ Wave 0 |
| STOR-05 | FIFO eviction at 100 entries | unit | `pytest tests/test_history.py::test_fifo_eviction -x` | ❌ Wave 0 |

### Wave 0 Gaps
- [ ] `tests/test_history.py` — covers STOR-01 through STOR-05
- [ ] `tests/conftest.py` — may need shared `HistoryRecorder` fixture

## Security Domain

> Required when `security_enforcement` is enabled (absent = enabled). Omit only if explicitly `false` in config.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Validate JSON structure before write; sanitize error_message string |
| V6 Cryptography | no | — |

### Known Threat Patterns for JSON file storage

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal in file path | Tampering | `ConfigManager.get_history_file()` resolves to controlled `ELM_TOOL_HOME` |
| JSON injection via error_message | Tampering | Ensure `error_message` is a plain string, never executed |
| Backup file disclosure | Information Disclosure | `.bak` files in same directory; acceptable for local tool |

## Sources

### Primary (HIGH confidence)
- Python 3.12 `json` module docs — verified via training knowledge, stdlib behavior is stable
- `elm/elm_utils/file_lock.py` — verified via file read (O_CREAT\|O_EXCL pattern)
- `elm/core/config.py` — verified via file read (existing `get_envs_file()`, `get_mask_file()`)

### Secondary (MEDIUM confidence)
- Python `shutil.copy2` behavior for backup — stdlib, stable since 2.3

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — stdlib only
- Architecture: HIGH — pattern follows existing project conventions
- Pitfalls: MEDIUM-HIGH — based on common file I/O race conditions, not project-specific issues

**Research date:** 2026-05-04
**Valid until:** 90 days (stdlib, stable)
