"""
ELM Tool Core History Recording Module

Persistent, thread-safe copy operation history storage.
"""

import os
import json
import shutil
import glob
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, asdict

from elm.core.types import OperationResult
from elm.core.config import get_config_manager, ConfigManager
from elm.elm_utils.file_lock import file_lock


@dataclass
class HistoryRecord:
    """Schema for a single history entry."""
    id: int
    date: str
    operation_type: str
    source: Optional[str]
    target: Optional[str]
    query: Optional[str]
    table: Optional[str]
    mode: Optional[str]
    batch_size: Optional[int]
    parallel_workers: Optional[int]
    start_time: Optional[str]
    end_time: Optional[str]
    status: Literal["success", "failure"]
    record_count: Optional[int]
    error_message: Optional[str]
    last_run_date: Optional[str] = None


class HistoryRecorder:
    """Records copy operations to a persistent JSON history file."""

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or get_config_manager()
        self._cleanup_old_backups()

    def record(self, result: OperationResult, params: Dict[str, Any]) -> bool:
        """Append a history record. Returns True on success."""
        try:
            history_file = self.config.get_history_file()
            max_entries = self.config.get_config_value("history.max_entries") or 100

            with file_lock(history_file):
                records = self._read_records(history_file)

                next_id = max((r.id for r in records), default=0) + 1
                record = HistoryRecord(
                    id=next_id,
                    date=datetime.now().isoformat(),
                    operation_type=params.get("operation_type", "unknown"),
                    source=params.get("source_env"),
                    target=params.get("target_env") or params.get("file_path"),
                    query=params.get("query"),
                    table=params.get("table"),
                    mode=params.get("mode"),
                    batch_size=params.get("batch_size"),
                    parallel_workers=params.get("parallel_workers"),
                    start_time=params.get("start_time") or datetime.now().isoformat(),
                    end_time=params.get("end_time") or datetime.now().isoformat(),
                    status="success" if result.success else "failure",
                    record_count=result.record_count,
                    error_message=result.error_details if not result.success else None,
                )

                records.append(record)
                records = records[-max_entries:]

                backup_path = self._create_backup(history_file)
                try:
                    self._write_records(history_file, records)
                    self._verify_and_cleanup(history_file, backup_path)
                except Exception:
                    self._restore_from_backup(history_file, backup_path)
                    return False

            return True
        except Exception:
            return False

    def _read_records(self, history_file: str) -> List[HistoryRecord]:
        """Read existing records from history file. Returns [] if missing."""
        if not os.path.exists(history_file):
            return []
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [HistoryRecord(**item) for item in data]

    def _write_records(self, history_file: str, records: List[HistoryRecord]) -> None:
        """Write records to history file."""
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)

    def _create_backup(self, file_path: str) -> Optional[str]:
        """Create a .bak copy if file exists. Returns backup path or None."""
        if os.path.exists(file_path):
            backup = f"{file_path}.bak"
            shutil.copy2(file_path, backup)
            return backup
        return None

    def _verify_and_cleanup(self, file_path: str, backup_path: Optional[str]) -> None:
        """Verify JSON parses and remove backup on success."""
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
        if backup_path and os.path.exists(backup_path):
            os.remove(backup_path)

    def _restore_from_backup(self, file_path: str, backup_path: Optional[str]) -> None:
        """Restore original file from backup if available."""
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)

    def _cleanup_old_backups(self) -> None:
        """Delete .bak files older than 24h in the same directory as history file."""
        try:
            history_file = self.config.get_history_file()
            backup_dir = os.path.dirname(history_file) or "."
            cutoff = datetime.now() - timedelta(hours=24)
            for bak in glob.glob(os.path.join(backup_dir, "*.bak")):
                mtime = datetime.fromtimestamp(os.path.getmtime(bak))
                if mtime < cutoff:
                    os.remove(bak)
        except Exception:
            pass

    def read_records(self) -> List[HistoryRecord]:
        """Read all history records."""
        return self._read_records(self.config.get_history_file())

    def get_record(self, record_id: int) -> Optional[HistoryRecord]:
        """Get a single history record by ID."""
        for record in self.read_records():
            if record.id == record_id:
                return record
        return None

    def update_record(self, record_id: int, **kwargs) -> bool:
        """Update fields on an existing history record. Returns True on success."""
        try:
            history_file = self.config.get_history_file()
            with file_lock(history_file):
                records = self._read_records(history_file)
                for record in records:
                    if record.id == record_id:
                        for key, value in kwargs.items():
                            if hasattr(record, key):
                                setattr(record, key, value)
                        self._write_records(history_file, records)
                        return True
            return False
        except Exception:
            return False
