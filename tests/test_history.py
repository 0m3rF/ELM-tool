"""Tests for the history recording module."""

import json
import os
import pytest
from unittest.mock import Mock, patch

from elm.core.history import HistoryRecorder, HistoryRecord
from elm.core.types import OperationResult


@pytest.fixture
def mock_config_manager(tmp_path):
    """Create a mock ConfigManager that points to a temp history file."""
    history_file = str(tmp_path / "history.json")
    cm = Mock()
    cm.get_history_file.return_value = history_file
    cm.get_config_value.side_effect = lambda key: 100 if key == "history.max_entries" else None
    return cm


class TestHistoryRecorder:
    """Test cases for HistoryRecorder."""

    def test_record_file_created(self, tmp_path, mock_config_manager):
        """Recording should create the history JSON file."""
        recorder = HistoryRecorder(config_manager=mock_config_manager)
        result = OperationResult(success=True, message="ok", record_count=5)
        params = {
            "operation_type": "db2db",
            "source_env": "src",
            "target_env": "tgt",
            "query": "SELECT 1",
            "table": "t",
            "mode": "APPEND",
            "batch_size": 100,
            "parallel_workers": 1,
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T00:00:01",
        }
        assert recorder.record(result, params) is True
        assert os.path.exists(mock_config_manager.get_history_file())

    def test_record_has_unique_incrementing_id(self, tmp_path, mock_config_manager):
        """Sequential records should get IDs 1, 2, 3."""
        recorder = HistoryRecorder(config_manager=mock_config_manager)
        for i in range(3):
            result = OperationResult(success=True, message="ok", record_count=i)
            params = {
                "operation_type": "db2db",
                "source_env": "src",
                "target_env": "tgt",
                "start_time": "2026-01-01T00:00:00",
                "end_time": "2026-01-01T00:00:01",
            }
            assert recorder.record(result, params) is True

        with open(mock_config_manager.get_history_file(), "r", encoding="utf-8") as f:
            records = json.load(f)
        ids = [r["id"] for r in records]
        assert ids == [1, 2, 3]

    def test_record_schema_fields(self, tmp_path, mock_config_manager):
        """Each record must contain all required schema fields."""
        recorder = HistoryRecorder(config_manager=mock_config_manager)
        result = OperationResult(success=True, message="ok", record_count=10)
        params = {
            "operation_type": "db2db",
            "source_env": "src",
            "target_env": "tgt",
            "query": "SELECT * FROM t",
            "table": "t",
            "mode": "APPEND",
            "batch_size": 100,
            "parallel_workers": 2,
            "start_time": "2026-01-01T00:00:00",
            "end_time": "2026-01-01T00:00:01",
        }
        assert recorder.record(result, params) is True

        with open(mock_config_manager.get_history_file(), "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == 1
        record = records[0]
        required_fields = [
            "id", "date", "operation_type", "source", "target", "query",
            "table", "mode", "batch_size", "parallel_workers",
            "start_time", "end_time", "status", "record_count", "error_message",
        ]
        for field in required_fields:
            assert field in record, f"Missing field: {field}"

    def test_fifo_eviction(self, tmp_path, mock_config_manager):
        """When max_entries is exceeded, oldest records should be evicted."""
        mock_config_manager.get_config_value.side_effect = lambda key: 3 if key == "history.max_entries" else None
        recorder = HistoryRecorder(config_manager=mock_config_manager)
        for i in range(5):
            result = OperationResult(success=True, message="ok", record_count=i)
            params = {
                "operation_type": "db2db",
                "source_env": "src",
                "target_env": "tgt",
                "start_time": "2026-01-01T00:00:00",
                "end_time": "2026-01-01T00:00:01",
            }
            assert recorder.record(result, params) is True

        with open(mock_config_manager.get_history_file(), "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) == 3
        ids = [r["id"] for r in records]
        assert ids == [3, 4, 5]

    def test_record_returns_bool(self, tmp_path, mock_config_manager):
        """record() must always return a bool."""
        recorder = HistoryRecorder(config_manager=mock_config_manager)
        result = OperationResult(success=True, message="ok")
        params = {"operation_type": "db2db", "source_env": "s", "target_env": "t"}
        retval = recorder.record(result, params)
        assert isinstance(retval, bool)

    def test_record_on_failure_still_returns_result(self, tmp_path, mock_config_manager):
        """If writing fails, record() should return False without raising."""
        recorder = HistoryRecorder(config_manager=mock_config_manager)
        result = OperationResult(success=True, message="ok")
        params = {"operation_type": "db2db", "source_env": "s", "target_env": "t"}

        with patch("elm.core.history.json.dump", side_effect=Exception("disk full")):
            retval = recorder.record(result, params)
        assert retval is False
