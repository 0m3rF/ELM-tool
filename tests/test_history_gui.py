"""Tests for GUI history panel integration."""

import json
import types
import pytest
from unittest.mock import MagicMock, Mock, patch

from elm.core.history import HistoryRecord
from elm.core.types import OperationResult


@pytest.fixture
def mock_history_with_records(tmp_path):
    """Create a mock history file with 3 records and return config manager."""
    history_file = str(tmp_path / "history.json")
    records = [
        HistoryRecord(
            id=1,
            date="2026-01-01T10:00:00",
            operation_type="db2db",
            source="src_env",
            target="tgt_env",
            query="SELECT * FROM t",
            table="t",
            mode="APPEND",
            batch_size=100,
            parallel_workers=1,
            start_time="2026-01-01T10:00:00",
            end_time="2026-01-01T10:00:01",
            status="success",
            record_count=50,
            error_message=None,
            last_run_date=None,
        ),
        HistoryRecord(
            id=2,
            date="2026-01-02T11:00:00",
            operation_type="db2file",
            source="src_env",
            target="/tmp/out.csv",
            query="SELECT * FROM t2",
            table=None,
            mode="REPLACE",
            batch_size=100,
            parallel_workers=1,
            start_time="2026-01-02T11:00:00",
            end_time="2026-01-02T11:00:01",
            status="failure",
            record_count=None,
            error_message="File not found",
            last_run_date=None,
        ),
        HistoryRecord(
            id=3,
            date="2026-01-03T12:00:00",
            operation_type="file2db",
            source="/tmp/in.csv",
            target="tgt_env",
            query=None,
            table="t3",
            mode="APPEND",
            batch_size=100,
            parallel_workers=1,
            start_time="2026-01-03T12:00:00",
            end_time="2026-01-03T12:00:01",
            status="success",
            record_count=100,
            error_message=None,
            last_run_date=None,
        ),
    ]

    from dataclasses import asdict
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)

    cm = Mock()
    cm.get_history_file.return_value = history_file
    cm.get_config_value.side_effect = lambda key: 100 if key == "history.max_entries" else None
    return cm


class TestHistoryPanelImport:
    def test_history_panel_import(self):
        """Importing HistoryPanel should succeed."""
        from elm.gui.history_panel import HistoryPanel

        assert HistoryPanel is not None


class TestApiWrappers:
    @patch("elm.api.HistoryRecorder")
    def test_api_list_history_returns_list(self, mock_recorder_cls, mock_history_with_records):
        mock_inst = Mock()
        mock_inst.read_records.return_value = [
            HistoryRecord(
                id=1,
                date="2026-01-01T10:00:00",
                operation_type="db2db",
                source="s",
                target="t",
                query="q",
                table="tbl",
                mode="APPEND",
                batch_size=100,
                parallel_workers=1,
                start_time=None,
                end_time=None,
                status="success",
                record_count=10,
                error_message=None,
            ),
        ]
        mock_recorder_cls.return_value = mock_inst

        from elm import api

        result = api.list_history()
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(r, dict) for r in result)
        assert "id" in result[0]
        assert "date" in result[0]
        assert "status" in result[0]

    @patch("elm.api.HistoryRecorder")
    def test_api_get_history_record_found(self, mock_recorder_cls, mock_history_with_records):
        mock_inst = Mock()
        mock_inst.get_record.return_value = HistoryRecord(
            id=1,
            date="2026-01-01T10:00:00",
            operation_type="db2db",
            source="s",
            target="t",
            query="q",
            table="tbl",
            mode="APPEND",
            batch_size=100,
            parallel_workers=1,
            start_time=None,
            end_time=None,
            status="success",
            record_count=10,
            error_message=None,
        )
        mock_recorder_cls.return_value = mock_inst

        from elm import api

        result = api.get_history_record(1)
        assert isinstance(result, dict)
        assert result["id"] == 1

    @patch("elm.api.HistoryRecorder")
    def test_api_get_history_record_not_found(self, mock_recorder_cls):
        mock_inst = Mock()
        mock_inst.get_record.return_value = None
        mock_recorder_cls.return_value = mock_inst

        from elm import api

        result = api.get_history_record(999)
        assert result is None


class TestOperationsPanelIntegration:
    def test_operations_panel_pre_fill_form(self):
        """pre_fill_form should populate form fields from a record dict."""
        from elm.gui.operations_panel import OperationsPanel

        mock_panel = MagicMock()
        mock_panel.source_var = MagicMock()
        mock_panel.target_var = MagicMock()
        mock_panel.query_input = MagicMock()
        mock_panel.table_entry = MagicMock()

        record = {
            "source": "src_env",
            "target": "tgt_env",
            "query": "SELECT 1",
            "table": "t",
        }

        OperationsPanel.pre_fill_form(mock_panel, record)

        mock_panel.source_var.set.assert_called_once_with("src_env")
        mock_panel.target_var.set.assert_called_once_with("tgt_env")
        mock_panel.query_input.delete.assert_called_once_with("1.0", "end")
        mock_panel.query_input.insert.assert_called_once_with("1.0", "SELECT 1")
        mock_panel.table_entry.delete.assert_called_once_with(0, "end")
        mock_panel.table_entry.insert.assert_called_once_with(0, "t")

    @patch("elm.gui.operations_panel.threading.Thread")
    def test_operations_panel_run_history_record(self, mock_thread):
        """run_history_record should start a background thread without form interaction."""
        from elm.gui.operations_panel import OperationsPanel

        mock_panel = MagicMock()
        mock_panel.status_label = MagicMock()
        mock_panel.execute_btn = MagicMock()
        mock_panel.cancel_btn = MagicMock()
        mock_panel._clear_log = MagicMock()
        mock_panel.after = MagicMock()
        mock_panel._copy_worker = MagicMock()
        mock_panel._validate_record_envs = MagicMock(return_value=[])
        mock_panel.log_queue = None
        mock_panel.cancel_event = None
        mock_panel.worker_thread = None

        record = {
            "source": "src_env",
            "target": "tgt_env",
            "query": "SELECT 1",
            "table": "t",
            "operation_type": "db2db",
        }

        OperationsPanel.run_history_record(mock_panel, record)

        mock_panel._validate_record_envs.assert_called_once_with(record)
        mock_panel.execute_btn.configure.assert_called_once_with(state="disabled")
        mock_panel.cancel_btn.configure.assert_called_once_with(state="normal")
        mock_panel._clear_log.assert_called_once()
        mock_thread.assert_called_once()
        assert mock_thread.call_args.kwargs["daemon"] is True


class TestHistoryPanelRefresh:
    @patch("elm.gui.history_panel.api.list_history")
    def test_history_panel_refresh_list_empty(self, mock_list_history):
        """When no records exist, empty_label should be shown."""
        from elm.gui.history_panel import HistoryPanel

        mock_list_history.return_value = []

        mock_hp = MagicMock()
        mock_hp.list_container = MagicMock()
        mock_hp.list_container.winfo_children.return_value = []
        mock_hp.empty_label = MagicMock()
        mock_hp.status_label = MagicMock()
        mock_hp._now = MagicMock(return_value="12:00:00")

        HistoryPanel._refresh_list(mock_hp)

        mock_hp.list_container.pack_forget.assert_called_once()
        mock_hp.empty_label.pack.assert_called_once()
        mock_hp.status_label.configure.assert_called_once()

    @patch("elm.gui.history_panel.ctk.CTkFrame")
    @patch("elm.gui.history_panel.ctk.CTkLabel")
    @patch("elm.gui.history_panel.ctk.CTkButton")
    @patch("elm.gui.history_panel.api.list_history")
    def test_history_panel_refresh_list_populated(self, mock_list_history, mock_btn, mock_label, mock_frame):
        """When records exist, list_container should have row children."""
        from elm.gui.history_panel import HistoryPanel

        mock_list_history.return_value = [
            {"id": 1, "status": "success", "date": "2026-01-01T10:00:00", "operation_type": "db2db", "source": "s", "target": "t", "table": "tbl"},
            {"id": 2, "status": "failure", "date": "2026-01-02T11:00:00", "operation_type": "db2file", "source": "s", "target": "/tmp/out.csv", "table": None},
        ]

        mock_hp = MagicMock()
        mock_hp.list_container = MagicMock()
        mock_hp.list_container.winfo_children.return_value = []
        mock_hp.empty_label = MagicMock()
        mock_hp.status_label = MagicMock()
        mock_hp._now = MagicMock(return_value="12:00:00")
        # Bind the real _build_row so _refresh_list calls actual implementation
        mock_hp._build_row = types.MethodType(HistoryPanel._build_row, mock_hp)

        HistoryPanel._refresh_list(mock_hp)

        mock_hp.empty_label.pack_forget.assert_called_once()
        mock_hp.list_container.pack.assert_called_once()
        # _build_row creates one CTkFrame per record
        assert mock_frame.call_count == 2

    @patch("elm.gui.history_panel.ctk.CTkFrame")
    @patch("elm.gui.history_panel.ctk.CTkLabel")
    @patch("elm.gui.history_panel.ctk.CTkButton")
    def test_history_panel_build_row_status_color(self, mock_btn, mock_label, mock_frame):
        """Success status should render with green text color."""
        from elm.gui.history_panel import HistoryPanel

        mock_hp = MagicMock()
        mock_hp.app = MagicMock()
        mock_hp.list_container = MagicMock()

        record = {
            "id": 1,
            "status": "success",
            "date": "2026-01-01T10:00:00",
            "operation_type": "db2db",
            "source": "s",
            "target": "t",
            "table": "tbl",
        }

        HistoryPanel._build_row(mock_hp, record)

        label_calls = mock_label.call_args_list
        found_green = any(call.kwargs.get("text_color") == "#28A745" for call in label_calls)
        assert found_green, "Expected a label with green text_color for success status"
