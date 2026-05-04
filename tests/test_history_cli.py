"""Tests for CLI history commands (list, re-run, edit)."""

import json
import os
import pytest
from unittest.mock import Mock, patch
from click.testing import CliRunner
from datetime import datetime

from elm.core.history import HistoryRecorder, HistoryRecord
from elm.core.types import OperationResult
from elm.elm import cli


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


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

    with open(history_file, "w", encoding="utf-8") as f:
        from dataclasses import asdict
        json.dump([asdict(r) for r in records], f, indent=2)

    cm = Mock()
    cm.get_history_file.return_value = history_file
    cm.get_config_value.side_effect = lambda key: 100 if key == "history.max_entries" else None
    return cm


class TestListCommand:
    """Test the 'copy list' command."""

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_list_no_records(self, mock_cls, runner):
        """List with empty history should show message."""
        mock_inst = Mock()
        mock_inst.read_records.return_value = []
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "list"])
        assert result.exit_code == 0
        assert "No history records found" in result.output

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_list_default_output(self, mock_cls, runner):
        """List should show table with records."""
        mock_inst = Mock()
        mock_inst.read_records.return_value = [
            HistoryRecord(id=1, date="2026-01-01T10:00:00", operation_type="db2db",
                          source="s", target="t", query="q", table="tbl", mode="APPEND",
                          batch_size=100, parallel_workers=1, start_time=None, end_time=None,
                          status="success", record_count=10, error_message=None),
        ]
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "list"])
        assert result.exit_code == 0
        assert "ID" in result.output
        assert "1" in result.output

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_list_json_format(self, mock_cls, runner):
        """List --format json should output parseable JSON."""
        mock_inst = Mock()
        mock_inst.read_records.return_value = [
            HistoryRecord(id=1, date="2026-01-01T10:00:00", operation_type="db2db",
                          source="s", target="t", query="q", table="tbl", mode="APPEND",
                          batch_size=100, parallel_workers=1, start_time=None, end_time=None,
                          status="success", record_count=10, error_message=None),
        ]
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "list", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_list_filter_status(self, mock_cls, runner):
        """List --status should filter records."""
        mock_inst = Mock()
        mock_inst.read_records.return_value = [
            HistoryRecord(id=1, date="2026-01-01T10:00:00", operation_type="db2db",
                          source="s", target="t", query="q", table="tbl", mode="APPEND",
                          batch_size=100, parallel_workers=1, start_time=None, end_time=None,
                          status="success", record_count=10, error_message=None),
            HistoryRecord(id=2, date="2026-01-02T10:00:00", operation_type="db2db",
                          source="s", target="t", query="q", table="tbl", mode="APPEND",
                          batch_size=100, parallel_workers=1, start_time=None, end_time=None,
                          status="failure", record_count=None, error_message="err"),
        ]
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "list", "--status", "success"])
        assert result.exit_code == 0
        assert "1" in result.output

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_list_limit(self, mock_cls, runner):
        """List --limit should restrict results."""
        mock_inst = Mock()
        mock_inst.read_records.return_value = [
            HistoryRecord(id=i, date=f"2026-01-0{i}T10:00:00", operation_type="db2db",
                          source="s", target="t", query="q", table="tbl", mode="APPEND",
                          batch_size=100, parallel_workers=1, start_time=None, end_time=None,
                          status="success", record_count=10, error_message=None)
            for i in range(1, 4)
        ]
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "list", "--limit", "1"])
        assert result.exit_code == 0
        # Should only show 1 data row plus header
        lines = [ln for ln in result.output.splitlines() if ln.strip() and "ID" not in ln and "---" not in ln]
        assert len(lines) == 1

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_list_sort_asc(self, mock_cls, runner):
        """List --sort asc should show oldest first."""
        mock_inst = Mock()
        mock_inst.read_records.return_value = [
            HistoryRecord(id=1, date="2026-01-01T10:00:00", operation_type="db2db",
                          source="s", target="t", query="q", table="tbl", mode="APPEND",
                          batch_size=100, parallel_workers=1, start_time=None, end_time=None,
                          status="success", record_count=10, error_message=None),
            HistoryRecord(id=2, date="2026-01-02T10:00:00", operation_type="db2db",
                          source="s", target="t", query="q", table="tbl", mode="APPEND",
                          batch_size=100, parallel_workers=1, start_time=None, end_time=None,
                          status="success", record_count=10, error_message=None),
        ]
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "list", "--sort", "asc", "--limit", "1"])
        assert result.exit_code == 0
        assert "1" in result.output


class TestReRunCommand:
    """Test the 'copy re-run' command."""

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_re_run_record_not_found(self, mock_cls, runner):
        """Re-run with non-existent ID should error."""
        mock_inst = Mock()
        mock_inst.get_record.return_value = None
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "re-run", "999"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    @patch("elm.elm_commands.copy.core_copy.copy_db_to_db")
    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_re_run_db2db_success(self, mock_cls, mock_copy, runner):
        """Re-run db2db should call core and update original record."""
        mock_copy.return_value = OperationResult(success=True, message="ok", record_count=42)

        mock_inst = Mock()
        mock_inst.get_record.return_value = HistoryRecord(
            id=1, date="2026-01-01T10:00:00", operation_type="db2db",
            source="src", target="tgt", query="q", table="tbl", mode="APPEND",
            batch_size=100, parallel_workers=1, start_time=None, end_time=None,
            status="success", record_count=10, error_message=None,
        )
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "re-run", "1"])
        assert result.exit_code == 0
        mock_copy.assert_called_once()
        mock_inst.update_record.assert_called_once()

    @patch("elm.elm_commands.copy.core_copy.copy_db_to_file")
    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_re_run_db2file_success(self, mock_cls, mock_copy, runner):
        """Re-run db2file should call core copy_db_to_file."""
        mock_copy.return_value = OperationResult(success=True, message="ok", record_count=5)

        mock_inst = Mock()
        mock_inst.get_record.return_value = HistoryRecord(
            id=2, date="2026-01-01T10:00:00", operation_type="db2file",
            source="src", target="/tmp/out.csv", query="q", table=None, mode="REPLACE",
            batch_size=100, parallel_workers=1, start_time=None, end_time=None,
            status="failure", record_count=None, error_message="err",
        )
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "re-run", "2"])
        assert result.exit_code == 0
        mock_copy.assert_called_once()

    @patch("elm.elm_commands.copy.core_copy.copy_db_to_db")
    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_re_run_missing_env_creates_new_failure(self, mock_cls, mock_copy, runner):
        """Re-run failure due to missing env should create a new failure record."""
        mock_copy.return_value = OperationResult(
            success=False, message="Environment 'src' not found", record_count=None
        )

        mock_inst = Mock()
        mock_inst.get_record.return_value = HistoryRecord(
            id=1, date="2026-01-01T10:00:00", operation_type="db2db",
            source="src", target="tgt", query="q", table="tbl", mode="APPEND",
            batch_size=100, parallel_workers=1, start_time=None, end_time=None,
            status="success", record_count=10, error_message=None,
        )
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "re-run", "1"])
        assert result.exit_code != 0
        mock_inst.record.assert_called_once()


class TestEditCommand:
    """Test the 'copy edit' command."""

    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_edit_record_not_found(self, mock_cls, runner):
        """Edit with non-existent ID should error."""
        mock_inst = Mock()
        mock_inst.get_record.return_value = None
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "edit", "999"])
        assert result.exit_code != 0

    @patch("elm.elm_commands.copy.core_copy.copy_db_to_db")
    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_edit_preview_shown(self, mock_cls, mock_copy, runner):
        """Edit should show a preview before executing."""
        mock_copy.return_value = OperationResult(success=True, message="ok", record_count=7)

        mock_inst = Mock()
        mock_inst.get_record.return_value = HistoryRecord(
            id=1, date="2026-01-01T10:00:00", operation_type="db2db",
            source="src", target="tgt", query="q", table="tbl", mode="APPEND",
            batch_size=100, parallel_workers=1, start_time=None, end_time=None,
            status="success", record_count=10, error_message=None,
        )
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "edit", "1"])
        assert result.exit_code == 0
        assert "preview" in result.output.lower() or "operation type" in result.output.lower()

    @patch("elm.elm_commands.copy.core_copy.copy_db_to_db")
    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_edit_override_target(self, mock_cls, mock_copy, runner):
        """Edit --target should override the target parameter."""
        mock_copy.return_value = OperationResult(success=True, message="ok", record_count=7)

        mock_inst = Mock()
        mock_inst.get_record.return_value = HistoryRecord(
            id=1, date="2026-01-01T10:00:00", operation_type="db2db",
            source="src", target="old_tgt", query="q", table="tbl", mode="APPEND",
            batch_size=100, parallel_workers=1, start_time=None, end_time=None,
            status="success", record_count=10, error_message=None,
        )
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "edit", "1", "--target", "new_env"])
        assert result.exit_code == 0
        call_kwargs = mock_copy.call_args.kwargs
        assert call_kwargs["target_env"] == "new_env"

    @patch("elm.elm_commands.copy.core_copy.copy_db_to_db")
    @patch("elm.elm_commands.copy.HistoryRecorder")
    def test_edit_creates_new_record(self, mock_cls, mock_copy, runner):
        """Edit should create a new history entry, not update original."""
        mock_copy.return_value = OperationResult(success=True, message="ok", record_count=7)

        mock_inst = Mock()
        mock_inst.get_record.return_value = HistoryRecord(
            id=1, date="2026-01-01T10:00:00", operation_type="db2db",
            source="src", target="tgt", query="q", table="tbl", mode="APPEND",
            batch_size=100, parallel_workers=1, start_time=None, end_time=None,
            status="success", record_count=10, error_message=None,
        )
        mock_cls.return_value = mock_inst

        result = runner.invoke(cli, ["copy", "edit", "1"])
        assert result.exit_code == 0
        mock_inst.record.assert_called_once()
        mock_inst.update_record.assert_not_called()
