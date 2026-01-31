import pytest
from click.testing import CliRunner
from unittest.mock import patch


from elm.elm import cli
from elm.core.types import OperationResult


class TestSyncCLICommands:
    def setup_method(self):
        self.runner = CliRunner()

    def test_sync_help_shows_metadata_subcommand(self):
        result = self.runner.invoke(cli, ["sync", "--help"])
        assert result.exit_code == 0
        assert "Synchronization commands" in result.output
        assert "metadata" in result.output

    def test_sync_metadata_dry_run_prints_warnings_and_ddl(self):
        mock_result = OperationResult(
            success=True,
            message="Metadata synchronize plan generated: 1 DDL statement(s), 1 warning(s).",
            data={
                "warnings": ["example warning"],
                "ddl": ["CREATE TABLE demo (id INT)"]
            },
            record_count=1,
        )

        with patch(
            "elm.elm_commands.sync.core_metadata_sync.synchronize_metadata",
            return_value=mock_result,
        ) as mock_sync:
            result = self.runner.invoke(
                cli,
                [
                    "sync",
                    "metadata",
                    "-s",
                    "src",
                    "-t",
                    "tgt",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0
            assert "Warnings:" in result.output
            assert "example warning" in result.output
            assert "DDL statements:" in result.output
            assert "CREATE TABLE demo" in result.output

            mock_sync.assert_called_once()
            kwargs = mock_sync.call_args.kwargs
            assert kwargs["source_env"] == "src"
            assert kwargs["target_env"] == "tgt"
            assert kwargs["dry_run"] is True
            assert kwargs["apply"] is False

    def test_sync_metadata_apply_passes_apply_true(self):
        mock_result = OperationResult(
            success=True,
            message="Metadata synchronized successfully: applied 0 DDL statement(s).",
            data={"warnings": [], "ddl": []},
        )

        with patch(
            "elm.elm_commands.sync.core_metadata_sync.synchronize_metadata",
            return_value=mock_result,
        ) as mock_sync:
            result = self.runner.invoke(
                cli,
                ["sync", "metadata", "-s", "src", "-t", "tgt", "--apply"],
            )

            assert result.exit_code == 0
            kwargs = mock_sync.call_args.kwargs
            assert kwargs["dry_run"] is False
            assert kwargs["apply"] is True
