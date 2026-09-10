"""
CLI smoke tests for comics-panel-extraction v0.2.0.
"""

from unittest.mock import patch
import pytest

from comics_panel_extraction.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "comics-panel-extraction" in captured.out
    assert "infer" in captured.out
    assert "postprocess" in captured.out
    assert "evaluate" in captured.out
    assert "historical-replay" in captured.out


def test_cli_no_subcommand(capsys):
    code = main([])
    assert code == 1
    captured = capsys.readouterr()
    assert "comics-panel-extraction" in captured.out
