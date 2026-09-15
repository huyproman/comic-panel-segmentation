import subprocess
import sys
import pytest

@pytest.mark.unit
def test_cli_help():
    res = subprocess.run([sys.executable, "-m", "comics_panel_extraction.cli", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Comic Panel Extraction" in res.stdout

@pytest.mark.unit
def test_cli_infer_requires_checkpoint():
    res = subprocess.run([sys.executable, "-m", "comics_panel_extraction.cli", "infer", "sample.png", "-o", "out.png"], capture_output=True, text=True)
    assert res.returncode != 0
    assert "the following arguments are required: -c/--checkpoint" in res.stderr
