import pytest
import subprocess
import json
from pathlib import Path


def test_cli_historical_replay_execution():
    """Verify CLI historical-replay executes successfully on sample directory."""
    sample_dir = Path("/tmp/comics_cli_sample_test")
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path("/tmp/comics_cli_sample_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use synthetic mask or sample mask
    import cv2
    import numpy as np
    sample_mask = np.zeros((384, 384), dtype=np.uint8)
    sample_mask[50:150, 50:150] = 255
    cv2.imwrite(str(sample_dir / "sample.png"), sample_mask)
    
    cmd = [
        "/tmp/p1_test_venv/bin/python", "-m", "comics_panel_extraction",
        "historical-replay",
        "--input-dir", str(sample_dir),
        "--output-dir", str(out_dir)
    ]
    env = {"PYTHONPATH": "comics_panel_extraction/src"}
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0
    assert (out_dir / "sample.png").exists()
