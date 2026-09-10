"""
Integration tests for CLI subcommands in v0.2.0.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from comics_panel_extraction.cli import main


@pytest.fixture
def temp_io_paths(tmp_path):
    in_img = tmp_path / "test_input.npy"
    np.save(in_img, np.zeros((384, 384, 3), dtype=np.uint8))
    out_prob = tmp_path / "out_prob.npy"
    return in_img, out_prob


def test_cli_infer_invocation(temp_io_paths):
    in_img, out_prob = temp_io_paths

    mock_prob = np.zeros((384, 384), dtype=np.float32)
    mock_model = MagicMock()

    with patch("comics_panel_extraction.cli.load_inference_model", return_value=mock_model), \
         patch("comics_panel_extraction.cli.predict_probability_mask", return_value=mock_prob):
        exit_code = main(["infer", str(in_img), "-o", str(out_prob)])

    assert exit_code == 0
    assert out_prob.exists()
    loaded = np.load(out_prob)
    assert loaded.shape == (384, 384)


def test_cli_evaluate_invocation(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    import cv2
    dummy_mask = np.zeros((448, 448), dtype=np.uint8)
    cv2.imwrite(str(gt_dir / "page_1.png"), dummy_mask)
    cv2.imwrite(str(pred_dir / "page_1.png"), dummy_mask)

    out_json = tmp_path / "metrics.json"

    exit_code = main([
        "evaluate",
        "-g", str(gt_dir),
        "-p", str(pred_dir),
        "-o", str(out_json),
    ])

    assert exit_code == 0
    assert out_json.exists()
