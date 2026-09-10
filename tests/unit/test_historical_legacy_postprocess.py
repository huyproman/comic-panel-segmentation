import pytest
import numpy as np
import cv2

from comics_panel_extraction.historical_legacy.pipeline import (
    is_connection_valid_historical,
    run_canonical_historical_postprocessing
)
from comics_panel_extraction.historical_legacy.config import HistoricalLegacyConfig


def test_bresenham_support_threshold_historical():
    """Verify is_connection_valid enforces 0.75 support."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    # Line length 100 px: (0, 0) to (99, 0)
    canvas[0, 0:74] = 255
    assert not is_connection_valid_historical(canvas, 0, 0, 99, 0, threshold=0.75)
    
    canvas[0, 74] = 255
    assert is_connection_valid_historical(canvas, 0, 0, 99, 0, threshold=0.75)


def test_canonical_pipeline_runs():
    """Smoke test running full pipeline on synthetic mask."""
    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[50:200, 50:200] = 255
    mask[250:400, 50:200] = 255
    
    out, diag = run_canonical_historical_postprocessing(mask)
    assert out.shape == (448, 448)
    assert "centroids_count" in diag
