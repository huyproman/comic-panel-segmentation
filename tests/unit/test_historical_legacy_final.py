import pytest
import numpy as np
import cv2

from comics_panel_extraction.historical_legacy.pipeline import (
    LegacyLine,
    is_connection_valid_historical,
    run_canonical_historical_postprocessing
)
from comics_panel_extraction.historical_legacy.config import HistoricalLegacyConfig


def test_reference_geometry_fixed_384():
    """Verify LegacyLine reference points evaluate to exact 384-geometry coordinates."""
    line = LegacyLine(10, 10, 100, 10)
    p1 = (384 * (5/7), 384 * 0.5)
    p2 = (384 / 6, 384 * (6/7))
    p3 = (384 * (3.5/9), 384 * (1/3))
    
    # Distance from horizontal line y=10 to point (x0, y0) is |10 - y0|
    assert abs(line.list_dis[0] - abs(10 - p1[1])) < 1e-5
    assert abs(line.list_dis[1] - abs(10 - p2[1])) < 1e-5
    assert abs(line.list_dis[2] - abs(10 - p3[1])) < 1e-5


def test_grouping_constants():
    """Verify config exposes exact 8.5 px and 3.0 deg."""
    cfg = HistoricalLegacyConfig()
    assert cfg.grouping.distance_tolerance == 8.5
    assert cfg.grouping.angle_tolerance == 3.0


def test_bresenham_validation():
    """Verify Bresenham validation enforces 0.75 support."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    canvas[0, 0:75] = 255
    assert is_connection_valid_historical(canvas, 0, 0, 99, 0, threshold=0.75)
    canvas[0, 74] = 0
    assert not is_connection_valid_historical(canvas, 0, 0, 99, 0, threshold=0.75)


def test_canonical_pipeline_runs_deterministic():
    """Verify canonical historical pipeline executes deterministically."""
    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[50:200, 50:200] = 255
    mask[250:400, 50:200] = 255
    out1, diag1 = run_canonical_historical_postprocessing(mask)
    out2, diag2 = run_canonical_historical_postprocessing(mask)
    assert np.array_equal(out1, out2)
