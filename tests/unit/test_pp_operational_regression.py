import numpy as np
import pytest
from comics_panel_extraction.postprocessing.line_analysis import (
    PostProcessParameters,
    run_line_segment_postprocessing,
    filter_components_by_size,
)
def test_operational_parameters_frozen():
    params = PostProcessParameters()
    assert params.opening_kernel_size == (20, 20)
    assert params.canny_thresh1 == 50
    assert params.canny_thresh2 == 100
    assert params.hough_rho == 2.0
    assert params.hough_threshold == 25
    assert params.hough_min_line_length == 10.0
    assert params.hough_max_line_gap == 30.0
    assert params.inner_bandwidth == 13.0
    assert params.border_bandwidth == 10.0
    assert params.coordinate_clip_max == 400
    assert params.bresenham_support_threshold == 0.75
    assert params.min_euclidean_distance == 40.0
    assert params.line_color == 0
    assert params.line_thickness == 3
    assert params.size_threshold == 1000
def test_pp_output_range_and_no_overflow():
    raw = np.zeros((448, 448), dtype=np.uint8)
    raw[50:200, 50:200] = 255
    raw[250:400, 250:400] = 255
    out, diag = run_line_segment_postprocessing(raw)
    assert out.shape == (448, 448)
    assert out.dtype == np.uint8
    unique_vals = set(np.unique(out))
    assert unique_vals.issubset({0, 255}), f"Expected subset of {{0, 255}}, got {unique_vals}"
    assert 255 in unique_vals, "Foreground panels must be 255, not 1"
    assert (out == 1).sum() == 0, "Overflow bug detected: found pixels with value 1"
