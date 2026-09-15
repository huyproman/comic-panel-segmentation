import numpy as np
import pytest
from comics_panel_extraction.postprocessing import run_line_segment_postprocessing

@pytest.mark.unit
def test_line_analysis_empty_mask():
    empty_mask = np.zeros((448, 448), dtype=np.uint8)
    out, diag = run_line_segment_postprocessing(empty_mask)
    assert out.shape == (448, 448)
    assert np.all(out == 0)

@pytest.mark.unit
def test_line_analysis_full_white_mask():
    white_mask = np.ones((448, 448), dtype=np.uint8) * 255
    out, diag = run_line_segment_postprocessing(white_mask)
    assert out.shape == (448, 448)
    assert np.all(out == 255)
