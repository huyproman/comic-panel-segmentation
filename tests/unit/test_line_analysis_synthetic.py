"""
Unit tests for line analysis post-processing contracts under v0.2.0.
"""

import numpy as np
import pytest

from comics_panel_extraction.historical_legacy import run_canonical_historical_postprocessing


def test_line_analysis_empty_mask():
    empty_mask = np.zeros((384, 384), dtype=np.uint8)
    out, diag = run_canonical_historical_postprocessing(empty_mask)
    assert out.shape == (384, 384)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_line_analysis_full_white_mask():
    white_mask = np.ones((384, 384), dtype=np.uint8) * 255
    out, diag = run_canonical_historical_postprocessing(white_mask)
    assert out.shape == (384, 384)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})
