"""Unit tests for parity comparator utilities."""

import numpy as np
import pytest
from comics_panel_extraction.utils.parity import (
    compare_arrays_exact,
    compare_binary_masks,
    compare_scalars,
    compute_array_sha256,
)


@pytest.mark.unit
def test_array_exact_match():
    a1 = np.ones((50, 50), dtype=np.uint8) * 255
    a2 = np.ones((50, 50), dtype=np.uint8) * 255
    res = compare_arrays_exact(a1, a2)
    assert res.exact_match is True
    assert res.mismatch_elements == 0
    assert res.mismatch_ratio == 0.0
    assert res.actual_sha256 == res.expected_sha256


@pytest.mark.unit
def test_array_mismatch():
    a1 = np.zeros((10, 10), dtype=np.uint8)
    a2 = np.zeros((10, 10), dtype=np.uint8)
    a2[2, 2] = 255
    res = compare_binary_masks(a1, a2)
    assert res.exact_match is False
    assert res.mismatch_elements == 1
    assert res.actual_foreground_count == 0
    assert res.expected_foreground_count == 1
    assert res.max_absolute_difference == 255.0


@pytest.mark.unit
def test_scalar_comparison():
    res1 = compare_scalars(90.96607, 90.96607, abs_tol=1e-5)
    assert res1.exact_match is True
    assert res1.within_tolerance is True
    
    res2 = compare_scalars(90.96607, 90.1661, abs_tol=1e-3)
    assert res2.exact_match is False
    assert res2.within_tolerance is False
    assert res2.absolute_delta > 0.79
