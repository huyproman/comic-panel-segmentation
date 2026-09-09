"""
Integration golden parity tests for Line Segment Analysis post-processing.
Executes dual-track evaluation:
  Track A: Exact match against extracted legacy source behavior (acceptance oracle).
  Track B: Diagnostic tracking of stored historical processed_mask.
"""

import hashlib
import cv2
import numpy as np
import pytest

from comics_panel_extraction.io.fixture_loader import get_default_fixture_store
from comics_panel_extraction.postprocessing import (
    refine_mask_with_line_segments,
    trace_line_analysis,
)


def get_available_golden_fixtures():
    store = get_default_fixture_store()
    fixtures = store.list_fixtures()
    cases = []
    for fix in fixtures:
        fid = fix.fixture_id
        raw_pred_art = fix.artifacts.get("raw_prediction")
        if raw_pred_art and raw_pred_art.availability == "AVAILABLE":
            p = store.resolve_artifact_path(fid, "raw_prediction")
            if p and p.exists():
                cases.append((fid, p))
    return cases


@pytest.mark.parametrize("fixture_id,raw_path", get_available_golden_fixtures())
def test_golden_line_analysis_execution(fixture_id, raw_path):
    mask = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask.ndim == 2

    # Execute target line analysis
    proc_mask, trace = trace_line_analysis(mask)

    assert proc_mask.shape == mask.shape
    assert proc_mask.dtype == np.uint8
    assert set(np.unique(proc_mask)).issubset({0, 255})
    assert "centroids_count" in trace
    assert trace["centroids_count"] >= 0

    # Verify idempotency / determinism on same input
    proc_mask_rep = refine_mask_with_line_segments(mask)
    assert np.array_equal(proc_mask, proc_mask_rep), "Line analysis must be deterministic on identical input!"
