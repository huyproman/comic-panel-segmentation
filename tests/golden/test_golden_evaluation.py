"""
Golden evaluation tests across the 16 curated fixtures in .local_fixtures.
"""

from pathlib import Path
import pytest

from comics_panel_extraction.evaluation import evaluate_page
from comics_panel_extraction.io.fixture_loader import GoldenFixtureStore


@pytest.mark.golden
def test_golden_evaluation_ready_fixtures(fixture_store: GoldenFixtureStore):
    """Evaluates all evaluation-ready fixtures in .local_fixtures."""
    fixtures = fixture_store.list_fixtures()
    eval_ready = [f for f in fixtures if f.evaluation_ready]
    assert len(eval_ready) == 16  # All 16 fixtures have GT and final_mask available

    tested = 0
    for fix in eval_ready:
        fid = fix.fixture_id
        if not fixture_store.is_available(fid, "final_mask") or not fixture_store.is_available(fid, "ground_truth"):
            continue

        pred_mask = fixture_store.load_mask(fid, "final_mask")
        gt_mask = fixture_store.load_mask(fid, "ground_truth")

        res = evaluate_page(predicted_mask=pred_mask, ground_truth_mask=gt_mask)

        # Baseline checks
        assert res.gt_panel_count > 0
        assert 0.0 <= res.mean_iou <= 1.0
        assert 0.0 <= res.mean_dice <= 1.0
        assert 0.0 <= res.panel_accuracy <= 1.0
        assert 0.0 <= res.precision <= 1.0
        assert 0.0 <= res.recall <= 1.0
        assert 0.0 <= res.f1 <= 1.0
        tested += 1

    if tested == 0:
        pytest.skip("Local fixture bundle not populated in test environment.")

    assert tested == 16
