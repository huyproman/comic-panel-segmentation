import pytest
import os
import cv2
import numpy as np

from comics_panel_extraction.historical_legacy import run_canonical_historical_postprocessing
from comics_panel_extraction.evaluation import evaluate_dataset


def test_table1_oracle_evaluation():
    """Verify Table 1 metrics reproduced exactly on archived raw masks."""
    raw_dir = "comics_analysis/preds/comic_preds/unetpp_comic_pred/pred_mask"
    gt_dir = "comics_analysis/preds/comic_preds/unetpp_comic_pred/true_mask"
    if not os.path.exists(raw_dir) or not os.path.exists(gt_dir):
        pytest.skip("Legacy archives not available")
        
    res = evaluate_dataset(gt_dir, raw_dir)
    assert round(res.mean_iou * 100, 2) == 90.97
    assert round(res.panel_accuracy * 100, 2) == 86.25
    assert round(res.page_accuracy * 100, 2) == 71.62


def test_table2_oracle_evaluation():
    """Verify Table 2 metrics reproduced exactly on archived final masks."""
    final_dir = "comics_analysis/preds/comic_preds/unetpp_comic_pred/final_mask"
    gt_dir = "comics_analysis/preds/comic_preds/unetpp_comic_pred/true_mask"
    if not os.path.exists(final_dir) or not os.path.exists(gt_dir):
        pytest.skip("Legacy archives not available")
        
    res = evaluate_dataset(gt_dir, final_dir)
    assert round(res.mean_iou * 100, 2) == 92.34
    assert round(res.panel_accuracy * 100, 2) == 93.19
    assert round(res.page_accuracy * 100, 2) == 81.08


def test_canonical_historical_sample_page_parity():
    """Verify canonical postprocessing on a sample page matches archived final mask within 1%."""
    raw_path = "comics_analysis/preds/comic_preds/unetpp_comic_pred/pred_mask/pred_mask_raw_3-amazing-spider-man_page_20_jpg.rf.39cca8b7dde4d28a8c19e55865fed3a3.png"
    final_path = "comics_analysis/preds/comic_preds/unetpp_comic_pred/final_mask/pred_mask_raw_3-amazing-spider-man_page_20_jpg.rf.39cca8b7dde4d28a8c19e55865fed3a3.png"
    if not os.path.exists(raw_path):
        pytest.skip("Sample page not available")
        
    raw_m = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    fin_m = cv2.imread(final_path, cv2.IMREAD_GRAYSCALE)
    
    cand_m, _ = run_canonical_historical_postprocessing(raw_m)
    xor_px = np.count_nonzero(cv2.bitwise_xor(cand_m, fin_m))
    assert (xor_px / cand_m.size) < 0.01  # less than 1% disagreement
