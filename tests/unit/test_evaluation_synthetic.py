"""
Comprehensive synthetic and adversarial unit test suite for Evaluation Subsystem.

Covers:
- Case 1: Perfect match
- Case 2: No overlap
- Case 3: Partial overlap
- Case 4: Extra false positive (demonstrates Panel Acc == 1.0, Page Acc == 1.0, Precision < 1.0)
- Case 5: Missing panel (1 of 2 detected)
- Case 6: Merged prediction (one large predicted blob covering two GT panels)
- Case 7: Multiple predictions around one GT panel
- Case 8: Both empty
- Case 9: GT nonempty / Pred empty
- Case 10: GT empty / Pred nonempty
- Case 11: Diagonal-touch connectivity (8-way CCL semantics)
- Case 12: Dtype representations ({0, 1} vs {0, 255})
- Case 13: Dice threshold exactly 0.9 boundary
- Case 14: Dice threshold just below 0.9
- Case 15: Dice threshold just above 0.9
"""

import numpy as np
import pytest

from comics_panel_extraction.evaluation import (
    calculate_dice,
    calculate_iou,
    calculate_mean_dice,
    calculate_mean_iou,
    calculate_panel_accuracy,
    calculate_precision_recall_f1,
    evaluate_page,
    extract_components,
)


@pytest.mark.unit
def test_case_01_perfect_match():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:40, 10:40] = 255
    gt[50:80, 50:80] = 255
    pred = gt.copy()

    res = evaluate_page(pred, gt)
    assert res.mean_iou == 1.0
    assert res.mean_dice == 1.0
    assert res.panel_accuracy == 1.0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.page_correct is True
    assert res.gt_panel_count == 2
    assert res.pred_panel_count == 2


@pytest.mark.unit
def test_case_02_no_overlap():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[60:80, 60:80] = 255

    res = evaluate_page(pred, gt)
    assert res.mean_iou == 0.0
    assert res.mean_dice == 0.0
    assert res.panel_accuracy == 0.0
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    assert res.page_correct is False


@pytest.mark.unit
def test_case_03_partial_overlap():
    # Box of 20x20 = 400 pixels
    b1 = np.zeros((50, 50), dtype=bool)
    b1[10:30, 10:30] = True

    # Box shifted by 10 pixels: overlap is 10x20 = 200 pixels
    b2 = np.zeros((50, 50), dtype=bool)
    b2[10:30, 20:40] = True

    # Intersection = 200, Union = 400 + 400 - 200 = 600 -> IoU = 1/3
    # Dice = 2 * 200 / 800 = 0.5
    iou = calculate_iou(b2, b1)
    dice = calculate_dice(b2, b1)
    assert abs(iou - (1.0 / 3.0)) < 1e-6
    assert abs(dice - 0.5) < 1e-6


@pytest.mark.unit
def test_case_04_extra_false_positive():
    """
    CRITICAL ADVERSARIAL CASE:
    Extra false positive panel prediction.
    Proves:
      - Panel Accuracy is unpenalized (remains 1.0)
      - Page Accuracy is unpenalized (remains 1.0 / True)
      - Precision is penalized (0.5)
      - Recall remains 1.0
    """
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:40, 10:40] = 255  # 1 GT panel

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:40, 10:40] = 255  # Correct match (Dice=1.0)
    pred[60:90, 60:90] = 255  # Extra False Positive panel

    res = evaluate_page(pred, gt)
    assert res.gt_panel_count == 1
    assert res.pred_panel_count == 2
    assert res.panel_accuracy == 1.0
    assert res.recall == 1.0
    assert res.precision == 0.5
    assert abs(res.f1 - (2.0 * 0.5 * 1.0 / 1.5)) < 1e-6
    assert res.page_correct is True  # Page Accuracy completely ignores false positives


@pytest.mark.unit
def test_case_05_missing_panel():
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255  # GT 1
    gt[50:70, 50:70] = 255  # GT 2

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:30, 10:30] = 255  # Pred 1 matches GT 1, GT 2 is missed

    res = evaluate_page(pred, gt)
    assert res.gt_panel_count == 2
    assert res.pred_panel_count == 1
    assert res.panel_accuracy == 0.5
    assert res.recall == 0.5
    assert res.precision == 1.0
    assert abs(res.f1 - (2.0 * 1.0 * 0.5 / 1.5)) < 1e-6
    assert res.page_correct is False


@pytest.mark.unit
def test_case_06_merged_prediction():
    """Single predicted blob covering 2 GT panels."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255  # 400 px
    gt[10:30, 50:70] = 255  # 400 px

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:30, 10:70] = 255  # 1200 px (bridges both)

    res = evaluate_page(pred, gt)
    # Dice with each GT is 2 * 400 / (1200 + 400) = 800 / 1600 = 0.5 < 0.9
    assert res.panel_accuracy == 0.0
    assert res.recall == 0.0
    assert res.precision == 0.0
    assert res.page_correct is False


@pytest.mark.unit
def test_case_07_multiple_predictions_around_one_gt():
    """Matching direction semantics: GT -> best Pred."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:50, 10:50] = 255  # 1600 px

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:50, 10:50] = 255  # Perfect match
    pred[60:70, 60:70] = 255  # Distant small pred

    mean_iou, _ = calculate_mean_iou(pred, gt)
    mean_dice, _ = calculate_mean_dice(pred, gt)
    assert mean_iou == 1.0
    assert mean_dice == 1.0


@pytest.mark.unit
def test_case_08_both_empty():
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)
    res = evaluate_page(pred, gt)
    assert res.mean_iou == 0.0
    assert res.mean_dice == 0.0
    assert res.panel_accuracy == 0.0
    assert res.page_correct is False


@pytest.mark.unit
def test_case_09_gt_nonempty_pred_empty():
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:20, 10:20] = 255
    pred = np.zeros((50, 50), dtype=np.uint8)
    res = evaluate_page(pred, gt)
    assert res.mean_iou == 0.0
    assert res.panel_accuracy == 0.0
    assert res.page_correct is False


@pytest.mark.unit
def test_case_10_gt_empty_pred_nonempty():
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)
    pred[10:20, 10:20] = 255
    res = evaluate_page(pred, gt)
    assert res.panel_accuracy == 0.0
    assert res.precision == 0.0
    assert res.page_correct is False


@pytest.mark.unit
def test_case_11_diagonal_touch_connectivity():
    """Verify skimage 8-way connectivity semantics."""
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 255
    mask[2, 2] = 255  # Diagonally adjacent

    labeled, labels = extract_components(mask)
    # Under skimage label default (connectivity=None or connectivity=2 for 2D, which is 8-way):
    # Diagonally touching pixels merge into 1 connected component
    assert len(labels) == 1


@pytest.mark.unit
def test_case_12_dtype_representations():
    """Verify both {0, 1} and {0, 255} binary encodings."""
    gt1 = np.zeros((20, 20), dtype=np.uint8)
    gt1[5:15, 5:15] = 1

    gt255 = np.zeros((20, 20), dtype=np.uint8)
    gt255[5:15, 5:15] = 255

    res = evaluate_page(gt1, gt255)
    assert res.mean_iou == 1.0
    assert res.mean_dice == 1.0


@pytest.mark.unit
def test_case_13_dice_threshold_exact_boundary():
    """
    Construct exact Dice = 0.9000:
    Let GT have 100 pixels, Pred have 100 pixels, overlap have 90 pixels.
    Dice = 2 * 90 / (100 + 100) = 180 / 200 = 0.9000.
    Since Dice >= 0.9, panel_accuracy must be 1.0.
    """
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True  # 100 pixels (indices 0..9)

    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:9] = True   # 90 pixels overlapping GT
    pred[10, 0:10] = True    # 10 pixels outside GT -> total 100 pixels

    dice = calculate_dice(pred, gt)
    assert abs(dice - 0.9) < 1e-9

    acc = calculate_panel_accuracy(pred, gt, threshold=0.9)
    assert acc == 1.0


@pytest.mark.unit
def test_case_14_dice_threshold_just_below():
    """
    Let GT have 100 pixels, Pred have 100 pixels, overlap have 89 pixels.
    Dice = 2 * 89 / 200 = 0.8900 < 0.9.
    Panel accuracy must be 0.0.
    """
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True  # 100 pixels

    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:8] = True   # 80 pixels
    pred[0:9, 8] = True      # +9 = 89 pixels overlapping
    pred[10, 0:11] = True    # 11 pixels outside -> total 100 pixels

    dice = calculate_dice(pred, gt)
    assert abs(dice - 0.89) < 1e-9

    acc = calculate_panel_accuracy(pred, gt, threshold=0.9)
    assert acc == 0.0


@pytest.mark.unit
def test_case_15_dice_threshold_just_above():
    """
    Let GT have 100 pixels, Pred have 100 pixels, overlap have 91 pixels.
    Dice = 2 * 91 / 200 = 0.9100 > 0.9.
    Panel accuracy must be 1.0.
    """
    gt = np.zeros((20, 20), dtype=bool)
    gt[0:10, 0:10] = True  # 100 pixels

    pred = np.zeros((20, 20), dtype=bool)
    pred[0:10, 0:9] = True   # 90 pixels
    pred[0, 9] = True        # +1 = 91 pixels overlapping
    pred[10, 0:9] = True     # 9 pixels outside -> total 100 pixels

    dice = calculate_dice(pred, gt)
    assert abs(dice - 0.91) < 1e-9

    acc = calculate_panel_accuracy(pred, gt, threshold=0.9)
    assert acc == 1.0
