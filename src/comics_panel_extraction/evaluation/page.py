"""
Page-level panel extraction evaluation.

Integrates all metrics for a single predicted vs ground-truth mask pair.
Preserves legacy Page Accuracy semantics:
- Page is counted correct (page_correct = True) if and only if panel_accuracy == 1.0.
- Extra false-positive predictions do NOT invalidate the page under legacy_2024.
"""

from typing import Optional
import numpy as np

from comics_panel_extraction.config.models import AppConfig, EvaluationConfig, get_legacy_2024_config
from comics_panel_extraction.evaluation.components import extract_components
from comics_panel_extraction.evaluation.metrics import (
    calculate_mean_dice,
    calculate_mean_iou,
    calculate_panel_accuracy,
    calculate_precision_recall_f1,
)
from comics_panel_extraction.evaluation.types import PageEvaluationResult


def evaluate_page(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    config: Optional[EvaluationConfig] = None,
) -> PageEvaluationResult:
    """
    Evaluates one pair of predicted and ground-truth segmentation masks.

    Parameters:
      predicted_mask: np.ndarray (H, W), single-channel binary mask
      ground_truth_mask: np.ndarray (H, W), single-channel binary mask
      config: Optional[EvaluationConfig], default uses legacy_2024 parameters (Dice threshold = 0.9)

    Returns:
      PageEvaluationResult dataclass containing all 7 legacy metrics.
    """
    if config is None:
        config = get_legacy_2024_config().evaluation

    threshold = config.dice_threshold

    # Component counts
    _, pred_objects = extract_components(predicted_mask)
    _, gt_objects = extract_components(ground_truth_mask)

    # 1. mIoU and best IoUs
    mean_iou, iou_list = calculate_mean_iou(predicted_mask, ground_truth_mask)

    # 2. mDice and best Dices
    mean_dice, dice_list = calculate_mean_dice(predicted_mask, ground_truth_mask)

    # 3. Panel Accuracy (GT -> best Pred matching)
    panel_accuracy = calculate_panel_accuracy(predicted_mask, ground_truth_mask, threshold=threshold)

    # 4. Precision, Recall, F1 (Pred -> best GT matching)
    precision, recall, f1 = calculate_precision_recall_f1(predicted_mask, ground_truth_mask, threshold=threshold)

    # 5. Page Accuracy criterion
    # Under legacy_2024: A page is correct if all GT panels are matched at Dice >= threshold (accuracy == 1)
    # Extra false positives are ignored.
    page_correct = bool(panel_accuracy == 1.0) and (len(gt_objects) > 0)

    return PageEvaluationResult(
        mean_iou=mean_iou,
        mean_dice=mean_dice,
        panel_accuracy=panel_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        page_correct=page_correct,
        gt_panel_count=len(gt_objects),
        pred_panel_count=len(pred_objects),
        iou_per_gt=iou_list,
        dice_per_gt=dice_list,
    )
