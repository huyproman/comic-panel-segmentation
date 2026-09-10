"""
Core matching and metric algorithms preserving exact legacy semantics.

Migrated from comics_analysis/post_process/metric.ipynb:
- calculate_mean_iou (Cell 0): GT -> best Pred IoU matching
- calculate_mean_dice (Cell 0): GT -> best Pred Dice matching
- calculate_accuracy / Panel Accuracy (Cell 4, 7): GT -> best Pred matching at Dice >= threshold
- calculate_metrics_with_threshold (Cell 10): Pred -> best GT matching for Precision / Recall / F1
"""

from typing import List, Set, Tuple
import numpy as np

from comics_panel_extraction.evaluation.components import extract_components
from comics_panel_extraction.evaluation.overlap import calculate_dice, calculate_iou


def calculate_mean_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Tuple[float, List[float]]:
    """
    Computes mean IoU over all ground-truth panels.

    Direction: GT panel -> best matching predicted panel.
    Returns: (mean_iou, list_of_best_ious_per_gt)
    """
    pred_labels, pred_objects = extract_components(predicted_mask)
    gt_labels, gt_objects = extract_components(ground_truth_mask)

    iou_values: List[float] = []

    for gt_label in gt_objects:
        gt_binary = (gt_labels == gt_label)
        best_iou = 0.0

        for pred_label in pred_objects:
            pred_binary = (pred_labels == pred_label)
            iou = calculate_iou(pred_binary, gt_binary)
            if iou > best_iou:
                best_iou = iou

        iou_values.append(best_iou)

    mean_iou = float(np.mean(iou_values)) if len(iou_values) > 0 else 0.0
    return mean_iou, iou_values


def calculate_mean_dice(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Tuple[float, List[float]]:
    """
    Computes mean Dice over all ground-truth panels.

    Direction: GT panel -> best matching predicted panel.
    Returns: (mean_dice, list_of_best_dices_per_gt)
    """
    pred_labels, pred_objects = extract_components(predicted_mask)
    gt_labels, gt_objects = extract_components(ground_truth_mask)

    dice_values: List[float] = []

    for gt_label in gt_objects:
        gt_binary = (gt_labels == gt_label)
        best_dice = 0.0

        for pred_label in pred_objects:
            pred_binary = (pred_labels == pred_label)
            dice = calculate_dice(pred_binary, gt_binary)
            if dice > best_dice:
                best_dice = dice

        dice_values.append(best_dice)

    mean_dice = float(np.mean(dice_values)) if len(dice_values) > 0 else 0.0
    return mean_dice, dice_values


def calculate_panel_accuracy(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.9,
) -> float:
    """
    Computes Panel Accuracy (proportion of GT panels with best Dice >= threshold).

    Direction: GT panel -> best matching predicted panel.
    Legacy implementation: calculate_accuracy (Cell 4 / Cell 7).
    """
    pred_labels, pred_objects = extract_components(predicted_mask)
    gt_labels, gt_objects = extract_components(ground_truth_mask)

    correct_predictions = 0

    for gt_label in gt_objects:
        gt_binary = (gt_labels == gt_label)
        best_dice = 0.0

        for pred_label in pred_objects:
            pred_binary = (pred_labels == pred_label)
            dice = calculate_dice(pred_binary, gt_binary)
            if dice > best_dice:
                best_dice = dice

        if best_dice >= threshold:
            correct_predictions += 1

    accuracy = float(correct_predictions / len(gt_objects)) if len(gt_objects) > 0 else 0.0
    return accuracy


def calculate_precision_recall_f1(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.9,
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F1-score preserving legacy matching semantics.

    Direction: Predicted panel -> best matching GT panel.
    Legacy implementation: calculate_metrics_with_threshold (Cell 10).
    Maintains matched_gt set to track unique ground-truth panels covered.
    """
    pred_labels, pred_objects = extract_components(predicted_mask)
    gt_labels, gt_objects = extract_components(ground_truth_mask)

    matched_gt: Set[int] = set()
    tp = 0
    fp = 0

    for pred_label in pred_objects:
        pred_binary = (pred_labels == pred_label)
        best_dice = 0.0
        best_match = None

        for gt_label in gt_objects:
            gt_binary = (gt_labels == gt_label)
            dice = calculate_dice(pred_binary, gt_binary)
            if dice > best_dice:
                best_dice = dice
                best_match = int(gt_label)

        if best_dice >= threshold:
            tp += 1
            if best_match is not None:
                matched_gt.add(best_match)
        else:
            fp += 1

    fn = len(gt_objects) - len(matched_gt)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score
