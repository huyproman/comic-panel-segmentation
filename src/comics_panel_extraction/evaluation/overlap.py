"""
Exact IoU and Dice overlap primitives.

Migrated directly from comics_analysis/post_process/metric.ipynb Cell 0.
"""

import numpy as np


def calculate_iou(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two binary boolean masks.

    Legacy formula:
      intersection = np.logical_and(pred_binary, gt_binary).sum()
      union = np.logical_or(pred_binary, gt_binary).sum()
      return intersection / union if union > 0 else 0
    """
    intersection = int(np.logical_and(pred_binary, gt_binary).sum())
    union = int(np.logical_or(pred_binary, gt_binary).sum())
    return float(intersection / union) if union > 0 else 0.0


def calculate_dice(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    """
    Computes Dice coefficient between two binary boolean masks.

    Legacy formula:
      intersection = np.logical_and(pred_binary, gt_binary).sum()
      sum_cardinality = pred_binary.sum() + gt_binary.sum()
      return 2 * intersection / sum_cardinality if sum_cardinality > 0 else 0
    """
    intersection = int(np.logical_and(pred_binary, gt_binary).sum())
    sum_cardinality = int(pred_binary.sum() + gt_binary.sum())
    return float(2 * intersection / sum_cardinality) if sum_cardinality > 0 else 0.0
