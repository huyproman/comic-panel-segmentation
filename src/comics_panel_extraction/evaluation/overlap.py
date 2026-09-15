import numpy as np
def calculate_iou(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    intersection = int(np.logical_and(pred_binary, gt_binary).sum())
    union = int(np.logical_or(pred_binary, gt_binary).sum())
    return float(intersection / union) if union > 0 else 0.0
def calculate_dice(pred_binary: np.ndarray, gt_binary: np.ndarray) -> float:
    intersection = int(np.logical_and(pred_binary, gt_binary).sum())
    sum_cardinality = int(pred_binary.sum() + gt_binary.sum())
    return float(2 * intersection / sum_cardinality) if sum_cardinality > 0 else 0.0
