from typing import List, Tuple
import numpy as np
from comics_panel_extraction.evaluation.evaluator import evaluate_page_instances

def calculate_mean_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Tuple[float, List[float]]:
    res = evaluate_page_instances(predicted_mask, ground_truth_mask)
    return res["mean_iou"], res["iou_per_gt"]

def calculate_mean_dice(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Tuple[float, List[float]]:
    res = evaluate_page_instances(predicted_mask, ground_truth_mask)
    return res["mean_dice"], res["dice_per_gt"]

def calculate_panel_accuracy(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.9,
) -> float:
    res = evaluate_page_instances(predicted_mask, ground_truth_mask, dice_correctness_threshold=threshold)
    return res["panel_accuracy"]

def calculate_precision_recall_f1(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.9,
) -> Tuple[float, float, float]:
    res = evaluate_page_instances(predicted_mask, ground_truth_mask, dice_correctness_threshold=threshold)
    return res["precision"], res["recall"], res["f1"]
