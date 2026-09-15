from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label

from comics_panel_extraction.evaluation.overlap import calculate_dice, calculate_iou
from comics_panel_extraction.evaluation.types import PageEvaluationResult

def extract_instances_ccl(mask: np.ndarray, connectivity: int = 2) -> Tuple[np.ndarray, List[int]]:
    bin_mask = (mask > 0).astype(np.uint8)
    labeled = label(bin_mask, connectivity=connectivity, background=0)
    unique_ids = [int(i) for i in np.unique(labeled) if i != 0]
    return labeled, unique_ids

def compute_instance_matching_matrix(
    pred_labeled: np.ndarray,
    pred_ids: List[int],
    gt_labeled: np.ndarray,
    gt_ids: List[int]
) -> np.ndarray:
    n_pred = len(pred_ids)
    n_gt = len(gt_ids)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p_id in enumerate(pred_ids):
        p_mask = (pred_labeled == p_id)
        for j, g_id in enumerate(gt_ids):
            g_mask = (gt_labeled == g_id)
            iou_matrix[i, j] = calculate_iou(p_mask, g_mask)
    return iou_matrix

def evaluate_page_instances(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dice_correctness_threshold: float = 0.90,
    ccl_connectivity: int = 2
) -> Dict[str, Any]:
    gt_bin = (gt_mask > 0.5)
    pred_bin = (pred_mask > 0.5)
    inter_px = float(np.logical_and(pred_bin, gt_bin).sum())
    union_px = float(np.logical_or(pred_bin, gt_bin).sum())
    fg_pixel_iou = inter_px / union_px if union_px > 0 else 1.0
    fg_pixel_dice = (2.0 * inter_px) / (float(pred_bin.sum()) + float(gt_bin.sum())) if (pred_bin.sum() + gt_bin.sum()) > 0 else 1.0

    pred_labeled, pred_ids = extract_instances_ccl(pred_bin, connectivity=ccl_connectivity)
    gt_labeled, gt_ids = extract_instances_ccl(gt_bin, connectivity=ccl_connectivity)
    n_pred = len(pred_ids)
    n_gt = len(gt_ids)

    if n_gt == 0 and n_pred == 0:
        return {
            "canonical_instance_miou": 0.0,
            "mean_iou": 0.0,
            "page_instance_miou": 0.0,
            "mean_dice": 0.0,
            "page_instance_mdice": 0.0,
            "panel_accuracy": 0.0,
            "page_accuracy": 0.0,
            "strict_page_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "panel_tp": 0,
            "panel_fp": 0,
            "panel_fn": 0,
            "gt_count": 0,
            "pred_count": 0,
            "page_correct": False,
            "gt_panel_count": 0,
            "pred_panel_count": 0,
            "iou_per_gt": [],
            "dice_per_gt": [],
            "foreground_pixel_iou": fg_pixel_iou,
            "foreground_pixel_dice": fg_pixel_dice,
        }

    if n_gt == 0 and n_pred > 0:
        return {
            "canonical_instance_miou": 0.0,
            "mean_iou": 0.0,
            "page_instance_miou": 0.0,
            "mean_dice": 0.0,
            "page_instance_mdice": 0.0,
            "panel_accuracy": 0.0,
            "page_accuracy": 0.0,
            "strict_page_accuracy": 0.0,
            "precision": 0.0,
            "recall": 1.0,
            "f1": 0.0,
            "panel_tp": 0,
            "panel_fp": n_pred,
            "panel_fn": 0,
            "gt_count": 0,
            "pred_count": n_pred,
            "page_correct": False,
            "gt_panel_count": 0,
            "pred_panel_count": n_pred,
            "iou_per_gt": [],
            "dice_per_gt": [],
            "foreground_pixel_iou": fg_pixel_iou,
            "foreground_pixel_dice": fg_pixel_dice,
        }

    if n_gt > 0 and n_pred == 0:
        return {
            "canonical_instance_miou": 0.0,
            "mean_iou": 0.0,
            "page_instance_miou": 0.0,
            "mean_dice": 0.0,
            "page_instance_mdice": 0.0,
            "panel_accuracy": 0.0,
            "page_accuracy": 0.0,
            "strict_page_accuracy": 0.0,
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "panel_tp": 0,
            "panel_fp": 0,
            "panel_fn": n_gt,
            "gt_count": n_gt,
            "pred_count": 0,
            "page_correct": False,
            "gt_panel_count": n_gt,
            "pred_panel_count": 0,
            "iou_per_gt": [0.0] * n_gt,
            "dice_per_gt": [0.0] * n_gt,
            "foreground_pixel_iou": fg_pixel_iou,
            "foreground_pixel_dice": fg_pixel_dice,
        }

    iou_matrix = compute_instance_matching_matrix(pred_labeled, pred_ids, gt_labeled, gt_ids)
    row_ind, col_ind = linear_sum_assignment(1.0 - iou_matrix)

    matched_gt = set()
    matched_pred = set()
    per_gt_ious = {g_id: 0.0 for g_id in gt_ids}
    per_gt_dices = {g_id: 0.0 for g_id in gt_ids}
    panel_tp = 0

    for r, c in zip(row_ind, col_ind):
        p_id = pred_ids[r]
        g_id = gt_ids[c]
        p_mask = (pred_labeled == p_id)
        g_mask = (gt_labeled == g_id)
        pair_iou = float(iou_matrix[r, c])
        pair_dice = float(calculate_dice(p_mask, g_mask))
        matched_gt.add(g_id)
        matched_pred.add(p_id)
        per_gt_ious[g_id] = pair_iou
        per_gt_dices[g_id] = pair_dice
        if pair_dice >= dice_correctness_threshold:
            panel_tp += 1

    panel_fp = n_pred - panel_tp
    panel_fn = n_gt - panel_tp
    page_miou = float(np.mean(list(per_gt_ious.values())))
    page_mdice = float(np.mean(list(per_gt_dices.values())))
    panel_accuracy = panel_tp / float(n_gt) if n_gt > 0 else 1.0
    page_accuracy = 1.0 if panel_tp == n_gt else 0.0
    strict_page_accuracy = 1.0 if (panel_tp == n_gt and panel_fp == 0) else 0.0
    prec = panel_tp / float(n_pred) if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    rec = panel_tp / float(n_gt) if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0

    return {
        "canonical_instance_miou": page_miou,
        "mean_iou": page_miou,
        "page_instance_miou": page_miou,
        "mean_dice": page_mdice,
        "page_instance_mdice": page_mdice,
        "panel_accuracy": panel_accuracy,
        "page_accuracy": page_accuracy,
        "strict_page_accuracy": strict_page_accuracy,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "panel_tp": panel_tp,
        "panel_fp": panel_fp,
        "panel_fn": panel_fn,
        "gt_count": n_gt,
        "pred_count": n_pred,
        "page_correct": (panel_accuracy == 1.0 and n_gt > 0),
        "gt_panel_count": n_gt,
        "pred_panel_count": n_pred,
        "iou_per_gt": [per_gt_ious[g] for g in gt_ids],
        "dice_per_gt": [per_gt_dices[g] for g in gt_ids],
        "foreground_pixel_iou": fg_pixel_iou,
        "foreground_pixel_dice": fg_pixel_dice,
    }

def aggregate_dataset_instances(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not page_results:
        return {
            "canonical_instance_miou": 0.0,
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "panel_accuracy": 0.0,
            "page_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
        }

    dataset_miou = float(np.mean([p["canonical_instance_miou"] for p in page_results]))
    dataset_mdice = float(np.mean([p["page_instance_mdice"] for p in page_results]))
    dataset_page_acc = float(np.mean([p["page_accuracy"] for p in page_results]))
    total_gt = sum(p["gt_count"] for p in page_results)
    total_pred = sum(p["pred_count"] for p in page_results)
    total_tp = sum(p["panel_tp"] for p in page_results)
    total_fp = sum(p["panel_fp"] for p in page_results)
    total_fn = sum(p["panel_fn"] for p in page_results)

    dataset_panel_acc = total_tp / float(total_gt) if total_gt > 0 else 1.0
    pooled_prec = total_tp / float(total_pred) if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
    pooled_rec = total_tp / float(total_gt) if total_gt > 0 else (1.0 if total_pred == 0 else 0.0)
    pooled_f1 = (2.0 * pooled_prec * pooled_rec) / (pooled_prec + pooled_rec) if (pooled_prec + pooled_rec) > 0.0 else 0.0

    return {
        "canonical_instance_miou": dataset_miou,
        "mean_iou": dataset_miou,
        "mean_dice": dataset_mdice,
        "panel_accuracy": dataset_panel_acc,
        "page_accuracy": dataset_page_acc,
        "precision": pooled_prec,
        "recall": pooled_rec,
        "f1": pooled_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "total_pred": total_pred,
    }

class PanelInstanceEvaluator:
    def __init__(self, dice_threshold: float = 0.90, connectivity: int = 2):
        self.dice_threshold = dice_threshold
        self.connectivity = connectivity

    def evaluate_page(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Any]:
        return evaluate_page_instances(
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            dice_correctness_threshold=self.dice_threshold,
            ccl_connectivity=self.connectivity,
        )

    def evaluate_dataset(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_dataset_instances(page_results)

def evaluate_page(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray, config: Optional[Any] = None) -> PageEvaluationResult:
    thresh = 0.90
    d = evaluate_page_instances(predicted_mask, ground_truth_mask, dice_correctness_threshold=thresh, ccl_connectivity=2)
    return PageEvaluationResult(
        mean_iou=d["mean_iou"],
        mean_dice=d["mean_dice"],
        panel_accuracy=d["panel_accuracy"],
        precision=d["precision"],
        recall=d["recall"],
        f1=d["f1"],
        page_correct=d["page_correct"],
        gt_panel_count=d["gt_panel_count"],
        pred_panel_count=d["pred_panel_count"],
        iou_per_gt=d["iou_per_gt"],
        dice_per_gt=d["dice_per_gt"],
    )

evaluate_instance_page = evaluate_page_instances
evaluate_instance_dataset = aggregate_dataset_instances
