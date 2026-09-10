"""
Dataset-level evaluation and directory pairing.

Replicates the exact folder processing and pairing loops from metric.ipynb:
- process_folders (Cell 1): mIoU and mDice
- iterate_masks (Cell 7): Panel Accuracy and Page Accuracy
- process_with_metrics (Cell 10): Precision, Recall, and F1
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from comics_panel_extraction.config.models import EvaluationConfig, get_legacy_2024_config
from comics_panel_extraction.evaluation.page import evaluate_page
from comics_panel_extraction.evaluation.types import DatasetEvaluationResult, PageEvaluationResult
from comics_panel_extraction.io.image_io import read_mask_binary


def pair_evaluation_directories(
    gt_dir: Union[str, Path],
    pred_dir: Union[str, Path],
) -> List[Tuple[str, Path, Path]]:
    """
    Pairs ground truth files with predicted mask files according to legacy conventions.

    Legacy pairing patterns:
      1. true_mask_<ID>.png -> pred_mask_raw_<ID>.png
      2. Direct identical filenames: <name>.png -> <name>.png
    """
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_path}")

    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(".png") or f.endswith(".jpg")])
    pred_files_set = set(os.listdir(pred_path))

    pairs: List[Tuple[str, Path, Path]] = []

    for gt_file in gt_files:
        p_gt = gt_path / gt_file

        # Check legacy prefix pattern: true_mask_<ID>.png -> pred_mask_raw_<ID>.png
        if gt_file.startswith("true_mask_"):
            identifier = gt_file.replace("true_mask_", "")
            candidate_pred = f"pred_mask_raw_{identifier}"
            if candidate_pred in pred_files_set:
                pairs.append((identifier, p_gt, pred_path / candidate_pred))
                continue

        # Check identical filename
        if gt_file in pred_files_set:
            pairs.append((gt_file, p_gt, pred_path / gt_file))
            continue

        # Check pred_mask_raw_<gt_file>
        cand2 = f"pred_mask_raw_{gt_file}"
        if cand2 in pred_files_set:
            pairs.append((gt_file, p_gt, pred_path / cand2))
            continue

    return pairs


def evaluate_dataset(
    gt_dir: Union[str, Path],
    pred_dir: Union[str, Path],
    corpus_id: str = "dataset",
    config: Optional[EvaluationConfig] = None,
) -> DatasetEvaluationResult:
    """
    Evaluates an entire directory of predicted masks against ground truth masks.

    Returns:
      DatasetEvaluationResult with aggregated 7 metrics and per-page records.
    """
    if config is None:
        config = get_legacy_2024_config().evaluation

    pairs = pair_evaluation_directories(gt_dir, pred_dir)
    if not pairs:
        raise ValueError(f"No valid mask pairs found between {gt_dir} and {pred_dir}")

    per_page: Dict[str, PageEvaluationResult] = {}
    ious: List[float] = []
    dices: List[float] = []
    panel_accs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    correct_pages = 0

    for identifier, gt_p, pred_p in pairs:
        gt_mask = read_mask_binary(gt_p)
        pred_mask = read_mask_binary(pred_p)

        page_res = evaluate_page(predicted_mask=pred_mask, ground_truth_mask=gt_mask, config=config)
        per_page[identifier] = page_res

        ious.append(page_res.mean_iou)
        dices.append(page_res.mean_dice)
        panel_accs.append(page_res.panel_accuracy)
        precisions.append(page_res.precision)
        recalls.append(page_res.recall)
        f1s.append(page_res.f1)
        if page_res.page_correct:
            correct_pages += 1

    total_pages = len(pairs)
    overall_iou = float(np.mean(ious)) if total_pages > 0 else 0.0
    overall_dice = float(np.mean(dices)) if total_pages > 0 else 0.0
    overall_panel_acc = float(np.mean(panel_accs)) if total_pages > 0 else 0.0
    overall_precision = float(np.mean(precisions)) if total_pages > 0 else 0.0
    overall_recall = float(np.mean(recalls)) if total_pages > 0 else 0.0
    overall_f1 = float(np.mean(f1s)) if total_pages > 0 else 0.0
    overall_page_acc = float(correct_pages / total_pages) if total_pages > 0 else 0.0

    return DatasetEvaluationResult(
        corpus_id=corpus_id,
        page_count=total_pages,
        mean_iou=overall_iou,
        mean_dice=overall_dice,
        panel_accuracy=overall_panel_acc,
        precision=overall_precision,
        recall=overall_recall,
        f1=overall_f1,
        page_accuracy=overall_page_acc,
        per_page_results=per_page,
    )
