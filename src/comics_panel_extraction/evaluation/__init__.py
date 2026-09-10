"""
Evaluation Subsystem.

Provides standalone evaluation of comic panel segmentation masks preserving
exact legacy matching semantics from the U-Net++ research project.
"""

from comics_panel_extraction.evaluation.components import extract_components
from comics_panel_extraction.evaluation.dataset import (
    evaluate_dataset,
    pair_evaluation_directories,
)
from comics_panel_extraction.evaluation.metrics import (
    calculate_mean_dice,
    calculate_mean_iou,
    calculate_panel_accuracy,
    calculate_precision_recall_f1,
)
from comics_panel_extraction.evaluation.overlap import calculate_dice, calculate_iou
from comics_panel_extraction.evaluation.page import evaluate_page
from comics_panel_extraction.evaluation.types import (
    DatasetEvaluationResult,
    PageEvaluationResult,
)

__all__ = [
    "extract_components",
    "calculate_iou",
    "calculate_dice",
    "calculate_mean_iou",
    "calculate_mean_dice",
    "calculate_panel_accuracy",
    "calculate_precision_recall_f1",
    "evaluate_page",
    "evaluate_dataset",
    "pair_evaluation_directories",
    "PageEvaluationResult",
    "DatasetEvaluationResult",
]
