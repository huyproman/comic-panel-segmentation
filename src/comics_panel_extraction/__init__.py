"""
Comic Panel Extraction Package.

Converged Production Package (v0.2.0):
- M1 Operational Inference (Best Current Empirical Comic Reconstruction)
- Canonical Recovered Historical Post-Processing (Line-Segment Analysis)
- Historical Dataset Evaluator (Table 1 & Table 2 Metric Reproduction)
"""

from comics_panel_extraction.inference import (
    load_inference_model,
    predict_probability_mask,
    binarize_probability_mask,
    resolve_default_checkpoint_path,
    M1_DEFAULT_CHECKPOINT_SHA256,
)
from comics_panel_extraction.historical_legacy import (
    run_canonical_historical_postprocessing,
    HistoricalLegacyConfig,
)
from comics_panel_extraction.evaluation import (
    evaluate_dataset,
    evaluate_page,
    calculate_mean_iou,
    calculate_mean_dice,
    calculate_panel_accuracy,
)

__version__ = "0.2.0"

__all__ = [
    "load_inference_model",
    "predict_probability_mask",
    "binarize_probability_mask",
    "resolve_default_checkpoint_path",
    "M1_DEFAULT_CHECKPOINT_SHA256",
    "run_canonical_historical_postprocessing",
    "HistoricalLegacyConfig",
    "evaluate_dataset",
    "evaluate_page",
    "calculate_mean_iou",
    "calculate_mean_dice",
    "calculate_panel_accuracy",
    "__version__",
]
