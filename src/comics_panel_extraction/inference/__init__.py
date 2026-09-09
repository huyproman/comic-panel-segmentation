"""Inference subsystem exports."""

from comics_panel_extraction.inference.model import (
    load_inference_model,
    resolve_default_checkpoint_path,
    verify_checkpoint_sha256,
    compute_file_sha256,
    M1_DEFAULT_CHECKPOINT_SHA256,
)
from comics_panel_extraction.inference.predictor import (
    predict_probability_mask,
    binarize_probability_mask,
    validate_inference_input,
)

EXPECTED_CHECKPOINT_SHA256 = M1_DEFAULT_CHECKPOINT_SHA256
EXPECTED_CHECKPOINT_ID = "M1_COMIC_UNETPP_CHECKPOINT"

__all__ = [
    "load_inference_model",
    "resolve_default_checkpoint_path",
    "verify_checkpoint_sha256",
    "compute_file_sha256",
    "M1_DEFAULT_CHECKPOINT_SHA256",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_CHECKPOINT_ID",
    "predict_probability_mask",
    "binarize_probability_mask",
    "validate_inference_input",
]
