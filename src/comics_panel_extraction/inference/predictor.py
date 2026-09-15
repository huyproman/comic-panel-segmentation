from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from comics_panel_extraction.postprocessing.line_analysis import run_line_segment_postprocessing

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (448, 448)) -> Tuple[np.ndarray, Tuple[int, int]]:
    orig_h, orig_w = image.shape[:2]
    if (orig_w, orig_h) != target_size:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        resized = image.copy()
    if resized.dtype == np.uint8:
        norm = resized.astype(np.float32) / 255.0
    else:
        norm = resized.astype(np.float32)
        if norm.max() > 1.0:
            norm = norm / 255.0
    return norm, (orig_h, orig_w)

def predict_probability_mask(image: np.ndarray, model: Any) -> np.ndarray:
    prep_img, (orig_h, orig_w) = preprocess_image(image, target_size=(448, 448))
    batch_input = np.expand_dims(prep_img, axis=0)
    pred = model.predict(batch_input, verbose=0)
    if isinstance(pred, list):
        pred = pred[0]
    prob_map = pred[0, ..., 0]
    return prob_map.astype(np.float32)

def binarize_probability_mask(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob_map >= threshold).astype(np.uint8) * 255

def predict_mask(
    image: np.ndarray,
    model: Any,
    threshold: float = 0.5,
    apply_postprocessing: bool = True,
    target_size: Tuple[int, int] = (448, 448),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    orig_h, orig_w = image.shape[:2]
    prob = predict_probability_mask(image, model)
    raw_binary_448 = binarize_probability_mask(prob, threshold=threshold)
    raw_mask = cv2.resize(raw_binary_448, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if apply_postprocessing:
        final_pp, diagnostics = run_line_segment_postprocessing(raw_binary_448)
        final_mask = cv2.resize(final_pp, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        final_mask = raw_mask.copy()
        diagnostics = {}

    return final_mask, raw_mask, diagnostics

def validate_inference_input(tensor: np.ndarray, expected_shape=(448, 448, 3)):
    if not isinstance(tensor, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")
    if tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")
    if np.isnan(tensor).any() or np.isinf(tensor).any():
        raise ValueError("Input tensor contains NaN or Inf values")
    if tensor.dtype in [np.float32, np.float64]:
        if tensor.min() < 0.0 or tensor.max() > 1.0:
            raise ValueError("Float image must be in range [0.0, 1.0]")
