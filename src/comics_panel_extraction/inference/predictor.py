"""
Prediction runner for checkpoint-native inference.

Exposes primary API:
  predict_probability_mask(image, model=...)
"""

from typing import Optional, Union
import keras
import numpy as np

from comics_panel_extraction.config.models import AppConfig, InferenceConfig, get_legacy_2024_config
from comics_panel_extraction.inference.model import load_inference_model


def validate_inference_input(image: np.ndarray, expected_shape=(384, 384, 3)) -> np.ndarray:
    """
    Validates input image array under the checkpoint-native contract.

    Rules:
      - Must be 3D array of shape (H, W, 3).
      - Strict native spatial dimensions required by default.
      - Must be in intensity domain (0-255).
      - Rejects already-normalized [0, 1] floating inputs to prevent double-normalization.
      - Rejects NaN / Inf.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be a numpy ndarray, got {type(image)}")

    if image.ndim != 3:
        raise ValueError(f"Input image must be a 3D array (H, W, C), got shape {image.shape}")

    if image.shape[2] != 3:
        raise ValueError(f"Input image must have exactly 3 channels, got shape {image.shape}")

    h, w, _ = expected_shape
    if (image.shape[0] != h) or (image.shape[1] != w):
        raise ValueError(
            f"Input spatial shape {image.shape[:2]} does not match model-native requirement {expected_shape[:2]}. "
            f"Arbitrary image resizing is not part of the checkpoint-native contract."
        )

    if not np.isfinite(image).all():
        raise ValueError("Input image contains NaN or Inf values.")

    # Guard against accidental double normalization:
    # If image is floating point and maximum value is <= 1.0 (with nonzero variance), reject
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.max(image))
        min_val = float(np.min(image))
        if max_val <= 1.0 and (max_val > min_val):
            raise ValueError(
                f"Input array has floating range [{min_val:.4f}, {max_val:.4f}] <= 1.0. "
                "The checkpoint-native contract expects 0-255 intensity-domain inputs. "
                "Pre-normalizing to [0, 1] causes fatal double-normalization via the embedded /255 layer."
            )

    # Cast cleanly to float32 for model feeding without scaling
    return image.astype(np.float32)


def predict_probability_mask(
    image: np.ndarray,
    model: Optional[keras.Model] = None,
    config: Optional[InferenceConfig] = None,
) -> np.ndarray:
    """
    Executes model inference on a single 384x384x3 intensity-domain image.

    Input contract:
      - image: np.ndarray (384, 384, 3), values in intensity domain [0, 255].
      - No external /255 division is performed.

    Output contract:
      - Returns raw sigmoid probability mask of shape (384, 384), dtype float32, range [0.0, 1.0].
      - No thresholding, post-processing, or file serialization is performed.
    """
    if model is None:
        model = load_inference_model()

    if config is None:
        model_shape = getattr(model, "input_shape", None)
        if model_shape and len(model_shape) == 4 and model_shape[1] is not None:
            expected_shape = (model_shape[1], model_shape[2], model_shape[3])
        else:
            expected_shape = (448, 448, 3)
    else:
        expected_shape = (config.input_height, config.input_width, config.input_channels)

    prepared_input = validate_inference_input(image, expected_shape=expected_shape)

    # Expand batch dimension: (1, 384, 384, 3)
    batch_input = np.expand_dims(prepared_input, axis=0)

    # Execute in inference mode (training=False ensures Dropout is inactive)
    # Using model(..., training=False) or model.predict(..., verbose=0)
    raw_output = model(batch_input, training=False)

    if hasattr(raw_output, "numpy"):
        raw_output = raw_output.numpy()

    # Raw model output shape is (1, 384, 384, 1), dtype float32
    # Squeeze batch and channel dimension to return 2D probability map (384, 384)
    prob_mask = raw_output[0, :, :, 0].astype(np.float32)
    return prob_mask


def binarize_probability_mask(prob_mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Utility function to binarize raw probability mask into uint8 {0, 255}.
    NOTE: Not called automatically by predict_probability_mask.
    """
    if prob_mask.ndim != 2:
        raise ValueError(f"Probability mask must be 2D, got shape {prob_mask.shape}")
    return (prob_mask > threshold).astype(np.uint8) * 255
