"""
Image, tensor, and mask I/O operations with strict contract enforcement and destination path safety.
"""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def prepare_output_path(
    output_path: Union[str, Path],
    expected_suffix: str,
    overwrite: bool = False
) -> Path:
    """
    Normalizes the output path suffix FIRST, then validates existence and overwrite safety.
    Creates parent directories if permitted.
    """
    p = Path(output_path)
    if not expected_suffix.startswith("."):
        expected_suffix = "." + expected_suffix

    # 1. Normalize suffix first
    if p.suffix.lower() != expected_suffix.lower():
        p = p.with_suffix(expected_suffix)

    # 2. Check existence after normalization
    if p.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {p}. Use --overwrite to replace.")

    # 3. Create parent directories
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_model_ready_tensor(
    path: Union[str, Path],
    expected_shape: tuple = (384, 384, 3)
) -> np.ndarray:
    """
    Loads a model-ready image tensor from either .npy array or standard image file (PNG/JPEG).
    Strictly requires shape (384, 384, 3) without performing automatic resizing.
    
    Operational Image Policy:
      - .npy: Loaded as-is and verified to be (384, 384, 3) uint8 in [0, 255] intensity domain.
      - image file: Decoded as RGB using cv2.cvtColor(BGR->RGB), preserving [0, 255] intensity domain.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
        if arr.dtype != np.uint8:
            raise TypeError(
                f"Model-ready .npy tensor must have canonical dtype np.uint8, got {arr.dtype}. "
                f"Float or non-uint8 arrays are not permitted as canonical model-ready inputs."
            )
    else:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode image: {p}")
        arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if arr.ndim != 3 or arr.shape != expected_shape:
        raise ValueError(
            f"Input tensor must have model-ready spatial shape {expected_shape}, got {arr.shape}. "
            f"Automatic resizing of arbitrary image dimensions is not supported."
        )

    return arr


def load_binary_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Loads an already-binarized mask from disk (PNG/npy).
    Strictly requires 2D array uint8 with values in {0, 255}.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mask file not found: {p}")

    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
    else:
        arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Failed to decode mask: {p}")

    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D (H, W), got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"Mask must have dtype uint8, got {arr.dtype}")
    if np.any((arr != 0) & (arr != 255)):
        raise ValueError(f"Mask must contain values strictly in {{0, 255}}. Found: {np.unique(arr).tolist()}")

    return arr


def load_probability_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Loads a continuous probability mask from a .npy file.
    Strictly requires 2D float32 array in [0.0, 1.0]. Rejects NaN and Inf.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Probability mask file not found: {p}")
    if p.suffix.lower() != ".npy":
        raise ValueError(f"Canonical probability mask must be stored as .npy array, got {p.name}")

    arr = np.load(str(p))
    if arr.ndim != 2:
        raise ValueError(f"Probability mask must be 2D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"Probability mask must be floating point, got {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Probability mask contains NaN or Inf values.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"Probability mask values must be in [0.0, 1.0], range is [{np.min(arr)}, {np.max(arr)}]")

    return arr.astype(np.float32)


def save_probability_mask(
    mask: np.ndarray,
    output_path: Union[str, Path],
    overwrite: bool = False
) -> Path:
    """
    Saves raw continuous probability mask to .npy file after validating finite range [0.0, 1.0].
    Normalizes suffix before checking existence.
    """
    if mask.ndim != 2:
        raise ValueError(f"Probability mask must be 2D, got shape {mask.shape}")
    if not np.issubdtype(mask.dtype, np.floating):
        raise TypeError(f"Probability mask must have floating-point dtype. Found: {mask.dtype}")

    if not np.all(np.isfinite(mask)):
        raise ValueError("Cannot save probability mask containing NaN or Inf.")
    if np.any((mask < 0.0) | (mask > 1.0)):
        raise ValueError(f"Probability mask values must be in [0.0, 1.0], range is [{np.min(mask)}, {np.max(mask)}]")

    p = prepare_output_path(output_path, expected_suffix=".npy", overwrite=overwrite)
    np.save(str(p), mask.astype(np.float32))
    return p


def save_binary_mask(
    mask: np.ndarray,
    output_path: Union[str, Path],
    overwrite: bool = False
) -> Path:
    """
    Saves binary mask losslessly as PNG image after validating 2D uint8 {0, 255}.
    Normalizes suffix before checking existence.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    if mask.dtype != np.uint8:
        raise TypeError(f"Mask must have dtype uint8, got {mask.dtype}")
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError(f"Mask must contain values strictly in {{0, 255}}. Found: {np.unique(mask).tolist()}")

    p = prepare_output_path(output_path, expected_suffix=".png", overwrite=overwrite)
    success = cv2.imwrite(str(p), mask)
    if not success:
        raise IOError(f"Failed to write mask to {p}")
    return p


def read_mask_binary(path: Union[str, Path]) -> np.ndarray:
    """Backward-compatible alias for load_binary_mask."""
    return load_binary_mask(path)


def write_mask_binary(mask: np.ndarray, output_path: Union[str, Path], overwrite: bool = True) -> Path:
    """Backward-compatible alias for save_binary_mask."""
    return save_binary_mask(mask, output_path, overwrite=overwrite)
