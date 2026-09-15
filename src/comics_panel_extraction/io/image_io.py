from pathlib import Path
from typing import Union
import cv2
import numpy as np

def prepare_output_path(
    output_path: Union[str, Path],
    expected_suffix: str,
    overwrite: bool = False
) -> Path:
    p = Path(output_path)
    if not expected_suffix.startswith("."):
        expected_suffix = "." + expected_suffix

    if p.suffix.lower() != expected_suffix.lower():
        p = p.with_suffix(expected_suffix)

    if p.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {p}. Use --overwrite to replace.")

    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def load_model_ready_tensor(
    path: Union[str, Path],
    expected_shape: tuple = (448, 448, 3)
) -> np.ndarray:
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
    return load_binary_mask(path)

def write_mask_binary(mask: np.ndarray, output_path: Union[str, Path], overwrite: bool = True) -> Path:
    return save_binary_mask(mask, output_path, overwrite=overwrite)
