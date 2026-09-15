import hashlib
import os
from pathlib import Path
from typing import Optional, Union
import keras

def compute_file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_checkpoint(path: Union[str, Path], expected_sha256: Optional[str] = None) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    if expected_sha256 is not None:
        file_sha = compute_file_sha256(p)
        return file_sha.lower() == expected_sha256.lower()
    return True

def locate_checkpoint(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists() and p.is_file():
            return p
        return None

    env_p = os.environ.get("COMICS_CHECKPOINT_PATH")
    if env_p:
        p = Path(env_p)
        if p.exists() and p.is_file():
            return p

    return None

def load_inference_model(
    checkpoint_path: Union[str, Path],
    expected_sha256: Optional[str] = None,
) -> keras.Model:
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for inference. No default checkpoint is bundled.")

    target_p = locate_checkpoint(custom_path=checkpoint_path)
    if target_p is None:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    if expected_sha256 is not None and not validate_checkpoint(target_p, expected_sha256=expected_sha256):
        raise ValueError(f"Checkpoint integrity check failed for {target_p}")

    try:
        model = keras.models.load_model(str(target_p), compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {target_p}: {e}") from e

    if model.input_shape != (None, 448, 448, 3) or model.output_shape != (None, 448, 448, 1):
        raise ValueError(f"Incompatible model input/output shape: {model.input_shape} -> {model.output_shape}. Expected (None, 448, 448, 3) -> (None, 448, 448, 1).")

    return model
