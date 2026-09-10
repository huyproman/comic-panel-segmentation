"""
Model loading logic for serialized Keras checkpoints.
"""

from pathlib import Path
from typing import Optional, Union
import hashlib
import os
import keras

M1_DEFAULT_CHECKPOINT_SHA256 = "b6e4e63bc98408a482b9b00d11c8a21b42e47f850ceae4a7154c2e0145dba57b"
M1_DEFAULT_FILENAME = "best_m1_unetpp_val_loss.keras"


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_sha256(
    checkpoint_path: Union[str, Path],
    expected_sha256: str = M1_DEFAULT_CHECKPOINT_SHA256,
) -> bool:
    """
    Verifies that checkpoint exists and matches the expected SHA-256 hash.
    Raises ValueError if hash mismatch occurs.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint artifact missing: {path}")

    actual_sha = compute_file_sha256(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {path}. Expected: {expected_sha256}, Got: {actual_sha}."
        )
    return True


def resolve_default_checkpoint_path() -> Path:
    """
    Resolves default checkpoint path checking environment or repository root.
    """
    env_path = os.environ.get("COMICS_CHECKPOINT_PATH")
    if env_path:
        return Path(env_path)

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    # Standard location in evidence/artifacts
    m1_path = (
        repo_root.parent
        / "markdowns_dir"
        / "manuscript_first_reconstruction"
        / "round_03_manuscript_faithful_reconstruction"
        / "phase_m1_unetpp_from_scratch"
        / "checkpoints"
        / M1_DEFAULT_FILENAME
    )
    if m1_path.exists():
        return m1_path

    # Fallback to local artifacts
    local_p = repo_root / ".local_artifacts" / M1_DEFAULT_FILENAME
    return local_p


def load_inference_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    expected_sha256: Optional[str] = M1_DEFAULT_CHECKPOINT_SHA256,
    compile: bool = False,
) -> keras.Model:
    """
    Loads serialized Keras model checkpoint after verifying cryptographic SHA-256.
    """
    if checkpoint_path is None:
        resolved_path = resolve_default_checkpoint_path()
    else:
        resolved_path = Path(checkpoint_path)

    if expected_sha256 is not None:
        verify_checkpoint_sha256(resolved_path, expected_sha256=expected_sha256)

    model = keras.models.load_model(str(resolved_path), compile=compile, safe_mode=False)
    return model
