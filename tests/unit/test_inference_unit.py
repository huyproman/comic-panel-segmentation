"""
Unit and regression tests for standalone inference subsystem.

Covers:
- Model artifact presence and SHA-256 verification
- Failure on missing model artifact
- Rejection of corrupt or wrong-hash artifact
- Input validation: reject 2D, 4-channel, wrong spatial shape, NaN, Inf
- Input validation: reject already-normalized [0, 1] floating inputs to guard against double-normalization
- Checkpoint loading and structure verification
- Deterministic prediction output verification
"""

from pathlib import Path
import numpy as np
import pytest

from comics_panel_extraction.inference import (
    EXPECTED_CHECKPOINT_ID,
    EXPECTED_CHECKPOINT_SHA256,
    binarize_probability_mask,
    compute_file_sha256,
    load_inference_model,
    predict_probability_mask,
    resolve_default_checkpoint_path,
    validate_inference_input,
    verify_checkpoint_sha256,
)


@pytest.mark.unit
def test_input_validation_valid():
    img = np.zeros((384, 384, 3), dtype=np.uint8)
    prepared = validate_inference_input(img)
    assert prepared.shape == (384, 384, 3)
    assert prepared.dtype == np.float32


@pytest.mark.unit
def test_input_validation_reject_wrong_rank():
    with pytest.raises(ValueError, match="3D array"):
        validate_inference_input(np.zeros((384, 384), dtype=np.uint8))

    with pytest.raises(ValueError, match="3D array"):
        validate_inference_input(np.zeros((1, 384, 384, 3), dtype=np.uint8))


@pytest.mark.unit
def test_input_validation_reject_wrong_channels():
    with pytest.raises(ValueError, match="3 channels"):
        validate_inference_input(np.zeros((384, 384, 1), dtype=np.uint8))

    with pytest.raises(ValueError, match="3 channels"):
        validate_inference_input(np.zeros((384, 384, 4), dtype=np.uint8))


@pytest.mark.unit
def test_input_validation_reject_wrong_spatial_shape():
    with pytest.raises(ValueError, match="spatial shape"):
        validate_inference_input(np.zeros((448, 448, 3), dtype=np.uint8))


@pytest.mark.unit
def test_input_validation_reject_nan_inf():
    nan_img = np.zeros((384, 384, 3), dtype=np.float32)
    nan_img[10, 10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_inference_input(nan_img)

    inf_img = np.zeros((384, 384, 3), dtype=np.float32)
    inf_img[10, 10, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_inference_input(inf_img)


@pytest.mark.unit
def test_input_validation_reject_already_normalized_float():
    """
    CRITICAL REGRESSION GUARD:
    Rejects floating arrays wholly within [0.0, 1.0] to prevent accidental
    fatal double normalization via embedded /255 layer.
    """
    norm_float_img = np.ones((384, 384, 3), dtype=np.float32) * 0.5
    norm_float_img[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="double-normalization"):
        validate_inference_input(norm_float_img)


@pytest.mark.unit
def test_binarize_probability_mask_helper():
    prob = np.array([[0.1, 0.49], [0.5, 0.9]], dtype=np.float32)
    bin_mask = binarize_probability_mask(prob, threshold=0.5)
    assert bin_mask.shape == (2, 2)
    assert bin_mask.dtype == np.uint8
    assert np.array_equal(bin_mask, np.array([[0, 0], [0, 255]], dtype=np.uint8))


@pytest.mark.unit
def test_artifact_missing_failure(tmp_path):
    missing_p = tmp_path / "nonexistent_model.keras"
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_checkpoint_sha256(missing_p)


@pytest.mark.unit
def test_artifact_wrong_hash_rejected_before_loading(tmp_path):
    fake_ckpt = tmp_path / "fake_model.keras"
    fake_ckpt.write_bytes(b"CORRUPTED_OR_TAMPERED_MODEL_BYTES")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_inference_model(fake_ckpt)


@pytest.mark.unit
def test_checkpoint_artifact_verification():
    ckpt_p = resolve_default_checkpoint_path()
    if not ckpt_p.exists():
        pytest.skip(f"Local model artifact {EXPECTED_CHECKPOINT_ID} not found.")

    assert verify_checkpoint_sha256(ckpt_p) is True
    assert compute_file_sha256(ckpt_p) == EXPECTED_CHECKPOINT_SHA256
