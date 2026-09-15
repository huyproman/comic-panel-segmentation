from pathlib import Path
import numpy as np
import pytest
from comics_panel_extraction.inference import (
    binarize_probability_mask,
    compute_file_sha256,
    load_inference_model,
    predict_probability_mask,
    validate_checkpoint,
    validate_inference_input,
)
from comics_panel_extraction.model.unetpp import build_unetpp

@pytest.mark.unit
def test_input_validation_valid():
    img = np.zeros((448, 448, 3), dtype=np.uint8)
    validate_inference_input(img, expected_shape=(448, 448, 3))

@pytest.mark.unit
def test_input_validation_reject_wrong_shape():
    with pytest.raises(ValueError, match="Expected shape"):
        validate_inference_input(np.zeros((448, 448), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected shape"):
        validate_inference_input(np.zeros((1, 448, 448, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected shape"):
        validate_inference_input(np.zeros((448, 448, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected shape"):
        validate_inference_input(np.zeros((448, 448, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="Expected shape"):
        validate_inference_input(np.zeros((224, 224, 3), dtype=np.uint8))

@pytest.mark.unit
def test_input_validation_reject_nan_inf():
    nan_img = np.zeros((448, 448, 3), dtype=np.float32)
    nan_img[10, 10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_inference_input(nan_img)
    inf_img = np.zeros((448, 448, 3), dtype=np.float32)
    inf_img[10, 10, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_inference_input(inf_img)

@pytest.mark.unit
def test_input_validation_reject_out_of_range_float():
    bad_float_img = np.ones((448, 448, 3), dtype=np.float32) * 2.5
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        validate_inference_input(bad_float_img)

@pytest.mark.unit
def test_binarize_probability_mask_helper():
    prob = np.array([[0.1, 0.49], [0.5, 0.9]], dtype=np.float32)
    bin_mask = binarize_probability_mask(prob, threshold=0.5)
    assert bin_mask.shape == (2, 2)
    assert bin_mask.dtype == np.uint8
    assert np.array_equal(bin_mask, np.array([[0, 0], [255, 255]], dtype=np.uint8))

@pytest.mark.unit
def test_checkpoint_validation_helpers(tmp_path):
    f = tmp_path / "dummy_file.txt"
    f.write_text("hello world")
    assert validate_checkpoint(f) is True
    assert validate_checkpoint(tmp_path / "nonexistent.txt") is False

    sha = compute_file_sha256(f)
    assert validate_checkpoint(f, expected_sha256=sha) is True
    assert validate_checkpoint(f, expected_sha256="wrong_sha") is False

@pytest.mark.unit
def test_load_inference_model_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        load_inference_model(checkpoint_path=None)

@pytest.mark.unit
def test_load_inference_model_nonexistent_checkpoint(tmp_path):
    bad_path = tmp_path / "does_not_exist.keras"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_inference_model(checkpoint_path=bad_path)

@pytest.mark.unit
def test_load_inference_model_valid_checkpoint(tmp_path):
    model = build_unetpp(input_shape=(448, 448, 3))
    ckpt_p = tmp_path / "valid.keras"
    model.save(str(ckpt_p))

    loaded = load_inference_model(checkpoint_path=ckpt_p)
    assert loaded.input_shape == (None, 448, 448, 3)
    assert loaded.output_shape == (None, 448, 448, 1)
