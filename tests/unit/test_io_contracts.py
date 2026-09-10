"""
Tests for IO operations, path validation, and model-ready tensor constraints.
"""

from pathlib import Path
import numpy as np
import pytest

from comics_panel_extraction.io.image_io import (
    load_model_ready_tensor,
    load_binary_mask,
    load_probability_mask,
    save_probability_mask,
    save_binary_mask,
)


def test_load_model_ready_tensor_npy(tmp_path):
    # Valid uint8 (384, 384, 3)
    valid_arr = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
    p = tmp_path / "test_tensor.npy"
    np.save(str(p), valid_arr)

    loaded = load_model_ready_tensor(p)
    assert loaded.shape == (384, 384, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, valid_arr)


def test_load_model_ready_tensor_reject_float_dtype(tmp_path):
    # Reject float32 even if shape is (384, 384, 3)
    arr_float = np.ones((384, 384, 3), dtype=np.float32)
    p = tmp_path / "test_float.npy"
    np.save(str(p), arr_float)

    with pytest.raises(TypeError, match="must have canonical dtype np.uint8"):
        load_model_ready_tensor(p)


def test_probability_mask_reject_nan_inf(tmp_path):
    prob_nan = np.zeros((384, 384), dtype=np.float32)
    prob_nan[10, 10] = np.nan
    p_nan = tmp_path / "nan.npy"
    np.save(str(p_nan), prob_nan)

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        load_probability_mask(p_nan)

    with pytest.raises(ValueError, match="Cannot save probability mask containing NaN or Inf"):
        save_probability_mask(prob_nan, tmp_path / "out_nan.npy")


def test_overwrite_after_suffix_normalization(tmp_path):
    # Existing file is out.png
    existing = tmp_path / "out.png"
    existing.touch()

    # Caller passes path WITHOUT suffix ("out") and overwrite=False
    with pytest.raises(FileExistsError, match="Output file already exists"):
        save_binary_mask(np.zeros((384, 384), dtype=np.uint8), tmp_path / "out", overwrite=False)

    # With overwrite=True, succeeds
    save_binary_mask(np.zeros((384, 384), dtype=np.uint8), tmp_path / "out", overwrite=True)
    assert existing.exists()



def test_load_model_ready_tensor_reject_arbitrary_shape(tmp_path):
    # Invalid (448, 448, 3)
    arr_448 = np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8)
    p = tmp_path / "test_448.npy"
    np.save(str(p), arr_448)

    with pytest.raises(ValueError, match="Input tensor must have model-ready spatial shape"):
        load_model_ready_tensor(p)


def test_binary_mask_io_lossless(tmp_path):
    mask = np.zeros((384, 384), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    p = tmp_path / "mask.png"
    save_binary_mask(mask, p)
    assert p.exists()

    loaded = load_binary_mask(p)
    assert loaded.shape == (384, 384)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, mask)


def test_binary_mask_io_reject_non_binary(tmp_path):
    # Mask with {0, 1}
    mask_01 = np.zeros((384, 384), dtype=np.uint8)
    mask_01[50:100, 50:100] = 1
    p = tmp_path / "mask_01.npy"
    np.save(str(p), mask_01)

    with pytest.raises(ValueError, match="strictly in {0, 255}"):
        load_binary_mask(p)


def test_probability_mask_reject_non_floating(tmp_path):
    """save_probability_mask must strictly reject integer or boolean dtypes with TypeError."""
    int_mask = np.ones((384, 384), dtype=np.int32)
    with pytest.raises(TypeError, match="Probability mask must have floating-point dtype"):
        save_probability_mask(int_mask, tmp_path / "int_mask.npy")

    bool_mask = np.ones((384, 384), dtype=bool)
    with pytest.raises(TypeError, match="Probability mask must have floating-point dtype"):
        save_probability_mask(bool_mask, tmp_path / "bool_mask.npy")


def test_probability_mask_io(tmp_path):
    prob = np.random.uniform(0.0, 1.0, (384, 384)).astype(np.float32)
    p = tmp_path / "prob.npy"
    save_probability_mask(prob, p)
    assert p.exists()

    loaded = load_probability_mask(p)
    assert loaded.shape == (384, 384)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, prob)


def test_overwrite_protection(tmp_path):
    mask = np.zeros((384, 384), dtype=np.uint8)
    p = tmp_path / "out.png"
    save_binary_mask(mask, p)

    with pytest.raises(FileExistsError, match="Output file already exists"):
        save_binary_mask(mask, p, overwrite=False)

    # With overwrite=True
    save_binary_mask(mask, p, overwrite=True)
    assert p.exists()
