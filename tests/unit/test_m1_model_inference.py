"""
Regression test for M1 model architecture and inference functionality.
"""

import numpy as np
import pytest
from comics_panel_extraction.model.m1 import build_manuscript_unetpp
from comics_panel_extraction.inference import (
    load_inference_model,
    predict_probability_mask,
    binarize_probability_mask,
    resolve_default_checkpoint_path,
)


def test_m1_architecture_structure():
    """Verify M1 model architecture parameter count and layers."""
    model = build_manuscript_unetpp(input_shape=(448, 448, 3))
    assert model.count_params() == 9056769
    assert model.input_shape == (None, 448, 448, 3)
    assert model.output_shape == (None, 448, 448, 1)


def test_m1_inference_smoke():
    """Verify inference pipeline on synthetic input."""
    ckpt_p = resolve_default_checkpoint_path()
    if not ckpt_p.exists():
        pytest.skip(f"M1 checkpoint not found at {ckpt_p}")

    model = load_inference_model(ckpt_p)
    h, w = model.input_shape[1], model.input_shape[2]
    dummy_input = np.ones((h, w, 3), dtype=np.uint8) * 128
    prob = predict_probability_mask(dummy_input, model=model)

    assert prob.shape == (h, w)
    assert prob.dtype == np.float32
    assert 0.0 <= float(np.min(prob)) <= 1.0
    assert 0.0 <= float(np.max(prob)) <= 1.0

    binary = binarize_probability_mask(prob, threshold=0.5)
    assert binary.shape == (h, w)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
