import numpy as np
import pytest
from comics_panel_extraction.model.unetpp import build_unetpp
from comics_panel_extraction.inference import (
    load_inference_model,
    predict_probability_mask,
    binarize_probability_mask,
)

@pytest.mark.unit
def test_unetpp_architecture_structure():
    model = build_unetpp(input_shape=(448, 448, 3))
    assert model.count_params() == 9056769
    assert model.input_shape == (None, 448, 448, 3)
    assert model.output_shape == (None, 448, 448, 1)

@pytest.mark.unit
def test_unetpp_inference_smoke(tmp_path):
    model = build_unetpp(input_shape=(448, 448, 3))
    ckpt_p = tmp_path / "temp_unetpp.keras"
    model.save(str(ckpt_p))

    loaded_model = load_inference_model(ckpt_p)
    h, w = loaded_model.input_shape[1], loaded_model.input_shape[2]
    dummy_input = np.ones((h, w, 3), dtype=np.uint8) * 128
    prob = predict_probability_mask(dummy_input, model=loaded_model)
    assert prob.shape == (h, w)
    assert prob.dtype == np.float32
    assert 0.0 <= float(np.min(prob)) <= 1.0
    assert 0.0 <= float(np.max(prob)) <= 1.0
    binary = binarize_probability_mask(prob, threshold=0.5)
    assert binary.shape == (h, w)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 255})
