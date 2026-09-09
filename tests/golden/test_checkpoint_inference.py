"""
Checkpoint execution tests verifying model prediction on synthetic input.
"""

from pathlib import Path
import numpy as np
import pytest

from comics_panel_extraction.inference import (
    load_inference_model,
    predict_probability_mask,
    resolve_default_checkpoint_path,
)


@pytest.mark.golden
def test_checkpoint_native_prediction():
    ckpt_p = resolve_default_checkpoint_path()
    if not ckpt_p.exists():
        pytest.skip("Model checkpoint artifact not present.")

    model = load_inference_model(ckpt_p)
    assert model.input_shape in [(None, 384, 384, 3), (None, 448, 448, 3)]
    assert model.output_shape in [(None, 384, 384, 1), (None, 448, 448, 1)]

    # Predict on deterministic synthetic intensity-domain image
    h, w = model.input_shape[1], model.input_shape[2]
    input_img = np.ones((h, w, 3), dtype=np.uint8) * 128
    prob_mask = predict_probability_mask(input_img, model=model)

    assert prob_mask.shape == (h, w)
    assert prob_mask.dtype == np.float32
    assert 0.0 <= float(np.min(prob_mask)) <= 1.0
    assert 0.0 <= float(np.max(prob_mask)) <= 1.0
