from pathlib import Path
import numpy as np
import pytest
from comics_panel_extraction.model.unetpp import build_unetpp
from comics_panel_extraction.inference.predictor import predict_mask
from comics_panel_extraction.postprocessing.line_analysis import run_line_segment_postprocessing
from comics_panel_extraction.evaluation.evaluator import PanelInstanceEvaluator
def test_model_construction():
    model = build_unetpp(input_shape=(448, 448, 3))
    assert model.count_params() == 9056769
    assert model.output_shape == (None, 448, 448, 1)
def test_postprocessing_default_integration():
    model = build_unetpp(input_shape=(448, 448, 3))
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[50:180, 50:180] = 255
    img[220:350, 50:180] = 255
    final_mask, raw_mask, diag = predict_mask(img, model, apply_postprocessing=True)
    assert final_mask.shape == (400, 400)
    assert raw_mask.shape == (400, 400)
    assert "validated_connections" in diag
    final_raw, raw_mask2, diag2 = predict_mask(img, model, apply_postprocessing=False)
    assert np.array_equal(final_raw, raw_mask2)
    assert len(diag2) == 0
def test_evaluator_hungarian_matching():
    evaluator = PanelInstanceEvaluator()
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:40, 10:40] = 1
    gt[60:90, 60:90] = 1
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:40, 10:40] = 1
    pred[60:90, 60:90] = 1
    res = evaluator.evaluate_page(pred, gt)
    assert res["page_instance_miou"] == pytest.approx(1.0, abs=1e-4)
    assert res["page_instance_mdice"] == pytest.approx(1.0, abs=1e-4)
    assert res["panel_tp"] == 2
    assert res["panel_fp"] == 0
    assert res["panel_fn"] == 0
