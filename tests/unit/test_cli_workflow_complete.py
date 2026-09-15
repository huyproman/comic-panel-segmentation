import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from comics_panel_extraction.cli import main
from comics_panel_extraction.model.unetpp import build_unetpp

def test_cli_train_dataset_smoke(tmp_path):
    data_dir = tmp_path / "dataset"
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for i in range(2):
        im = np.zeros((100, 100, 3), dtype=np.uint8)
        im[20:80, 20:80] = 200
        mk = np.zeros((100, 100), dtype=np.uint8)
        mk[20:80, 20:80] = 255
        cv2.imwrite(str(img_dir / f"page_{i}.png"), im)
        cv2.imwrite(str(mask_dir / f"page_{i}.png"), mk)

    out_dir = tmp_path / "train_run"
    code = main([
        "train",
        "--dataset", str(data_dir),
        "--epochs", "1",
        "--batch-size", "2",
        "--output-dir", str(out_dir)
    ])
    assert code == 0
    assert (out_dir / "latest_model.keras").exists()

def test_cli_infer_and_raw_only_and_postprocess(tmp_path):
    model = build_unetpp(input_shape=(448, 448, 3))
    ckpt_p = tmp_path / "dummy_model.keras"
    model.save(str(ckpt_p))

    img = np.zeros((448, 448, 3), dtype=np.uint8)
    img[50:200, 50:200] = 255
    img_p = tmp_path / "sample_page.png"
    cv2.imwrite(str(img_p), img)

    out_final = tmp_path / "final_mask.png"
    out_raw = tmp_path / "raw_mask.png"
    code = main([
        "infer",
        str(img_p),
        "-c", str(ckpt_p),
        "-o", str(out_final),
        "--raw-output", str(out_raw)
    ])
    assert code == 0
    assert out_final.exists()
    assert out_raw.exists()

    final_m = cv2.imread(str(out_final), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(final_m)).issubset({0, 255})

    out_raw_only = tmp_path / "raw_only.png"
    code2 = main([
        "infer",
        str(img_p),
        "-c", str(ckpt_p),
        "-o", str(out_raw_only),
        "--raw-only"
    ])
    assert code2 == 0
    assert out_raw_only.exists()

    out_post = tmp_path / "standalone_post.png"
    code3 = main([
        "postprocess",
        str(out_raw),
        "-o", str(out_post)
    ])
    assert code3 == 0
    assert out_post.exists()
    post_m = cv2.imread(str(out_post), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(post_m)).issubset({0, 255})

def test_cli_evaluate(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    cv2.imwrite(str(gt_dir / "p1.png"), mask)
    cv2.imwrite(str(pred_dir / "p1.png"), mask)

    out_json = tmp_path / "metrics.json"
    code = main([
        "evaluate",
        "-g", str(gt_dir),
        "-p", str(pred_dir),
        "-o", str(out_json)
    ])
    assert code == 0
    assert out_json.exists()
    data = json.load(open(out_json))
    assert data["canonical_instance_miou"] == pytest.approx(1.0, abs=1e-4)

def test_cli_evaluate_strict_keys_and_missing_key_failure(tmp_path, monkeypatch):
    gt_dir = tmp_path / "gt2"
    pred_dir = tmp_path / "pred2"
    gt_dir.mkdir()
    pred_dir.mkdir()

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    cv2.imwrite(str(gt_dir / "p1.png"), mask)
    cv2.imwrite(str(pred_dir / "p1.png"), mask)

    import comics_panel_extraction.cli as cli_mod
    def mock_eval_incomplete(results):
        return {"instance_miou": 0.5}

    monkeypatch.setattr(cli_mod, "evaluate_instance_dataset", mock_eval_incomplete)

    out_json = tmp_path / "fail_metrics.json"
    with pytest.raises(KeyError, match="Expected evaluation metric 'canonical_instance_miou' missing"):
        cli_mod.main([
            "evaluate",
            "-g", str(gt_dir),
            "-p", str(pred_dir),
            "-o", str(out_json)
        ])
    assert not out_json.exists()
