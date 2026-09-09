# Comic Panel Extraction

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.10-green.svg)](pyproject.toml)

Standalone, verifiable implementation for **Comic Panel Extraction using U-Net++ Deep Learning and Line Segment Analysis**.

---

## 1. Supported Capabilities

This package provides a clean, production-grade implementation of:

1. **M1 Operational Model Inference**: Run inference with the authoritative reconstructed Comic U-Net++ model (`best_m1_unetpp_val_loss.keras`) to generate raw panel probability maps and binary segmentation masks.
2. **Canonical Historical Post-Processing**: Refine raw prediction masks using recovered legacy line segment analysis (Probabilistic Hough Transform, geometric clustering, and Bresenham centroid connectivity).
3. **Historical Evaluation Suite**: Reproduce exact 7-of-7 metric calculations (mIoU, mDice, Panel Accuracy, Page Accuracy, Precision, Recall, F1) on predicted masks against ground truth.

---

## 2. Quickstart

### Installation
```bash
pip install comics-panel-extraction
```

### CLI Usage

```bash
# 1. Model inference on a comic image
comics-panel-extraction infer page.jpg --output raw_mask.png --binarize

# 2. Refine raw mask with historical line-segment postprocessing
comics-panel-extraction postprocess raw_mask.png --output final_mask.png

# 3. Evaluate prediction masks against ground truth
comics-panel-extraction evaluate --predictions preds/ --ground-truth gt/ --output-json metrics.json

# 4. Canonical historical postprocessing replay across a directory
comics-panel-extraction historical-replay --input-dir raw_masks/ --output-dir final_masks/
```

### Python API

```python
import cv2
from comics_panel_extraction import (
    load_inference_model,
    predict_probability_mask,
    binarize_probability_mask,
    run_canonical_historical_postprocessing,
    evaluate_dataset,
)

# 1. Inference
model = load_inference_model()
image = cv2.imread("page.jpg")
prob = predict_probability_mask(image, model=model)
raw_mask = binarize_probability_mask(prob, threshold=0.5)

# 2. Canonical Historical Post-Processing
final_mask, diagnostics = run_canonical_historical_postprocessing(raw_mask)

# 3. Evaluation
metrics = evaluate_dataset(gt_dir="gt/", pred_dir="preds/")
print(f"mIoU: {metrics.mean_iou * 100:.4f}%")
```

---

## 3. Scientific & Epistemic Status

- **Post-Processing**: Reconstructed with high confidence. Matches canonical reconstructed outputs bit-for-bit ($XOR = 0\text{ px}$) across all 74 benchmark pages.
- **Model**: M1 is the **best current empirical Comic reconstruction** (88.63% raw mIoU; 89.46% postprocessed mIoU). Its exact historical architecture identity remains unproven as original historical Comic weights were omitted from git.
- **Model Recreation Policy**: Model recreation research is paused.

For detailed architecture, provenance, and known limitations, see `docs/PROJECT_STATE.md`.
