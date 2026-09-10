# Project State & Architecture Policy (v0.2.0)

## 1. Supported Operational Workflows
The `comics_panel_extraction` package supports three core workflows:

1. **M1 Operational Model Inference**:
   - Module: `comics_panel_extraction.inference`
   - Purpose: Load the M1 Comic U-Net++ checkpoint (`best_m1_unetpp_val_loss.keras`, SHA-256 `b6e4e63b...`) and generate raw panel segmentation masks from comic page images.
   - Status: **`M1_BEST_CURRENT_EMPIRICAL_COMIC_RECONSTRUCTION`**. (Exact historical architecture identity remains **`UNPROVEN`**).

2. **Canonical Recovered Historical Post-Processing**:
   - Module: `comics_panel_extraction.historical_legacy`
   - Purpose: Deterministic line-segment analysis and geometric panel boundary extraction matching legacy `Pipeline.ipynb` Cell 22.
   - Status: **`HIGH_CONFIDENCE_HISTORICAL_PIPELINE_FAMILY_RECOVERED`**. (Bit-identical $XOR = 0\text{ px}$ on canonical replay; 18/74 exact pages and 407.85 mean XOR px/page against archived historical final masks).

3. **Historical Dataset Evaluator**:
   - Module: `comics_panel_extraction.evaluation`
   - Purpose: Deterministic evaluation reproducing exact historical Table 1 (90.9661% mIoU) and Table 2 (92.3440% mIoU) metric suites.

---

## 2. Intentionally Pruned / Unsupported Scaffolding
The following components have been intentionally pruned from the active package:
- **M2 Manuscript-Faithful Post-Processing**: Experimental research code pruned to avoid maintaining superseded algorithmic variants.
- **M1 Retraining Scaffolding**: Data sequences, learning-rate schedules, and from-scratch training scripts removed; inference capability is preserved.
- **Legacy-First Round-06 Retraining Scaffolding**: The 87-layer / 50% dropout retraining experiment code was concluded and pruned (results preserved in `markdowns_dir/round_06_legacy_first_comic_model_reconstruction/`).
- **Migration Archaeology & Temporary JSONs**: Slices 01–06 temporary contracts and register files removed.

---

## 3. Known Non-Blocking Provenance Gaps
1. `EXACT_COMIC_CHECKPOINT_UNRESOLVED`: Historical Comic weights were omitted from the legacy repository. M1 serves as the authoritative reconstructed Comic model.
2. `EXACT_COMIC_RAW_PREDICTION_PRODUCER_UNRESOLVED`: The script generating raw comic prediction PNGs was omitted from git.
3. `EXACT_HISTORICAL_COMIC_TRAINING_REVISION_UNRESOLVED`: Exact historical Comic training configuration is lost.
4. `EXACT_POSTPROCESSING_PRODUCER_REVISION_UNRESOLVED`: A 407.85 px/page mean residual remains between the canonical historical post-processing pipeline and the archived 2024 final masks.
