# Known Limitations & Historical Provenance Gaps

To prevent misleading claims of exact provenance, the project formally documents the surviving historical gaps. None of these gaps block reproduction or packaging.

---

## 1. Unresolved Historical Gaps

1. **Exact Historical Comic Checkpoint (`EXACT_COMIC_CHECKPOINT_UNRESOLVED`)**:
   - All 19 surviving `.keras` checkpoints in the legacy repository were trained on Manga/Vagabond datasets. Zero checkpoints with Comic training logs survive on disk.
   - The authors checked the resulting raw prediction PNGs into git, but omitted the model weights.
   - *Resolution*: M1 from-scratch Comic U-Net++ checkpoint (`best_m1_unetpp_val_loss.keras`) serves as the authoritative reconstructed Comic model.

2. **Exact Historical Postprocessing Producer Revision (`EXACT_POSTPROCESSING_PRODUCER_REVISION_UNRESOLVED`)**:
   - The canonical legacy postprocessing pipeline recovers the exact algorithm from `Pipeline.ipynb` Cells 5–22, matching published Table 2 metrics within 0.12 pp mIoU and achieving 18 bit-identical pages.
   - An average residual of 407.85 XOR pixels/page (0.20% canvas disagreement) remains between the canonical output and the archived `final_mask` PNGs.
   - *Resolution*: The canonical pipeline is promoted as a high-confidence historical family reconstruction.

3. **Residual Cause (`RESIDUAL_CAUSE_UNRESOLVED`)**:
   - The remaining 407 px/page residual cannot be definitively attributed to a specific revision or execution environment. Speculative explanations (such as OpenCV CPU rounding or uncommitted notebook iterations) are formally rejected as unproven.

4. **Historical Finalization Linkage**:
   - While `Post Process shenanigans.ipynb` Cell 33 provides direct evidence for a downstream batch filtering chain (`segregation(1000)` and `test_func(77, 80)`), no script explicitly names the target directory `unetpp_comic_pred/final_mask`.
   - *Resolution*: Core line-analysis is the default canonical replay; downstream filters are provided as modular extensions.
