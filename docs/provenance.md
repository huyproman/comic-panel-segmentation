# Provenance & Four-Layer Truth Model

To maintain scientific integrity across conflicting sources, the project enforces a strict four-layer epistemic partition:

```
+-----------------------------------------------------------------------------------+
| LAYER 1: MANUSCRIPT-DESCRIBED METHOD                                              |
| - Authority: main.pdf (SHA-256: f9dbf652...)                                      |
| - Contents: Equations and prose (BatchNorm+ReLU U-Net++, trigonometric rotation,  |
|   random acute reference points, Section 2.3 area/perimeter filtering)            |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| LAYER 2: HISTORICAL EXECUTED IMPLEMENTATION                                       |
| - Authority: comics_analysis repository working tree                              |
| - Contents: Executable notebook cells (Pipeline.ipynb)                            |
| - Quirks: Fixed 384 reference points, no geometric line rotation, 0..400 clipping |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| LAYER 3: ARCHIVED EXPERIMENTAL TRUTH (EMPIRICAL ORACLE)                           |
| - Authority: comics_analysis/preds/comic_preds/unetpp_comic_pred/                 |
| - Contents: 74 raw masks (Table 1), 74 final masks (Table 2), 74 ground truth    |
| - Role: Empirical ground truth replicating published numbers bit-for-bit          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| LAYER 4: MODERN RECONSTRUCTED STANDALONE IMPLEMENTATIONS                          |
| - Authority: comics_panel_extraction repository                                   |
| - Contents:                                                                       |
|   1. historical-replay (Canonical legacy path -> 92.46% mIoU)                     |
|   2. reconstructed (M1 Comic model + legacy path -> 89.46% mIoU)                  |
|   3. manuscript-faithful (M1 Comic model + M2 manuscript postprocess -> 87.70%)   |
+-----------------------------------------------------------------------------------+
```

---

## Major Historical Divergences Documented

1. **Reference Geometry**: Manuscript Section 2.2.4 specifies random acute triangle reference points. Historical code hardcodes parametric formulas derived from fixed 384×384 geometry.
2. **Geometric Rotation**: Manuscript Section 2.2.4 provides explicit trigonometric rotation formulas around line midpoints. Historical code uses angle mode/median only to discard outliers without rotating line coordinates.
3. **Canvas Clipping**: Historical intersections are clipped to $0 \le x, y \le 400$, whereas the comic canvas is $448 \times 448$.
4. **Morphology Order**: Historical code applies morphological dilation and erosion directly onto the Canny edge map before calling Hough lines.
