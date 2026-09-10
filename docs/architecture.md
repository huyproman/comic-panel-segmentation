# Architecture & System Design

The `comics_panel_extraction` package is architected to separate canonical operational reproduction from scientific research reconstruction.

---

## 1. System Structure

```
comics_panel_extraction/
├── src/comics_panel_extraction/
│   ├── historical_legacy/       # Canonical historical reproduction pipeline
│   │   ├── config.py            # Parameters from Pipeline.ipynb Cells 3-22
│   │   ├── pipeline.py          # Fixed 384 reference geometry, Bresenham validation
│   │   └── __init__.py
│   ├── evaluation/              # Historical evaluation harness (Lanes A & B)
│   │   ├── dataset.py           # Corpus evaluation
│   │   ├── page.py              # Page-level matching & aggregation
│   │   └── metrics.py           # mIoU, mDice, Panel/Page Accuracy
│   ├── model/                   # Operational M1 U-Net++ architecture
│   │   └── m1.py                # 15-node U-Net++ model definition
│   ├── io/                      # Image IO & fixture loading
│   ├── config/                  # Legacy operational settings
│   └── cli.py                   # Unified CLI entrypoint
├── tests/                       # 141 regression, unit, and parity tests
├── configs/                     # YAML profile definitions
└── docs/                        # Complete technical documentation
```

---

## 2. Canonical Historical Pipeline Workflow

The canonical historical pipeline executes the exact stages recovered from `comics_analysis/post_process/Pipeline.ipynb`:
1. **Mask Smoothing**: Morphological opening with 20×20 rectangular structuring element.
2. **Canny Edge Detection**: `threshold1=50, threshold2=100`.
3. **Edge Morphing**: 3-stage morphological sequence on Canny edge map (dilate 5×5 ×2 $\to$ erode 5×5 ×1 $\to$ erode 3×3 ×1).
4. **Hough Transform**: Probabilistic Hough line extraction ($\rho=2.0, \theta=1^\circ, \text{threshold}=25$).
5. **Orientation Classification**: Maps lines to horizontal or vertical based on $45^\circ$ threshold.
6. **Redundant Line Grouping**: Groups parallel lines within $8.5\text{ px}$ perpendicular distance to 3 fixed reference points ($P_1, P_2, P_3$) derived from $384\times 384$ geometry. Angle tolerance is $3^\circ$ or $177^\circ$. Lines within groups are voted by mode or median angle (no geometric rotation).
7. **Line Intersection**: Calculates intersections among vertical, horizontal, and boundary lines (clipped to $[0, 400]$).
8. **MeanShift Clustering**: Clusters inner intersections (bandwidth 13.0) and border intersections (bandwidth 10.0) to identify panel corner centroids.
9. **Bresenham Connection Validation**: Evaluates all centroid pairs separated by $>40\text{ px}$. A connection is validated only if $\ge 75\%$ of pixels along the Bresenham line overlap detected Hough lines.
10. **Boundary Carving**: Carves confirmed boundary lines ($\text{color}=0, \text{thickness}=3$) directly onto the smoothed mask.
