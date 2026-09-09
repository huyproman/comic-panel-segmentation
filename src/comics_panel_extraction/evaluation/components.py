"""
Connected-component labeling and label-extraction primitives for evaluation.

Preserves exact legacy semantics from comics_analysis/post_process/metric.ipynb:
- Treats mask > 0.5 (or mask > 0) as binary foreground
- Applies skimage.measure.label without noise suppression
- Extracts unique component labels strictly greater than 0
"""

from typing import List, Tuple
import numpy as np
from skimage.measure import label


def extract_components(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies connected component labeling to single-channel binary mask.

    Parameters:
      mask: np.ndarray, 2D array (H, W), boolean or uint8 ({0, 1} or {0, 255})

    Returns:
      labeled_mask: np.ndarray (H, W) int, CCL labels from skimage.measure.label
      component_labels: np.ndarray 1D int, sorted unique labels > 0
    """
    if mask.ndim != 2:
        raise ValueError(f"Evaluation mask must be 2D, got shape {mask.shape}")

    # Legacy binarization check: if bool or uint8, threshold at > 0.5 (handles both {0, 1} and {0, 255})
    binary = mask > 0.5
    labeled = label(binary)
    unique_labels = np.unique(labeled[labeled > 0])
    return labeled, unique_labels
