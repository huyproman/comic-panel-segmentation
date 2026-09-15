from typing import List, Tuple
import numpy as np
from skimage.measure import label
def extract_components(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mask.ndim != 2:
        raise ValueError(f"Evaluation mask must be 2D, got shape {mask.shape}")
    binary = mask > 0.5
    labeled = label(binary)
    unique_labels = np.unique(labeled[labeled > 0])
    return labeled, unique_labels
