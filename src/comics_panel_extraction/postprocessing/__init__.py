"""Post-processing subsystem for comics_panel_extraction."""

import numpy as np

from comics_panel_extraction.historical_legacy.pipeline import (
    run_canonical_historical_postprocessing,
    is_connection_valid_historical,
)
from comics_panel_extraction.historical_legacy.config import (
    HistoricalLegacyConfig,
)


def refine_mask_with_line_segments(mask: np.ndarray, **kwargs) -> np.ndarray:
    """Canonical historical post-processing convenience wrapper."""
    out, _ = run_canonical_historical_postprocessing(mask)
    return out


def trace_line_analysis(mask: np.ndarray, **kwargs):
    """Canonical historical post-processing with diagnostics trace."""
    return run_canonical_historical_postprocessing(mask)


__all__ = [
    "run_canonical_historical_postprocessing",
    "is_connection_valid_historical",
    "HistoricalLegacyConfig",
    "refine_mask_with_line_segments",
    "trace_line_analysis",
]
