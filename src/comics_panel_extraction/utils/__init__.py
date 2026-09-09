"""Parity comparison harness and array evaluation utilities."""

from comics_panel_extraction.utils.parity import (
    ArrayParityResult,
    ScalarParityResult,
    compare_arrays_exact,
    compare_binary_masks,
    compare_scalars,
    compute_array_sha256,
)

__all__ = [
    "ArrayParityResult",
    "ScalarParityResult",
    "compare_arrays_exact",
    "compare_binary_masks",
    "compare_scalars",
    "compute_array_sha256",
]
