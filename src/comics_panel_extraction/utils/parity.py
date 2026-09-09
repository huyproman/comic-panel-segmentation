"""
Generic parity comparator infrastructure.

Supports strict exact equality (default) and explicit contract-governed tolerances.
Does not contain legacy algorithm logic.
"""

from dataclasses import dataclass
import hashlib
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ArrayParityResult:
    """Detailed comparison report between two arrays/masks."""
    exact_match: bool
    shape_match: bool
    dtype_match: bool
    actual_shape: Tuple[int, ...]
    expected_shape: Tuple[int, ...]
    actual_dtype: str
    expected_dtype: str
    total_elements: int
    mismatch_elements: int
    mismatch_ratio: float
    actual_sha256: str
    expected_sha256: str
    actual_foreground_count: Optional[int] = None
    expected_foreground_count: Optional[int] = None
    max_absolute_difference: Optional[float] = None


@dataclass(frozen=True)
class ScalarParityResult:
    """Detailed scalar metric comparison report."""
    exact_match: bool
    within_tolerance: bool
    actual_value: float
    expected_value: float
    absolute_delta: float
    relative_delta: float
    allowed_abs_tol: float
    allowed_rel_tol: float


def compute_array_sha256(arr: np.ndarray) -> str:
    """Computes SHA-256 hash of decoded contiguous array bytes."""
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def compare_arrays_exact(actual: np.ndarray, expected: np.ndarray) -> ArrayParityResult:
    """Strict exact bitwise array comparison."""
    shape_match = (actual.shape == expected.shape)
    dtype_match = (actual.dtype == expected.dtype)
    actual_sha = compute_array_sha256(actual)
    expected_sha = compute_array_sha256(expected)
    
    if not shape_match:
        return ArrayParityResult(
            exact_match=False,
            shape_match=False,
            dtype_match=dtype_match,
            actual_shape=actual.shape,
            expected_shape=expected.shape,
            actual_dtype=str(actual.dtype),
            expected_dtype=str(expected.dtype),
            total_elements=int(actual.size),
            mismatch_elements=int(actual.size),
            mismatch_ratio=1.0,
            actual_sha256=actual_sha,
            expected_sha256=expected_sha,
        )
    
    diff_mask = (actual != expected)
    mismatch_count = int(np.sum(diff_mask))
    total_count = int(actual.size)
    mismatch_ratio = float(mismatch_count / total_count) if total_count > 0 else 0.0
    exact = (mismatch_count == 0) and dtype_match
    
    max_diff = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if total_count > 0 else 0.0
    
    return ArrayParityResult(
        exact_match=exact,
        shape_match=True,
        dtype_match=dtype_match,
        actual_shape=actual.shape,
        expected_shape=expected.shape,
        actual_dtype=str(actual.dtype),
        expected_dtype=str(expected.dtype),
        total_elements=total_count,
        mismatch_elements=mismatch_count,
        mismatch_ratio=mismatch_ratio,
        actual_sha256=actual_sha,
        expected_sha256=expected_sha,
        max_absolute_difference=max_diff,
    )


def compare_binary_masks(actual: np.ndarray, expected: np.ndarray) -> ArrayParityResult:
    """Compares binary masks (0/255 or boolean) including foreground pixel counts."""
    base_result = compare_arrays_exact(actual, expected)
    act_fg = int(np.sum(actual > 0))
    exp_fg = int(np.sum(expected > 0))
    
    return ArrayParityResult(
        exact_match=base_result.exact_match,
        shape_match=base_result.shape_match,
        dtype_match=base_result.dtype_match,
        actual_shape=base_result.actual_shape,
        expected_shape=base_result.expected_shape,
        actual_dtype=base_result.actual_dtype,
        expected_dtype=base_result.expected_dtype,
        total_elements=base_result.total_elements,
        mismatch_elements=base_result.mismatch_elements,
        mismatch_ratio=base_result.mismatch_ratio,
        actual_sha256=base_result.actual_sha256,
        expected_sha256=base_result.expected_sha256,
        actual_foreground_count=act_fg,
        expected_foreground_count=exp_fg,
        max_absolute_difference=base_result.max_absolute_difference,
    )


def compare_scalars(
    actual: float,
    expected: float,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
) -> ScalarParityResult:
    """Compares scalar metrics with explicit tolerance."""
    abs_delta = abs(actual - expected)
    denom = max(abs(actual), abs(expected))
    rel_delta = float(abs_delta / denom) if denom > 0.0 else 0.0
    
    exact = (abs_delta == 0.0)
    within_tol = (abs_delta <= abs_tol) or (rel_delta <= rel_tol)
    
    return ScalarParityResult(
        exact_match=exact,
        within_tolerance=within_tol,
        actual_value=float(actual),
        expected_value=float(expected),
        absolute_delta=float(abs_delta),
        relative_delta=float(rel_delta),
        allowed_abs_tol=float(abs_tol),
        allowed_rel_tol=float(rel_tol),
    )
