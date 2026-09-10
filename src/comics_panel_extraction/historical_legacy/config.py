"""Canonical historical configuration dataclasses preserving all legacy constants."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class HistoricalMorphologyConfig:
    mask_opening_kernel_size: Tuple[int, int] = (20, 20)
    canny_threshold1: int = 50
    canny_threshold2: int = 100
    edge_dilation_kernel_1: Tuple[int, int] = (5, 5)
    edge_dilation_iterations_1: int = 2
    edge_erosion_kernel_1: Tuple[int, int] = (5, 5)
    edge_erosion_iterations_1: int = 1
    edge_erosion_kernel_2: Tuple[int, int] = (3, 3)
    edge_erosion_iterations_2: int = 1


@dataclass(frozen=True)
class HistoricalHoughConfig:
    rho: float = 2.0
    theta_degrees: float = 1.0  # np.pi / 180
    threshold: int = 25
    min_line_length: float = 10.0
    max_line_gap: float = 30.0


@dataclass(frozen=True)
class HistoricalGroupingConfig:
    angle_tolerance: float = 3.0  # degrees
    distance_tolerance: float = 8.5
    # Parametric reference points as defined in Pipeline.ipynb Cell 3:
    # P1 = (5/7 W, 0.5 H), P2 = (1/6 W, 6/7 H), P3 = (3.5/9 W, 1/3 H)
    reference_formula_desc: str = "P1=(5/7W, 0.5H), P2=(1/6W, 6/7H), P3=(3.5/9W, 1/3H)"


@dataclass(frozen=True)
class HistoricalReconstructionConfig:
    coordinate_clip_min: int = 0
    coordinate_clip_max: int = 400
    inner_bandwidth: float = 13.0
    border_bandwidth: float = 10.0
    support_threshold: float = 0.75
    min_euclidean_distance: float = 40.0
    line_color: int = 0
    line_thickness: int = 3


@dataclass(frozen=True)
class HistoricalLegacyConfig:
    morphology: HistoricalMorphologyConfig = field(default_factory=HistoricalMorphologyConfig)
    hough: HistoricalHoughConfig = field(default_factory=HistoricalHoughConfig)
    grouping: HistoricalGroupingConfig = field(default_factory=HistoricalGroupingConfig)
    reconstruction: HistoricalReconstructionConfig = field(default_factory=HistoricalReconstructionConfig)
