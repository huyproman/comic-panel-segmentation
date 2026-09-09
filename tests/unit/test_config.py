"""Unit tests for configuration subsystem and legacy_2024 profile."""

from pathlib import Path
import pytest
from comics_panel_extraction.config import (
    BehaviorProfile,
    AppConfig,
    load_config,
    get_legacy_2024_config,
)


@pytest.mark.unit
def test_legacy_2024_frozen_defaults():
    """Verifies that legacy_2024 profile has exact frozen parameters."""
    cfg = get_legacy_2024_config()
    assert cfg.profile == BehaviorProfile.LEGACY_2024
    
    # Section 2.3 segregation parameters
    assert cfg.component_filter.size_threshold == 1000
    assert cfg.component_filter.connectivity == 8
    
    # Line analysis parameters
    assert cfg.line_analysis.smooth_kernel_size == (20, 20)
    assert cfg.line_analysis.canny_thresh1 == 50.0
    assert cfg.line_analysis.canny_thresh2 == 100.0
    assert cfg.line_analysis.hough_threshold == 25
    assert cfg.line_analysis.meanshift_bandwidth_inner == 13.0
    assert cfg.line_analysis.meanshift_bandwidth_border == 10.0
    assert cfg.line_analysis.bresenham_color_ratio_thresh == 0.75
    assert cfg.line_analysis.drawn_line_thickness == 3
    
    # Evaluation semantics
    assert cfg.evaluation.dice_threshold == 0.9
    assert cfg.evaluation.panel_accuracy_direction == "gt_to_pred"
    assert cfg.evaluation.precision_recall_direction == "pred_to_gt"
    assert cfg.evaluation.page_accuracy_ignore_false_positives is True


@pytest.mark.unit
def test_load_config_from_yaml(repo_root: Path):
    """Verifies loading configuration from YAML file."""
    yaml_path = repo_root / "configs" / "legacy_2024.yaml"
    cfg = load_config(yaml_path)
    assert cfg.profile == BehaviorProfile.LEGACY_2024
    assert cfg.component_filter.size_threshold == 1000
