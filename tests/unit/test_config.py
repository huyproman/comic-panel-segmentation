from pathlib import Path
import pytest
from comics_panel_extraction.config import (
    BehaviorProfile,
    AppConfig,
    load_config,
    get_default_config,
)

@pytest.mark.unit
def test_default_config_values():
    cfg = get_default_config()
    assert cfg.profile == BehaviorProfile.DEFAULT
    assert cfg.component_filtering.min_component_size == 1000
    assert cfg.preprocessing.opening_kernel_size == [20, 20]
    assert cfg.preprocessing.canny_low == 50
    assert cfg.preprocessing.canny_high == 100
    assert cfg.hough.threshold == 25
    assert cfg.intersections.cluster_bandwidth_inner == 13.0
    assert cfg.intersections.cluster_bandwidth_border == 10.0
    assert cfg.line_verification.edge_support_threshold == 0.75
    assert cfg.line_verification.line_thickness == 3
    assert cfg.inference.input_height == 448
    assert cfg.inference.input_width == 448

@pytest.mark.unit
def test_load_config_from_yaml(repo_root: Path):
    yaml_path = repo_root / "configs" / "default.yaml"
    cfg = load_config(str(yaml_path))
    assert cfg.profile == BehaviorProfile.DEFAULT
    assert cfg.component_filtering.min_component_size == 1000
    assert cfg.inference.input_height == 448
    assert cfg.inference.input_width == 448
