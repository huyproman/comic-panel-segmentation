from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import yaml

class BehaviorProfile(str, Enum):
    DEFAULT = "default"

@dataclass
class PreprocessingConfig:
    canny_low: int = 50
    canny_high: int = 100
    opening_kernel_size: List[int] = field(default_factory=lambda: [20, 20])
    bridge_dilate_size: List[int] = field(default_factory=lambda: [5, 5])
    bridge_erode1_size: List[int] = field(default_factory=lambda: [5, 5])
    bridge_erode2_size: List[int] = field(default_factory=lambda: [3, 3])

@dataclass
class HoughConfig:
    rho: int = 2
    theta_divisor: int = 180
    threshold: int = 25
    min_line_length: int = 10
    max_line_gap: int = 30

@dataclass
class LineGroupingConfig:
    min_angle: float = 0.0
    max_angle: float = 180.0
    angle_step: float = 1.0
    reference_geometry: List[int] = field(default_factory=lambda: [384, 384])

@dataclass
class IntersectionConfig:
    cluster_bandwidth_inner: float = 13.0
    cluster_bandwidth_border: float = 10.0
    coord_clip_min: float = 0.0
    coord_clip_max: float = 400.0

@dataclass
class LineVerificationConfig:
    edge_support_threshold: float = 0.75
    min_line_length_euclidean: float = 40.0
    line_thickness: int = 3
    line_color: int = 0

@dataclass
class ComponentFilteringConfig:
    min_component_size: int = 1000

@dataclass
class InferenceConfig:
    input_height: int = 448
    input_width: int = 448
    input_channels: int = 3
    threshold: float = 0.5
    raw_only: bool = False

@dataclass
class TrainingSettingsConfig:
    epochs: int = 150
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    output_dir: str = "runs/train"
    dataset_dir: Optional[str] = None

@dataclass
class AppConfig:
    profile: BehaviorProfile = BehaviorProfile.DEFAULT
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)
    line_grouping: LineGroupingConfig = field(default_factory=LineGroupingConfig)
    intersections: IntersectionConfig = field(default_factory=IntersectionConfig)
    line_verification: LineVerificationConfig = field(default_factory=LineVerificationConfig)
    component_filtering: ComponentFilteringConfig = field(default_factory=ComponentFilteringConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingSettingsConfig = field(default_factory=TrainingSettingsConfig)

def get_default_config() -> AppConfig:
    return AppConfig(profile=BehaviorProfile.DEFAULT)

def load_config_from_yaml(path: str) -> AppConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at '{path}' must contain a key-value mapping.")

    profile_str = data.get("profile", "default")
    if profile_str not in ["default"]:
        raise ValueError(f"Unsupported profile: '{profile_str}'. Only 'default' is authorized.")

    cfg = AppConfig(profile=BehaviorProfile(profile_str))
    for section_name in ["preprocessing", "hough", "line_grouping", "intersections", "line_verification", "component_filtering", "inference", "training"]:
        if section_name in data:
            section_data = data[section_name]
            section_obj = getattr(cfg, section_name)
            for k, v in section_data.items():
                if hasattr(section_obj, k):
                    setattr(section_obj, k, v)
    return cfg

def validate_config_immutability(cfg: AppConfig, overrides: Optional[Dict[str, Any]] = None) -> None:
    if overrides:
        for k in overrides:
            raise ValueError(f"Cannot override frozen default parameter '{k}'.")
