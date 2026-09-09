"""
Validated configuration models, behavior profiles, and operational settings.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml


class BehaviorProfile(str, Enum):
    """Supported behavioral compatibility profiles."""
    LEGACY_2024 = "legacy_2024"


@dataclass(frozen=True)
class LineAnalysisConfig:
    """Configuration for line segment post-processing."""
    smooth_kernel_size: Tuple[int, int] = (20, 20)
    smooth_morph_op: str = "cv2.MORPH_OPEN"
    smooth_struct_elem: str = "cv2.MORPH_RECT"
    canny_thresh1: float = 50.0
    canny_thresh2: float = 100.0
    bridge_dilate_kernel: Tuple[int, int] = (5, 5)
    bridge_dilate_iters: int = 2
    bridge_erode1_kernel: Tuple[int, int] = (5, 5)
    bridge_erode1_iters: int = 1
    bridge_erode2_kernel: Tuple[int, int] = (3, 3)
    bridge_erode2_iters: int = 1
    hough_rho: float = 2.0
    hough_theta_div: float = 180.0
    hough_threshold: int = 25
    hough_min_line_length: float = 10.0
    hough_max_line_gap: float = 30.0
    ref_point_ratios: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (5.0 / 7.0, 0.5),
        (1.0 / 6.0, 6.0 / 7.0),
        (3.5 / 9.0, 1.0 / 3.0),
    )
    intersection_bound_min: int = 0
    intersection_bound_max: int = 400
    meanshift_bandwidth_inner: float = 13.0
    meanshift_bandwidth_border: float = 10.0
    bresenham_color_ratio_thresh: float = 0.75
    bresenham_min_distance: float = 40.0
    drawn_line_thickness: int = 3


@dataclass(frozen=True)
class ComponentFilterConfig:
    """Configuration for Section 2.3 connected component filtering."""
    size_threshold: int = 1000
    connectivity: int = 8


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluation metrics preserving legacy semantics."""
    dice_threshold: float = 0.9
    connectivity: int = 8
    panel_accuracy_direction: str = "gt_to_pred"
    precision_recall_direction: str = "pred_to_gt"
    page_accuracy_ignore_false_positives: bool = True


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for model inference runner."""
    model_checkpoint_rel_path: str = ".local_artifacts/best_model_unetpp_1.keras"
    expected_checkpoint_sha256: str = "a33299794f014c57b2590385f85d189c592117452ab6a93b2552082569eb29c7"
    input_height: int = 384
    input_width: int = 384
    input_channels: int = 3
    input_color_space: str = "RGB"
    embedded_lambda_div_255: bool = True


@dataclass
class OperationalConfig:
    """Operational settings explicitly separated from research algorithm profile."""
    checkpoint_path: Optional[str] = None
    binarization_threshold: Optional[float] = None
    output_dir: Optional[str] = None
    overwrite: bool = False


@dataclass
class AppConfig:
    """Master application configuration."""
    profile: BehaviorProfile = BehaviorProfile.LEGACY_2024
    line_analysis: LineAnalysisConfig = field(default_factory=LineAnalysisConfig)
    component_filter: ComponentFilterConfig = field(default_factory=ComponentFilterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    operational: OperationalConfig = field(default_factory=OperationalConfig)


def get_legacy_2024_config() -> AppConfig:
    """Returns the immutable legacy_2024 frozen configuration."""
    return AppConfig(profile=BehaviorProfile.LEGACY_2024)


def _normalize_config_val(val: Any) -> Any:
    """Recursively normalizes lists and floats for equality comparison against frozen config values."""
    if isinstance(val, (list, tuple)):
        return tuple(_normalize_config_val(x) for x in val)
    if isinstance(val, float):
        return round(val, 6)
    return val


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads and validates a YAML/JSON configuration file.
    Rejects unknown top-level and operational keys.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping/dictionary, got {type(data).__name__}")

    # Validate top-level keys
    allowed_top_keys = {"profile", "operational", "line_analysis", "component_filter", "evaluation", "inference"}
    unknown_top = set(data.keys()) - allowed_top_keys
    if unknown_top:
        raise ValueError(f"Unknown configuration key(s): {sorted(list(unknown_top))}")

    # Validate profile
    profile_str = data.get("profile", "legacy_2024")
    if profile_str != "legacy_2024":
        raise ValueError(f"Unsupported profile: '{profile_str}'. Only 'legacy_2024' is authorized.")

    # In legacy_2024 profile, if algorithm configuration sections are supplied,
    # validate that their parameters match frozen legacy_2024 values to prevent silent ignoring of overrides.
    frozen_cfg = get_legacy_2024_config()
    for section_name, frozen_obj in [
        ("line_analysis", frozen_cfg.line_analysis),
        ("component_filter", frozen_cfg.component_filter),
        ("evaluation", frozen_cfg.evaluation),
        ("inference", frozen_cfg.inference),
    ]:
        if section_name in data:
            sec_dict = data[section_name]
            if not isinstance(sec_dict, dict):
                raise ValueError(f"{section_name} section must be a dictionary, got {type(sec_dict).__name__}")
            for k, val in sec_dict.items():
                if not hasattr(frozen_obj, k):
                    raise ValueError(
                        f"Unknown key '{k}' in algorithm section '{section_name}'. "
                        f"Only frozen legacy_2024 parameters are authorized."
                    )
                frozen_val = getattr(frozen_obj, k)
                # Compare normalized value with frozen default
                norm_val = _normalize_config_val(val)
                norm_frozen = _normalize_config_val(frozen_val)
                if norm_val != norm_frozen:
                    raise ValueError(
                        f"Cannot override frozen legacy_2024 parameter '{section_name}.{k}'. "
                        f"Expected frozen value {frozen_val}, got {val}."
                    )

    # Validate profile
    profile_str = data.get("profile", "legacy_2024")
    if profile_str != "legacy_2024":
        raise ValueError(f"Unsupported profile: '{profile_str}'. Only 'legacy_2024' is authorized.")

    # Validate operational section
    if "operational" in data:
        op = data["operational"]
        if not isinstance(op, dict):
            raise ValueError(f"operational section must be a dictionary, got {type(op).__name__}")
        allowed_op_keys = {"checkpoint_path", "binarization_threshold", "output_dir", "overwrite"}
        unknown_op = set(op.keys()) - allowed_op_keys
        if unknown_op:
            raise ValueError(f"Unknown operational configuration key(s): {sorted(list(unknown_op))}")

    return data


def resolve_operational_settings(
    cli_checkpoint: Optional[str] = None,
    cli_threshold: Optional[float] = None,
    cli_output_dir: Optional[str] = None,
    cli_overwrite: bool = False,
    config_path: Optional[Union[str, Path]] = None,
) -> Tuple[OperationalConfig, Dict[str, str]]:
    """
    Deterministically resolves operational settings with field-level provenance tracking:
      Precedence: CLI argument > Environment variable > Config file value > Default
    """
    sources: Dict[str, str] = {}
    cfg_data = load_config_file(config_path) if config_path else {}
    op_data = cfg_data.get("operational", {})

    # 1. Checkpoint Path
    # Precedence: CLI > COMICS_CHECKPOINT_PATH > Config > Default
    if cli_checkpoint:
        resolved_ckpt = cli_checkpoint
        sources["checkpoint_path"] = "CLI"
    elif os.environ.get("COMICS_CHECKPOINT_PATH"):
        resolved_ckpt = os.environ.get("COMICS_CHECKPOINT_PATH")
        sources["checkpoint_path"] = "ENV"
    elif op_data.get("checkpoint_path"):
        resolved_ckpt = op_data.get("checkpoint_path")
        sources["checkpoint_path"] = "CONFIG_FILE"
    else:
        resolved_ckpt = None
        sources["checkpoint_path"] = "DEFAULT"

    # 2. Binarization Threshold
    # Precedence: CLI > Config > None (NO default 0.5)
    if cli_threshold is not None:
        resolved_thresh = float(cli_threshold)
        sources["binarization_threshold"] = "CLI"
    elif op_data.get("binarization_threshold") is not None:
        resolved_thresh = float(op_data["binarization_threshold"])
        sources["binarization_threshold"] = "CONFIG_FILE"
    else:
        resolved_thresh = None
        sources["binarization_threshold"] = "NONE"

    # Validate threshold if present
    if resolved_thresh is not None:
        if not (0.0 <= resolved_thresh <= 1.0) or not math.isfinite(resolved_thresh):
            raise ValueError(f"binarization_threshold must be a finite float in [0.0, 1.0], got {resolved_thresh}")

    # 3. Output Directory
    # Precedence: CLI > Config > None
    if cli_output_dir:
        resolved_out = cli_output_dir
        sources["output_dir"] = "CLI"
    elif op_data.get("output_dir"):
        resolved_out = op_data.get("output_dir")
        sources["output_dir"] = "CONFIG_FILE"
    else:
        resolved_out = None
        sources["output_dir"] = "NONE"

    # 4. Overwrite
    # Precedence: CLI True > Config True > False
    if cli_overwrite:
        resolved_overwrite = True
        sources["overwrite"] = "CLI"
    elif op_data.get("overwrite") is True:
        resolved_overwrite = True
        sources["overwrite"] = "CONFIG_FILE"
    else:
        resolved_overwrite = False
        sources["overwrite"] = "DEFAULT"

    op_config = OperationalConfig(
        checkpoint_path=resolved_ckpt,
        binarization_threshold=resolved_thresh,
        output_dir=resolved_out,
        overwrite=resolved_overwrite,
    )

    return op_config, sources


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Loads configuration and returns validated AppConfig."""
    data = load_config_file(config_path)
    cfg = get_legacy_2024_config()
    if "operational" in data:
        op, _ = resolve_operational_settings(config_path=config_path)
        cfg.operational = op
    return cfg
