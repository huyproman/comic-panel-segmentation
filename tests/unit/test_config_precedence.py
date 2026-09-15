from pathlib import Path
import pytest
import yaml
from comics_panel_extraction.config.models import (
    load_config_from_yaml,
    get_default_config,
    validate_config_immutability,
)

@pytest.mark.unit
def test_config_file_valid(tmp_path):
    cfg_data = {
        "profile": "default",
        "component_filtering": {
            "min_component_size": 1500
        }
    }
    cfg_p = tmp_path / "test_config.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)
    cfg = load_config_from_yaml(str(cfg_p))
    assert cfg.component_filtering.min_component_size == 1500

@pytest.mark.unit
def test_config_reject_unsupported_profile(tmp_path):
    cfg_data = {
        "profile": "unsupported_profile"
    }
    cfg_p = tmp_path / "bad_prof.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)
    with pytest.raises(ValueError, match="Unsupported profile"):
        load_config_from_yaml(str(cfg_p))

@pytest.mark.unit
def test_config_immutability_validation():
    cfg = get_default_config()
    with pytest.raises(ValueError, match="Cannot override frozen default parameter"):
        validate_config_immutability(cfg, overrides={"some_param": 123})
