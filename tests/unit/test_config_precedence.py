"""
Tests for configuration parsing, precedence order, and validation.
"""

from pathlib import Path
import os
import pytest
import yaml

from comics_panel_extraction.config.models import (
    load_config_file,
    resolve_operational_settings,
    get_legacy_2024_config,
)


def test_config_file_valid_operational(tmp_path):
    cfg_data = {
        "profile": "legacy_2024",
        "operational": {
            "checkpoint_path": "/path/to/custom.keras",
            "binarization_threshold": 0.45,
            "output_dir": "/path/to/out",
            "overwrite": True
        }
    }
    cfg_p = tmp_path / "test_config.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    op, sources = resolve_operational_settings(config_path=cfg_p)
    assert op.checkpoint_path == "/path/to/custom.keras"
    assert op.binarization_threshold == 0.45
    assert op.output_dir == "/path/to/out"
    assert op.overwrite is True
    assert sources["binarization_threshold"] == "CONFIG_FILE"


def test_config_precedence_env_over_file(tmp_path, monkeypatch):
    cfg_data = {
        "profile": "legacy_2024",
        "operational": {
            "checkpoint_path": "/file/path/ckpt.keras"
        }
    }
    cfg_p = tmp_path / "test_config.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    monkeypatch.setenv("COMICS_CHECKPOINT_PATH", "/env/path/ckpt.keras")
    op, sources = resolve_operational_settings(config_path=cfg_p)
    assert op.checkpoint_path == "/env/path/ckpt.keras"
    assert sources["checkpoint_path"] == "ENV"


def test_config_precedence_cli_over_env(tmp_path, monkeypatch):
    cfg_data = {
        "profile": "legacy_2024",
        "operational": {
            "checkpoint_path": "/file/path/ckpt.keras"
        }
    }
    cfg_p = tmp_path / "test_config.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    monkeypatch.setenv("COMICS_CHECKPOINT_PATH", "/env/path/ckpt.keras")
    op, sources = resolve_operational_settings(cli_checkpoint="/cli/path/ckpt.keras", config_path=cfg_p)
    assert op.checkpoint_path == "/cli/path/ckpt.keras"
    assert sources["checkpoint_path"] == "CLI"


def test_config_reject_unknown_top_level_key(tmp_path):
    cfg_data = {
        "profile": "legacy_2024",
        "unknown_section": {"foo": "bar"}
    }
    cfg_p = tmp_path / "bad_config.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config_file(cfg_p)


def test_config_reject_nested_unknown_algorithm_key(tmp_path):
    cfg_data = {
        "profile": "legacy_2024",
        "line_analysis": {"typo_parameter": 123}
    }
    cfg_p = tmp_path / "bad_nested.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    with pytest.raises(ValueError, match="Unknown key 'typo_parameter' in algorithm section 'line_analysis'"):
        load_config_file(cfg_p)


def test_config_reject_nested_override_algorithm_key(tmp_path):
    cfg_data = {
        "profile": "legacy_2024",
        "line_analysis": {"hough_threshold": 999}
    }
    cfg_p = tmp_path / "algo_override.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    with pytest.raises(ValueError, match="Cannot override frozen legacy_2024 parameter"):
        load_config_file(cfg_p)


def test_config_reject_unknown_operational_key(tmp_path):
    cfg_data = {
        "profile": "legacy_2024",
        "operational": {
            "unsupported_key": 123
        }
    }
    cfg_p = tmp_path / "bad_op.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    with pytest.raises(ValueError, match="Unknown operational configuration key"):
        load_config_file(cfg_p)


def test_config_reject_unsupported_profile(tmp_path):
    cfg_data = {
        "profile": "modern_2026"
    }
    cfg_p = tmp_path / "bad_prof.yaml"
    with open(cfg_p, "w") as f:
        yaml.dump(cfg_data, f)

    with pytest.raises(ValueError, match="Unsupported profile"):
        load_config_file(cfg_p)
