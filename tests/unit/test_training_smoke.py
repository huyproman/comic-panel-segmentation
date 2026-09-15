from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml

from comics_panel_extraction.config.models import load_config_from_yaml
from comics_panel_extraction.data.loader import ComicDatasetLoader, build_tf_dataset
from comics_panel_extraction.training.config import TrainingConfig
from comics_panel_extraction.training.trainer import Trainer
from comics_panel_extraction.cli import main

@pytest.mark.unit
def test_no_cli_duplicate_dataset_helpers():
    import comics_panel_extraction.cli as cli_mod
    assert not hasattr(cli_mod, "load_dataset_pairs")
    assert not hasattr(cli_mod, "build_tf_dataset_from_pairs")
    assert hasattr(cli_mod, "ComicDatasetLoader")
    assert hasattr(cli_mod, "build_tf_dataset")

@pytest.mark.unit
def test_training_uses_public_data_loader_and_validation(tmp_path):
    train_dir = tmp_path / "train_ds"
    val_dir = tmp_path / "val_ds"
    (train_dir / "images").mkdir(parents=True)
    (train_dir / "masks").mkdir(parents=True)
    (val_dir / "images").mkdir(parents=True)
    (val_dir / "masks").mkdir(parents=True)

    for i in range(2):
        im = np.zeros((448, 448, 3), dtype=np.uint8)
        im[50:150, 50:150] = 200
        mk = np.zeros((448, 448), dtype=np.uint8)
        mk[50:150, 50:150] = 255
        cv2.imwrite(str(train_dir / "images" / f"p_{i}.png"), im)
        cv2.imwrite(str(train_dir / "masks" / f"p_{i}.png"), mk)
        cv2.imwrite(str(val_dir / "images" / f"p_{i}.png"), im)
        cv2.imwrite(str(val_dir / "masks" / f"p_{i}.png"), mk)

    train_loader = ComicDatasetLoader(dataset_dir=train_dir, target_size=(448, 448), shuffle=True)
    val_loader = ComicDatasetLoader(dataset_dir=val_dir, target_size=(448, 448), shuffle=False)
    assert len(train_loader) == 2
    assert len(val_loader) == 2

    x_sample, y_sample = train_loader.load_sample(0)
    assert x_sample.shape == (448, 448, 3)
    assert y_sample.shape == (448, 448, 1)

    custom_cfg_data = {
        "profile": "default",
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.0002,
            "warmup_epochs": 1,
            "weight_decay": 0.0001,
            "output_dir": str(tmp_path / "runs"),
            "dataset_dir": str(train_dir),
        }
    }
    cfg_file = tmp_path / "custom_train.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(custom_cfg_data, f)

    app_cfg = load_config_from_yaml(str(cfg_file))
    assert app_cfg.training.learning_rate == 0.0002
    assert app_cfg.training.dataset_dir == str(train_dir)

    train_ds = build_tf_dataset(
        dataset_dir=train_dir,
        batch_size=app_cfg.training.batch_size,
        target_size=(448, 448),
        shuffle=True,
        repeat=True,
    )
    val_ds = build_tf_dataset(
        dataset_dir=val_dir,
        batch_size=app_cfg.training.batch_size,
        target_size=(448, 448),
        shuffle=False,
        repeat=True,
    )

    t_cfg = TrainingConfig(
        epochs=app_cfg.training.epochs,
        batch_size=app_cfg.training.batch_size,
        learning_rate=app_cfg.training.learning_rate,
        warmup_epochs=app_cfg.training.warmup_epochs,
        weight_decay=app_cfg.training.weight_decay,
        dataset_dir=app_cfg.training.dataset_dir,
        output_dir=app_cfg.training.output_dir,
    )
    trainer = Trainer(config=t_cfg, steps_per_epoch=1)

    initial_weights = [w.numpy().copy() for w in trainer.model.trainable_variables[:2]]
    res = trainer.train_dataset(train_ds=train_ds, val_ds=val_ds, steps_per_epoch=1, val_steps=1)

    assert "train_loss" in res["history"]
    assert "val_loss" in res["history"]
    assert "lr" in res["history"]
    assert len(res["history"]["train_loss"]) == 2
    assert len(res["history"]["val_loss"]) == 2

    updated_weights = [w.numpy().copy() for w in trainer.model.trainable_variables[:2]]
    for init_w, up_w in zip(initial_weights, updated_weights):
        assert not np.array_equal(init_w, up_w)

    assert (tmp_path / "runs" / "best_model.keras").exists()
    assert (tmp_path / "runs" / "latest_model.keras").exists()

@pytest.mark.unit
def test_cli_train_with_config_and_real_loader(tmp_path):
    data_dir = tmp_path / "cli_ds"
    (data_dir / "images").mkdir(parents=True)
    (data_dir / "masks").mkdir(parents=True)

    for i in range(2):
        im = np.zeros((448, 448, 3), dtype=np.uint8)
        im[50:150, 50:150] = 200
        mk = np.zeros((448, 448), dtype=np.uint8)
        mk[50:150, 50:150] = 255
        cv2.imwrite(str(data_dir / "images" / f"p_{i}.png"), im)
        cv2.imwrite(str(data_dir / "masks" / f"p_{i}.png"), mk)

    out_runs = tmp_path / "cli_runs"
    custom_yaml = tmp_path / "train_conf.yaml"
    with open(custom_yaml, "w") as f:
        yaml.dump({
            "profile": "default",
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.0001,
                "warmup_epochs": 1,
                "output_dir": str(out_runs),
            }
        }, f)

    ret = main([
        "train",
        "--config", str(custom_yaml),
        "--dataset", str(data_dir),
    ])
    assert ret == 0
    assert (out_runs / "latest_model.keras").exists()
