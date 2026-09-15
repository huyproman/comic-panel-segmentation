import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
import json
import sys
from pathlib import Path
import cv2
import keras
import numpy as np

from comics_panel_extraction.config.models import load_config_from_yaml
from comics_panel_extraction.data.loader import ComicDatasetLoader, build_tf_dataset
from comics_panel_extraction.evaluation.evaluator import PanelInstanceEvaluator, evaluate_instance_dataset
from comics_panel_extraction.inference.predictor import binarize_probability_mask, predict_probability_mask
from comics_panel_extraction.postprocessing.line_analysis import run_line_segment_postprocessing
from comics_panel_extraction.training.trainer import Trainer
from comics_panel_extraction.training.config import TrainingConfig

def build_parser():
    parser = argparse.ArgumentParser(
        prog="comics-panel-extraction",
        description="Comic Panel Extraction using U-Net++ Deep Learning and Line Segment Analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("image", type=str)
    infer_parser.add_argument("-c", "--checkpoint", type=str, required=True)
    infer_parser.add_argument("-o", "--output", type=str, required=True)
    infer_parser.add_argument("--raw-output", type=str, default=None)
    infer_parser.add_argument("--raw-only", action="store_true")
    infer_parser.add_argument("--threshold", type=float, default=0.50)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml")
    train_parser.add_argument("--dataset", type=str, default=None)
    train_parser.add_argument("--val-dataset", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--output-dir", type=str, default=None)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("-p", "--predictions", type=str, required=True)
    eval_parser.add_argument("-g", "--ground-truth", type=str, required=True)
    eval_parser.add_argument("-o", "--output-json", type=str, default="metrics.json")

    pp_parser = subparsers.add_parser("postprocess")
    pp_parser.add_argument("input_mask", type=str)
    pp_parser.add_argument("-o", "--output", type=str, required=True)

    return parser

def main(args=None):
    parser = build_parser()
    if args is not None and len(args) == 0:
        parser.print_help()
        return 1

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "infer":
        img_p = Path(parsed_args.image)
        if not img_p.exists():
            print(f"Error: image not found at {img_p}", file=sys.stderr)
            return 1

        if str(img_p).endswith(".npy"):
            image = np.load(str(img_p))
        else:
            image = cv2.imread(str(img_p))

        if image is None:
            print(f"Error: failed to load image at {img_p}", file=sys.stderr)
            return 1

        ckpt_p = Path(parsed_args.checkpoint)
        if not ckpt_p.exists():
            print(f"Error: checkpoint not found at {ckpt_p}", file=sys.stderr)
            return 1

        try:
            model = keras.models.load_model(str(ckpt_p), compile=False)
        except Exception as e:
            print(f"Error: failed to load model from {ckpt_p}: {e}", file=sys.stderr)
            return 1

        if model.input_shape != (None, 448, 448, 3) or model.output_shape != (None, 448, 448, 1):
            print(f"Error: model has incompatible shapes {model.input_shape} -> {model.output_shape}", file=sys.stderr)
            return 1

        apply_pp = not parsed_args.raw_only
        prob = predict_probability_mask(image, model=model)

        raw_binary_canvas = binarize_probability_mask(prob, threshold=parsed_args.threshold)
        orig_h, orig_w = image.shape[:2]
        raw_mask = cv2.resize(raw_binary_canvas, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        if apply_pp:
            final_pp_canvas, diagnostics = run_line_segment_postprocessing(raw_binary_canvas)
            final_mask = cv2.resize(final_pp_canvas, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            final_mask = raw_mask.copy()

        out_p = Path(parsed_args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if str(out_p).endswith(".npy"):
            np.save(str(out_p), final_mask)
        else:
            cv2.imwrite(str(out_p), final_mask)

        if parsed_args.raw_output:
            raw_out_p = Path(parsed_args.raw_output)
            raw_out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(raw_out_p), raw_mask)

        return 0

    elif parsed_args.command == "postprocess":
        mask_p = Path(parsed_args.input_mask)
        if not mask_p.exists():
            print(f"Error: input mask not found at {mask_p}", file=sys.stderr)
            return 1

        raw_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        final_mask, diagnostics = run_line_segment_postprocessing(raw_mask)
        out_p = Path(parsed_args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), final_mask)
        return 0

    elif parsed_args.command == "evaluate":
        preds_dir = Path(parsed_args.predictions)
        gt_dir = Path(parsed_args.ground_truth)
        pred_files = sorted(list(preds_dir.glob("*.png")) + list(preds_dir.glob("*.jpg")))
        page_results = []
        evaluator = PanelInstanceEvaluator()

        for pf in pred_files:
            gf = gt_dir / pf.name
            if gf.exists():
                p_mask = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
                g_mask = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
                page_results.append(evaluator.evaluate_page(p_mask, g_mask))

        res = evaluate_instance_dataset(page_results)
        required_keys = ["canonical_instance_miou", "panel_accuracy", "page_accuracy", "precision", "recall", "f1"]
        for k in required_keys:
            if k not in res:
                raise KeyError(f"Expected evaluation metric '{k}' missing from evaluator output.")

        with open(parsed_args.output_json, "w") as f:
            json.dump(res, f, indent=2)
        return 0

    elif parsed_args.command == "train":
        app_cfg = None
        if parsed_args.config and Path(parsed_args.config).exists():
            app_cfg = load_config_from_yaml(parsed_args.config)

        dataset_arg = parsed_args.dataset or (app_cfg.training.dataset_dir if app_cfg and hasattr(app_cfg.training, "dataset_dir") else None)
        if not dataset_arg:
            print("Error: --dataset path or config dataset_dir is required for training", file=sys.stderr)
            return 1

        epochs = parsed_args.epochs or (app_cfg.training.epochs if app_cfg else 150)
        batch_size = parsed_args.batch_size or (app_cfg.training.batch_size if app_cfg else 8)
        lr = app_cfg.training.learning_rate if app_cfg else 1e-4
        warmup_epochs = app_cfg.training.warmup_epochs if app_cfg else 5
        weight_decay = app_cfg.training.weight_decay if app_cfg else 1e-4
        output_dir = parsed_args.output_dir or (app_cfg.training.output_dir if app_cfg else "runs/train")

        train_cfg = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            warmup_epochs=warmup_epochs,
            weight_decay=weight_decay,
            dataset_dir=dataset_arg,
            output_dir=output_dir,
        )

        train_loader = ComicDatasetLoader(dataset_dir=train_cfg.dataset_dir, target_size=(448, 448), shuffle=True)
        if len(train_loader) == 0:
            print(f"Error: no valid image/mask pairs found under {train_cfg.dataset_dir}", file=sys.stderr)
            return 1

        steps_per_epoch = max(1, len(train_loader) // train_cfg.batch_size)
        trainer = Trainer(config=train_cfg, steps_per_epoch=steps_per_epoch)

        train_ds = build_tf_dataset(
            dataset_dir=train_cfg.dataset_dir,
            batch_size=train_cfg.batch_size,
            target_size=(448, 448),
            shuffle=True,
            repeat=True,
        )

        val_ds = None
        val_steps = 1
        val_dataset_arg = parsed_args.val_dataset
        if val_dataset_arg and Path(val_dataset_arg).exists():
            val_loader = ComicDatasetLoader(dataset_dir=val_dataset_arg, target_size=(448, 448), shuffle=False)
            if len(val_loader) > 0:
                val_steps = max(1, len(val_loader) // train_cfg.batch_size)
                val_ds = build_tf_dataset(
                    dataset_dir=val_dataset_arg,
                    batch_size=train_cfg.batch_size,
                    target_size=(448, 448),
                    shuffle=False,
                    repeat=True,
                )

        res = trainer.train_dataset(
            train_ds=train_ds,
            val_ds=val_ds,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
        )
        print(f"Training completed. Best loss: {res.get('best_loss', 0.0):.4f}")
        return 0

if __name__ == "__main__":
    main()
