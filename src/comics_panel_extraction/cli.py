"""
Command-line interface for comics_panel_extraction.
Exposes clean commands: infer, postprocess, evaluate, historical-replay.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

from comics_panel_extraction.io.image_io import (
    prepare_output_path,
    load_model_ready_tensor,
    load_binary_mask,
    load_probability_mask,
    save_probability_mask,
    save_binary_mask,
)
from comics_panel_extraction.inference import (
    load_inference_model,
    predict_probability_mask,
    binarize_probability_mask,
)
from comics_panel_extraction.historical_legacy import run_canonical_historical_postprocessing
from comics_panel_extraction.evaluation.dataset import evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comics-panel-extraction",
        description="Comic Panel Extraction CLI (Standalone IO, Config, and Operational Pipeline)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. infer
    p_infer = subparsers.add_parser("infer", help="Run M1 model inference on input image to predict continuous probability mask")
    p_infer.add_argument("input", help="Model-ready input image or tensor file (.npy, .jpg, .png)")
    p_infer.add_argument("--output", "-o", default="output_mask.png", help="Output file path (.npy for probability mask, .png for binary mask)")
    p_infer.add_argument("--checkpoint", "-c", default=None, help="Path to M1 checkpoint (.keras)")
    p_infer.add_argument("--binarize", action="store_true", help="Automatically binarize probability mask with threshold 0.5")
    p_infer.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")

    # 2. postprocess
    p_post = subparsers.add_parser("postprocess", help="Apply canonical historical line-segment postprocessing to binary mask")
    p_post.add_argument("input", help="Binary mask file (.png or .npy)")
    p_post.add_argument("--output", "-o", required=True, help="Output final refined mask file (.png)")
    p_post.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")

    # 3. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate segmentation metrics across dataset")
    p_eval.add_argument("--predictions", "-p", required=True, help="Directory containing prediction masks (.png)")
    p_eval.add_argument("--ground-truth", "-g", required=True, help="Directory containing ground truth masks (.png)")
    p_eval.add_argument("--output-json", "-o", default=None, help="Optional output path to save JSON metrics")
    p_eval.add_argument("--overwrite", action="store_true", help="Overwrite output JSON if it exists")

    # 4. historical-replay
    p_hist = subparsers.add_parser("historical-replay", help="Canonical historical legacy postprocessing replay across raw prediction masks")
    p_hist.add_argument("--input-dir", "-i", required=True, help="Directory containing raw prediction PNG masks")
    p_hist.add_argument("--output-dir", "-o", required=True, help="Directory to save final processed masks")
    p_hist.add_argument("--ground-truth", "-g", default=None, help="Optional GT directory to run evaluation")
    p_hist.add_argument("--overwrite", action="store_true", help="Overwrite output files")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "infer":
            inp = load_model_ready_tensor(args.input)
            model = load_inference_model(args.checkpoint)
            prob = predict_probability_mask(inp, model=model)
            
            if args.binarize or args.output.lower().endswith(".png"):
                bin_m = binarize_probability_mask(prob, threshold=0.5)
                out_p = save_binary_mask(bin_m, args.output, overwrite=args.overwrite)
                print(f"Saved binary prediction mask to {out_p}")
            else:
                out_p = save_probability_mask(prob, args.output, overwrite=args.overwrite)
                print(f"Saved probability mask to {out_p}")
            return 0

        elif args.command == "postprocess":
            import cv2
            mask = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
            final_m, _ = run_canonical_historical_postprocessing(mask)
            out_p = save_binary_mask(final_m, args.output, overwrite=args.overwrite)
            print(f"Saved postprocessed mask to {out_p}")
            return 0

        elif args.command == "evaluate":
            summary = evaluate_dataset(
                gt_dir=args.ground_truth,
                pred_dir=args.predictions,
                corpus_id="cli_evaluation"
            )
            print("Evaluation Results:")
            print(f"  mIoU:      {summary.mean_iou * 100:.4f}%")
            print(f"  mDice:     {summary.mean_dice * 100:.4f}%")
            print(f"  Panel Acc: {summary.panel_accuracy * 100:.4f}%")
            print(f"  Precision: {summary.precision * 100:.4f}%")
            print(f"  Recall:    {summary.recall * 100:.4f}%")
            print(f"  F1:        {summary.f1 * 100:.4f}%")
            print(f"  Page Acc:  {summary.page_accuracy * 100:.4f}%")

            if args.output_json:
                out_p = prepare_output_path(args.output_json, expected_suffix=".json", overwrite=args.overwrite)
                with open(out_p, "w") as f:
                    json.dump(summary.to_dict(), f, indent=2)
                print(f"Saved evaluation JSON to {out_p}")
            return 0

        elif args.command == "historical-replay":
            import cv2
            in_dir = Path(args.input_dir)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            files = sorted([f for f in in_dir.iterdir() if f.suffix.lower() == ".png"])
            print(f"Executing historical-replay on {len(files)} masks from {in_dir}...")

            for f in files:
                raw_m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                fin_m, _ = run_canonical_historical_postprocessing(raw_m)
                cv2.imwrite(str(out_dir / f.name), fin_m)

            print(f"Historical replay complete. Saved {len(files)} final masks to {out_dir}")

            if args.ground_truth:
                gt_p = Path(args.ground_truth)
                res = evaluate_dataset(str(gt_p), str(out_dir))
                print("Evaluation against GT:")
                print(f"  mIoU: {res.mean_iou * 100:.4f}%")
                print(f"  Panel Accuracy: {res.panel_accuracy * 100:.4f}%")
                print(f"  Page Accuracy: {res.page_accuracy * 100:.4f}%")
            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
