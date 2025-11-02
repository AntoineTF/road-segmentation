"""
Run the full workflow: training, prediction, and submission generation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from scripts.generate_submission import FOREGROUND_THRESHOLD, masks_to_submission
from scripts.predict import load_model, predict_masks
from scripts.train import (
    DEFAULT_PLOTS_DIR,
    DEFAULT_MODELS_DIR,
    MODEL_REGISTRY,
    get_device,
    plot_training_curves,
    train_model,
)


def collect_mask_paths(mask_dir: Path) -> List[Path]:
    """Return mask paths sorted by their numeric identifier."""
    return sorted(
        mask_dir.glob("*.png"),
        key=lambda path: int(re.search(r"\d+", path.stem).group(0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, predict, and prepare a submission.")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint to skip training.")
    parser.add_argument("--train_images", type=Path, default=Path("dataset/training/images"))
    parser.add_argument("--train_masks", type=Path, default=Path("dataset/training/groundtruth"))
    parser.add_argument("--val_images", type=Path, default=Path("dataset/validation/images"))
    parser.add_argument("--val_masks", type=Path, default=Path("dataset/validation/groundtruth"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss_name", type=str, default="squared_dice")
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--plots_dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--test_set", type=Path, default=Path("dataset/test_set_images"))
    parser.add_argument("--output_masks", type=Path, default=Path("predictions/test_masks"))
    parser.add_argument("--submission_file", type=Path, default=Path("submission.csv"))
    parser.add_argument("--prediction_threshold", type=float, default=0.5)
    parser.add_argument("--submission_threshold", type=float, default=FOREGROUND_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Skipping training and using checkpoint at {checkpoint_path}")
    else:
        _, history, checkpoint_path = train_model(
            model_name=args.model_name,
            train_images=args.train_images,
            train_masks=args.train_masks,
            val_images=args.val_images,
            val_masks=args.val_masks,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            loss_name=args.loss_name,
            models_dir=args.models_dir,
        )
        plot_training_curves(
            history["train_losses"],
            history["val_losses"],
            args.plots_dir,
            args.model_name,
        )
        print(f"Training completed. Checkpoint saved at {checkpoint_path}")

    model = load_model(args.model_name, checkpoint_path, device)
    predict_masks(model, args.test_set, args.output_masks, device, args.prediction_threshold)
    print(f"Predicted masks stored in {args.output_masks}")

    mask_paths = collect_mask_paths(args.output_masks)
    masks_to_submission(args.submission_file, mask_paths, args.submission_threshold)
    print(f"Submission generated at {args.submission_file}")


if __name__ == "__main__":
    main()
