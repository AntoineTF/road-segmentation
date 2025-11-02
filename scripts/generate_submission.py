"""
Convert predicted masks into the AIcrowd submission format.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import numpy as np

FOREGROUND_THRESHOLD = 0.25  # Mean pixel intensity threshold per patch
PATCH_SIZE = 16


def patch_to_label(patch: np.ndarray, threshold: float) -> int:
    """Assign a binary label to a patch given a foreground threshold."""
    return int(patch.mean() > threshold)


def mask_to_submission_rows(image_path: Path, threshold: float) -> Iterable[str]:
    """Yield CSV rows for a single predicted mask."""
    match = re.search(r"\d+", image_path.stem)
    if not match:
        raise ValueError(f"Could not extract image index from filename '{image_path.name}'.")
    img_number = int(match.group(0))

    mask = mpimg.imread(image_path)

    for j in range(0, mask.shape[1], PATCH_SIZE):
        for i in range(0, mask.shape[0], PATCH_SIZE):
            patch = mask[i : i + PATCH_SIZE, j : j + PATCH_SIZE]
            label = patch_to_label(patch, threshold)
            yield f"{img_number:03d}_{j}_{i},{label}"


def masks_to_submission(output_file: Path, mask_paths: Iterable[Path], threshold: float) -> None:
    """Create the submission CSV file from a list of mask paths."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        handle.write("id,prediction\n")
        for mask_path in mask_paths:
            for row in mask_to_submission_rows(mask_path, threshold):
                handle.write(f"{row}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create AIcrowd submission from predicted masks.")
    parser.add_argument("--input_masks", type=Path, default=Path("predictions/test_masks"))
    parser.add_argument("--output_file", type=Path, default=Path("submission.csv"))
    parser.add_argument("--threshold", type=float, default=FOREGROUND_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_dir = args.input_masks
    if not mask_dir.exists():
        raise FileNotFoundError(f"No predicted masks found at {mask_dir}")

    mask_paths = sorted(mask_dir.glob("*.png"), key=lambda path: int(re.search(r"\d+", path.stem).group(0)))
    masks_to_submission(args.output_file, mask_paths, args.threshold)
    print(f"Submission written to {args.output_file}")


if __name__ == "__main__":
    main()
