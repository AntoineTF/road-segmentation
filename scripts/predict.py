"""
Generate segmentation masks on the test set using a trained checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm

from src.data import TestDataset
from src.models import (
    DLinkNet34,
    LinkNet34,
    NL_LinkNet_EGaussian,
    SegNet,
    UNet,
)

MODEL_REGISTRY = {
    "NL_LinkNet_EGaussian": NL_LinkNet_EGaussian,
    "LinkNet34": LinkNet34,
    "DLinkNet34": DLinkNet34,
    "UNet": UNet,
    "SegNet": SegNet,
}


def load_model(model_name: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Instantiate and load weights for the requested architecture."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    model = MODEL_REGISTRY[model_name](num_channels=3, num_classes=1).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    return model.eval()


def predict_masks(
    model: torch.nn.Module,
    dataset_root: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float,
) -> None:
    """Run inference on the provided dataset and persist predicted masks."""
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = TestDataset(str(dataset_root), transform=transform)

    output_dir.mkdir(parents=True, exist_ok=True)

    for mask_file in output_dir.glob("*.png"):
        mask_file.unlink()

    for idx in tqdm(range(len(test_dataset)), desc="Predicting"):
        image = test_dataset[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image)
            probabilities = torch.sigmoid(logits)
            mask = (probabilities > threshold).float().squeeze().cpu().numpy()

        output_path = output_dir / f"test_{idx + 1}_mask.png"
        plt.imsave(output_path, mask, cmap="gray")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for road-segmentation models.")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test_set", type=Path, default=Path("dataset/test_set_images"))
    parser.add_argument("--output_masks", type=Path, default=Path("predictions/test_masks"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_name, args.checkpoint, device)
    predict_masks(model, args.test_set, args.output_masks, device, args.threshold)

    print(f"Predicted masks saved to {args.output_masks}")


if __name__ == "__main__":
    main()
