"""
Command-line entry point for training segmentation models.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import AugmentedDataset
from src.models import (
    DLinkNet34,
    LinkNet34,
    Losses,
    NL_LinkNet_EGaussian,
    SegNet,
    UNet,
)
from src.utils import compute_IoU, compute_metrics, save_losses, save_model

THRESHOLD = 0.25
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_PLOTS_DIR = Path("plots")

MODEL_REGISTRY = {
    "NL_LinkNet_EGaussian": NL_LinkNet_EGaussian,
    "LinkNet34": LinkNet34,
    "DLinkNet34": DLinkNet34,
    "UNet": UNet,
    "SegNet": SegNet,
}


def get_device() -> torch.device:
    """Return the best available compute device."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_dir: Path,
    model_name: str,
) -> Path:
    """Generate and store a training/validation loss plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{model_name}_loss_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss · {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def train_model(
    model_name: str,
    train_images: Path,
    train_masks: Path,
    val_images: Path,
    val_masks: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    loss_name: str,
    models_dir: Path,
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Path]:
    """Train the requested model and return training artefacts."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    device = get_device()
    print(f"Using device: {device}")

    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = (
        f"{model_name}_{loss_name}_lr{learning_rate}_ep{epochs}_bs{batch_size}.pt"
    )
    checkpoint_path = models_dir / checkpoint_name

    target_size = (512, 512) if model_name == "SegNet" else (416, 416)
    train_dataset = AugmentedDataset(
        images_dir=str(train_images),
        groundtruth_dir=str(train_masks),
        target_size=target_size,
        threshold=THRESHOLD,
    )
    val_dataset = AugmentedDataset(
        images_dir=str(val_images),
        groundtruth_dir=str(val_masks),
        target_size=target_size,
        threshold=THRESHOLD,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ModelClass = MODEL_REGISTRY[model_name]
    sigmoid_bool = loss_name not in ["bce"]
    model = ModelClass(num_classes=1, sigmoid_bool=sigmoid_bool).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    loss_fn = Losses(beta=0.8)

    history = {"train_losses": [], "val_losses": [], "val_f1": []}
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        print("-" * 20 + f" Epoch {epoch + 1}/{epochs} " + "-" * 20)
        since = time.time()

        model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(labels, outputs, loss_name)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        train_epoch_loss = running_loss / max(total_samples, 1)
        history["train_losses"].append(train_epoch_loss)
        print(f"Training Loss: {train_epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_preds, val_targets = [], []

        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(labels, outputs, loss_name)
                if not sigmoid_bool:
                    outputs = torch.sigmoid(outputs)

            val_loss += loss.item() * inputs.size(0)
            val_samples += inputs.size(0)
            val_preds.append((outputs > THRESHOLD).view(-1).cpu())
            val_targets.append((labels > THRESHOLD).view(-1).cpu())

        val_epoch_loss = val_loss / max(val_samples, 1)
        history["val_losses"].append(val_epoch_loss)

        preds_flat = torch.cat(val_preds).numpy()
        targets_flat = torch.cat(val_targets).numpy()
        val_f1 = f1_score(targets_flat, preds_flat)
        history["val_f1"].append(val_f1)

        metrics = compute_metrics(preds_flat, targets_flat, threshold=THRESHOLD)
        print(f"Validation Loss: {val_epoch_loss:.4f}")
        print(f"F1 score: {val_f1:.4f}")
        print(f"IoU score: {compute_IoU(preds_flat, targets_flat):.4f}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.capitalize()}: {metric_value:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            checkpoint_path = save_model(model, models_dir, checkpoint_name)
            print(f"New best model saved to {checkpoint_path} (F1 = {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        if patience_counter > patience:
            print("Early stopping triggered.")
            break

        scheduler.step()
        elapsed = time.time() - since
        print(f"Epoch duration: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    metrics_dir = models_dir / checkpoint_name.replace(".pt", "")
    save_losses(
        history["train_losses"],
        history["val_losses"],
        history["val_f1"],
        metrics_dir,
    )

    if not checkpoint_path.exists():
        checkpoint_path = save_model(model, models_dir, checkpoint_name)

    return model, history, checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss_name", type=str, default="squared_dice")
    parser.add_argument("--train_images", type=Path, default=Path("dataset/training/images"))
    parser.add_argument("--train_masks", type=Path, default=Path("dataset/training/groundtruth"))
    parser.add_argument("--val_images", type=Path, default=Path("dataset/validation/images"))
    parser.add_argument("--val_masks", type=Path, default=Path("dataset/validation/groundtruth"))
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--plots_dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, history, checkpoint_path = train_model(
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

    plot_path = plot_training_curves(
        history["train_losses"],
        history["val_losses"],
        args.plots_dir,
        args.model_name,
    )

    print(f"Training complete. Best checkpoint: {checkpoint_path}")
    print(f"Loss curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
