"""
Utility helpers for training and evaluation workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_model(model: torch.nn.Module, savepath: str | Path, model_name: str) -> Path:
    """
    Persist a model checkpoint on disk and return the resulting path.
    """
    output_dir = ensure_dir(savepath)
    model_path = output_dir / model_name
    torch.save(model.state_dict(), model_path)
    return model_path


def save_losses(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    f1_scores: Sequence[float],
    savepath: str | Path,
) -> Path:
    """
    Store loss curves and validation F1 scores in CSV form for later analysis.
    """
    output_dir = ensure_dir(savepath)
    csv_path = output_dir / "training_metrics.csv"

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,val_loss,val_f1\n")
        for idx, (train_loss, val_loss, f1) in enumerate(
            zip(train_losses, val_losses, f1_scores), start=1
        ):
            handle.write(f"{idx},{train_loss},{val_loss},{f1}\n")

    return csv_path


def compute_IoU(predictions: Iterable[int], targets: Iterable[int]) -> float:
    """
    Compute the Intersection-over-Union metric for binary predictions.
    """
    preds = np.asarray(list(predictions)).astype(bool)
    labels = np.asarray(list(targets)).astype(bool)

    intersection = np.logical_and(preds, labels).sum()
    union = np.logical_or(preds, labels).sum()
    return float(intersection / union) if union else 0.0


def compute_metrics(
    predictions: Iterable[int],
    targets: Iterable[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute accuracy, precision, recall, and balanced accuracy for binary outputs.
    """
    preds = np.asarray(list(predictions))
    labels = np.asarray(list(targets))

    if preds.dtype != np.bool_:
        preds = preds > threshold
    if labels.dtype != np.bool_:
        labels = labels > threshold

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }
