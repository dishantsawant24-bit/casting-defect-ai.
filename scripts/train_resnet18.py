"""Train a reproducible ResNet18 classifier from local split manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CLASS_NAMES = [
    "ok", "scratch", "crack", "porosity", "pinhole", "inclusion",
    "shrinkage", "surface_roughness", "dent", "cold_shut", "flash", "mixed_defects",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CastingCsvDataset(Dataset):
    def __init__(self, csv_path: Path, dataset_root: Path, transform: transforms.Compose):
        with csv_path.open("r", newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        self.dataset_root = dataset_root.expanduser().resolve()
        self.transform = transform
        if not self.rows:
            raise ValueError(f"Split manifest is empty: {csv_path}")
        for row in self.rows:
            if row.get("label") not in CLASS_TO_INDEX:
                raise ValueError(f"Unexpected label in {csv_path}: {row.get('label')}")
            image_path = (self.dataset_root / row["image_path"]).resolve()
            if self.dataset_root not in image_path.parents:
                raise ValueError(f"Image path escapes dataset root: {row['image_path']}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image from manifest was not found: {image_path}")
            row["_resolved_path"] = str(image_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["_resolved_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), CLASS_TO_INDEX[row["label"]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))
    if total == 0:
        raise RuntimeError("DataLoader returned no images.")
    return loss_sum / total, correct / total


def train(
    dataset_root: Path,
    splits_dir: Path,
    output_checkpoint: Path,
    history_path: Path,
    epochs: int = 25,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
    pretrained: bool = True,
) -> dict[str, Any]:
    seed_everything(seed)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    train_dataset = CastingCsvDataset(train_csv, dataset_root, train_transform)
    val_dataset = CastingCsvDataset(val_csv, dataset_root, eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_accuracy)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        print(
            f"Epoch {epoch:02d}/{epochs} | train {train_accuracy:.3f} "
            f"({train_loss:.4f}) | val {val_accuracy:.3f} ({val_loss:.4f})"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": len(CLASS_NAMES),
                "class_names": CLASS_NAMES,
                "best_val_accuracy": best_val_accuracy,
                "epoch": best_epoch,
                "training_config": {
                    "seed": seed,
                    "epochs_requested": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "pretrained_imagenet": pretrained,
                    "image_size": 224,
                    "mean": MEAN,
                    "std": STD,
                    "train_manifest_sha256": manifest_sha256(train_csv),
                    "val_manifest_sha256": manifest_sha256(val_csv),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            }, output_checkpoint)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
            break

    result = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "epochs_completed": len(history),
        "elapsed_seconds": round(time.time() - started, 2),
        "checkpoint": str(output_checkpoint),
        "train_manifest_sha256": manifest_sha256(train_csv),
        "val_manifest_sha256": manifest_sha256(val_csv),
        "history": history,
    }
    history_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_best.pth"))
    parser.add_argument("--history", type=Path, default=Path("outputs/reports/training_history.json"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    result = train(
        args.dataset_root, args.splits_dir, args.checkpoint, args.history,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        patience=args.patience, seed=args.seed, pretrained=not args.no_pretrained,
    )
    print(
        f"Best validation accuracy: {result['best_validation_accuracy']:.2%} "
        f"at epoch {result['best_epoch']}"
    )
    print(f"Checkpoint saved: {args.checkpoint}")


if __name__ == "__main__":
    main()
