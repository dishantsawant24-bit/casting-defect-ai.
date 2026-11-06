"""Evaluate a ResNet18 checkpoint against an explicit CSV split manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

CLASS_NAMES = [
    "ok", "scratch", "crack", "porosity", "pinhole", "inclusion",
    "shrinkage", "surface_roughness", "dent", "cold_shut", "flash", "mixed_defects",
]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvaluationDataset(Dataset):
    def __init__(self, manifest: Path, dataset_root: Path):
        with manifest.open("r", newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows:
            raise ValueError(f"Evaluation manifest is empty: {manifest}")
        self.dataset_root = dataset_root.expanduser().resolve()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        for row in self.rows:
            if row.get("label") not in CLASS_TO_INDEX:
                raise ValueError(f"Unexpected label: {row.get('label')}")
            path = (self.dataset_root / row["image_path"]).resolve()
            if self.dataset_root not in path.parents:
                raise ValueError(f"Image path escapes dataset root: {row['image_path']}")
            if not path.is_file():
                raise FileNotFoundError(path)
            row["_resolved_path"] = str(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(row["_resolved_path"]) as source:
            image = source.convert("RGB")
        return self.transform(image), CLASS_TO_INDEX[row["label"]]


def evaluate(
    checkpoint_path: Path,
    split_csv: Path,
    dataset_root: Path,
    output_dir: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint must contain a model_state_dict dictionary.")
    if list(checkpoint.get("class_names", [])) != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match the dataset class order.")
    if int(checkpoint.get("num_classes", -1)) != len(CLASS_NAMES):
        raise ValueError("Checkpoint class count is not 12.")

    dataset = EvaluationDataset(split_csv, dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    true_indices: list[int] = []
    predicted_indices: list[int] = []
    confidences: list[float] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = probabilities.max(dim=1)
            true_indices.extend(int(value) for value in labels.tolist())
            predicted_indices.extend(int(value) for value in prediction.cpu().tolist())
            confidences.extend(float(value) for value in confidence.cpu().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        true_indices,
        predicted_indices,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_indices, predicted_indices, labels=list(range(len(CLASS_NAMES)))
    )
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "true_label", "predicted_label", "confidence"])
        for row, true_index, predicted_index, confidence in zip(
            dataset.rows, true_indices, predicted_indices, confidences
        ):
            writer.writerow([
                row["image_path"], CLASS_NAMES[true_index],
                CLASS_NAMES[predicted_index], f"{confidence:.8f}",
            ])
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *CLASS_NAMES])
        for label, values in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([label, *values])

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "ResNet18",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_training_metadata": {
            "epoch": checkpoint.get("epoch"),
            "best_val_accuracy": checkpoint.get("best_val_accuracy"),
            "training_config": checkpoint.get("training_config"),
        },
        "split_manifest": str(split_csv.resolve()),
        "split_manifest_sha256": sha256_file(split_csv),
        "dataset_root": str(dataset_root.resolve()),
        "device": str(device),
        "sample_count": len(dataset),
        "accuracy": float(accuracy_score(true_indices, predicted_indices)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
        "predictions_csv": str(predictions_path),
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint, args.split_csv, args.dataset_root, args.output_dir,
        batch_size=args.batch_size,
    )
    print(f"Evaluated {result['sample_count']} images")
    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"Report: {(args.output_dir / 'evaluation_report.json').resolve()}")


if __name__ == "__main__":
    main()
