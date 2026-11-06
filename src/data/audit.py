"""Validate class-folder images and generate reproducible stratified CSV splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, UnidentifiedImageError

DEFAULT_CLASSES = (
    "ok", "scratch", "crack", "porosity", "pinhole", "inclusion",
    "shrinkage", "surface_roughness", "dent", "cold_shut", "flash", "mixed_defects",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(
    dataset_root: Path | str,
    expected_classes: Sequence[str] = DEFAULT_CLASSES,
) -> tuple[dict, list[dict]]:
    \"\"\"Return a JSON-safe audit report and records for images that can be decoded.\"\"\"
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    classes = tuple(expected_classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("expected_classes must contain unique class names")

    existing_dirs = {path.name for path in root.iterdir() if path.is_dir()}
    missing = sorted(set(classes) - existing_dirs)
    unexpected = sorted(existing_dirs - set(classes))
    class_counts: dict[str, dict[str, int]] = {}
    valid_records: list[dict] = []
    corrupt: list[dict] = []
    hashes: dict[str, list[dict]] = defaultdict(list)
    formats: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()

    for class_name in classes:
        class_dir = root / class_name
        paths = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if class_dir.is_dir() else []
        valid_count = 0
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    image_format = str(image.format or "unknown").upper()
                    image_mode = image.mode
                    width, height = image.size
                record = {
                    "image_path": path.relative_to(root).as_posix(),
                    "label": class_name,
                    "sha256": _file_sha256(path),
                }
                valid_records.append(record)
                hashes[record["sha256"]].append(record)
                formats[image_format] += 1
                modes[image_mode] += 1
                dimensions[f"{width}x{height}"] += 1
                valid_count += 1
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                corrupt.append({
                    "image_path": path.relative_to(root).as_posix(),
                    "label": class_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        class_counts[class_name] = {
            "found": len(paths), "valid": valid_count, "corrupt": len(paths) - valid_count,
        }

    duplicates = [group for group in hashes.values() if len(group) > 1]
    duplicate_conflicts = [group for group in duplicates if len({item["label"] for item in group}) > 1]
    image_count = sum(item["found"] for item in class_counts.values())
    non_image_count = sum(
        1 for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS
    )
    report = {
        "dataset_root": str(root),
        "expected_classes": list(classes),
        "class_counts": class_counts,
        "total_images": image_count,
        "valid_images": len(valid_records),
        "corrupt_images": len(corrupt),
        "missing_classes": missing,
        "unexpected_class_directories": unexpected,
        "non_image_files": non_image_count,
        "formats": dict(sorted(formats.items())),
        "modes": dict(sorted(modes.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "exact_duplicate_groups": len(duplicates),
        "exact_duplicate_files": sum(len(group) for group in duplicates),
        "duplicate_label_conflicts": len(duplicate_conflicts),
        "duplicates": duplicates,
        "corrupt_file_details": corrupt,
    }
    return report, valid_records


def _allocate_counts(total: int, ratios: Sequence[float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (raw[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def create_stratified_splits(
    records: Iterable[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict]]:
    \"\"\"Create deterministic per-class splits and reject exact duplicates to prevent leakage.\"\"\"
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(value <= 0 or value >= 1 for value in ratios):
        raise ValueError("Each split ratio must be greater than 0 and less than 1")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Split ratios must sum to 1.0")

    items = list(records)
    seen_hashes: set[str] = set()
    for item in items:
        digest = item.get("sha256")
        if digest and digest in seen_hashes:
            raise ValueError("Exact duplicate images detected; resolve them before splitting")
        if digest:
            seen_hashes.add(digest)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_class[str(item["label"])].append(item)
    if not by_class:
        raise ValueError("No valid images are available to split")

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_name in sorted(by_class):
        class_items = sorted(by_class[class_name], key=lambda item: item["image_path"])
        if len(class_items) < 3:
            raise ValueError(f"Class '{class_name}' needs at least 3 valid images")
        rng.shuffle(class_items)
        train_count, val_count, test_count = _allocate_counts(len(class_items), ratios)
        splits["train"].extend(class_items[:train_count])
        splits["val"].extend(class_items[train_count:train_count + val_count])
        splits["test"].extend(class_items[train_count + val_count:train_count + val_count + test_count])

    for split_items in splits.values():
        split_items.sort(key=lambda item: (item["label"], item["image_path"]))
    return splits


def write_outputs(
    output_dir: Path | str,
    report: dict,
    splits: dict[str, list[dict]] | None,
    seed: int,
    ratios: Sequence[float],
) -> None:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["split_seed"] = seed if splits is not None else None
    payload["split_ratios"] = dict(zip(("train", "val", "test"), ratios)) if splits else None
    payload["split_counts"] = {name: len(items) for name, items in splits.items()} if splits else None
    (destination / "dataset_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    with (destination / "class_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "found", "valid", "corrupt"])
        for name, counts in report["class_counts"].items():
            writer.writerow([name, counts["found"], counts["valid"], counts["corrupt"]])

    if splits:
        split_dir = destination / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, split_items in splits.items():
            with (split_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "sha256"])
                writer.writeheader()
                writer.writerows(split_items)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit a class-folder image dataset and make reproducible splits."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args(argv)

    report, records = scan_dataset(args.dataset_root)
    has_structure_issues = bool(report["missing_classes"] or report["unexpected_class_directories"])
    has_duplicates = report["exact_duplicate_groups"] > 0
    splits = None
    if not has_structure_issues and not has_duplicates and records:
        splits = create_stratified_splits(
            records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    write_outputs(
        args.output_dir, report, splits, args.seed,
        (args.train_ratio, args.val_ratio, args.test_ratio),
    )
    print(f"Dataset images: {report['total_images']}")
    print(f"Valid images: {report['valid_images']}")
    print(f"Corrupt images: {report['corrupt_images']}")
    print(f"Exact duplicate groups: {report['exact_duplicate_groups']}")
    print(f"Audit report: {Path(args.output_dir) / 'dataset_audit.json'}")
    if splits is None:
        print("Split files not created; resolve missing/unexpected class folders or duplicates.", file=sys.stderr)
        return 2
    print("Split counts:", {name: len(values) for name, values in splits.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
