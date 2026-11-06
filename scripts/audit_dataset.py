"""Audit a class-folder casting image dataset without modifying its files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.fftpack import dct
from skimage.metrics import structural_similarity

CLASS_NAMES = [
    "ok", "scratch", "crack", "porosity", "pinhole", "inclusion",
    "shrinkage", "surface_roughness", "dent", "cold_shut", "flash", "mixed_defects",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> int:
    """Create a compact DCT perceptual hash from a 32x32 grayscale thumbnail."""
    resized = image.convert("L").resize(
        (hash_size * 4, hash_size * 4), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(resized, dtype=np.float32)
    coefficients = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low_frequency = coefficients[:hash_size, :hash_size]
    median = float(np.median(low_frequency))
    value = 0
    for bit in (low_frequency > median).flat:
        value = (value << 1) | int(bit)
    return value


def connected_components(pairs: list[dict[str, Any]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    for pair in pairs:
        first, second = pair["image_path_1"], pair["image_path_2"]
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    groups: dict[str, list[str]] = defaultdict(list)
    for image_path in sorted(parent):
        groups[find(image_path)].append(image_path)
    ordered = sorted((sorted(items) for items in groups.values()), key=lambda items: items[0])
    return {
        image_path: f"similar_group_{index:02d}"
        for index, group in enumerate(ordered, start=1)
        for image_path in group
    }


def audit_dataset(
    dataset_root: Path,
    output_dir: Path,
    phash_max_distance: int = 5,
    ssim_min: float = 0.95,
    ssim_size: int = 256,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {dataset_root}")

    actual_dirs = {path.name for path in dataset_root.iterdir() if path.is_dir()}
    expected_dirs = set(CLASS_NAMES)
    manifest: list[dict[str, Any]] = []
    cached_hashes: list[int] = []
    cached_ssim_images: list[np.ndarray] = []
    class_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    exact_hashes: dict[str, list[str]] = defaultdict(list)
    corrupt_files: list[dict[str, str]] = []
    non_image_files: list[str] = []
    exif_count = 0

    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(dataset_root).as_posix()
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                non_image_files.append(rel_path)
                continue
            digest = sha256_file(path)
            try:
                with Image.open(path) as source:
                    source.load()
                    image = source.convert("RGB")
                    image_format = source.format or path.suffix.lstrip(".").upper()
                    image_mode = source.mode
                    width, height = source.size
                    if source.getexif():
                        exif_count += 1
                record = {
                    "image_path": rel_path,
                    "label": class_name,
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "mode": image_mode,
                }
                manifest.append(record)
                cached_hashes.append(perceptual_hash(image))
                resized_gray = image.convert("L").resize(
                    (ssim_size, ssim_size), Image.Resampling.LANCZOS
                )
                cached_ssim_images.append(np.asarray(resized_gray, dtype=np.uint8))
                class_counts[class_name] += 1
                format_counts[image_format] += 1
                mode_counts[image_mode] += 1
                dimension_counts[f"{width}x{height}"] += 1
                exact_hashes[digest].append(rel_path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupt_files.append({"image_path": rel_path, "error": str(exc)})

    manifest.sort(key=lambda item: item["image_path"])
    with (output_dir / "image_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["image_path", "label", "sha256", "width", "height", "format", "mode"],
        )
        writer.writeheader()
        writer.writerows(manifest)

    pairs: list[dict[str, Any]] = []
    for first_index in range(len(manifest)):
        first_record = manifest[first_index]
        for second_index in range(first_index + 1, len(manifest)):
            distance = (
                cached_hashes[first_index] ^ cached_hashes[second_index]
            ).bit_count()
            if distance > phash_max_distance:
                continue
            similarity = float(
                structural_similarity(
                    cached_ssim_images[first_index],
                    cached_ssim_images[second_index],
                    data_range=255,
                )
            )
            if similarity < ssim_min:
                continue
            second_record = manifest[second_index]
            pairs.append(
                {
                    "image_path_1": first_record["image_path"],
                    "label_1": first_record["label"],
                    "image_path_2": second_record["image_path"],
                    "label_2": second_record["label"],
                    "phash_distance": distance,
                    "ssim": round(similarity, 6),
                    "cross_class": first_record["label"] != second_record["label"],
                }
            )

    with (output_dir / "perceptual_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "image_path_1", "label_1", "image_path_2", "label_2",
            "phash_distance", "ssim", "cross_class",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)

    duplicate_groups = [paths for paths in exact_hashes.values() if len(paths) > 1]
    candidate_group_map = connected_components(pairs)
    component_sizes = Counter(candidate_group_map.values())
    candidate_images_by_class = Counter(
        image_path.split("/", 1)[0] for image_path in candidate_group_map
    )
    stats_path = dataset_root / "dataset_stats.json"
    provided_stats: dict[str, Any] | None = None
    if stats_path.is_file():
        try:
            provided_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provided_stats = {"read_error": "dataset_stats.json could not be parsed"}

    audit: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "expected_classes": CLASS_NAMES,
        "class_counts": dict(sorted(class_counts.items())),
        "total_images": len(manifest),
        "valid_images": len(manifest),
        "corrupt_images": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "missing_classes": sorted(expected_dirs - actual_dirs),
        "unexpected_class_directories": sorted(actual_dirs - expected_dirs),
        "non_image_files": sorted(non_image_files),
        "formats": dict(format_counts),
        "modes": dict(mode_counts),
        "dimensions": dict(dimension_counts),
        "files_with_exif": exif_count,
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_files": sum(len(group) for group in duplicate_groups),
        "exact_duplicates": duplicate_groups,
        "perceptual_similarity": {
            "phash_max_distance": phash_max_distance,
            "ssim_min": ssim_min,
            "ssim_image_size": ssim_size,
            "candidate_pairs": len(pairs),
            "candidate_images": len(candidate_group_map),
            "connected_components": len(component_sizes),
            "component_size_distribution": dict(
                sorted(Counter(component_sizes.values()).items())
            ),
            "cross_class_pairs": sum(bool(pair["cross_class"]) for pair in pairs),
            "candidate_images_by_class": dict(sorted(candidate_images_by_class.items())),
            "interpretation": (
                "Similarity candidates are not confirmed duplicate images or proven label errors. "
                "Review them manually before making label corrections."
            ),
        },
        "provided_dataset_stats": provided_stats,
    }
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    with (output_dir / "class_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "image_count", "perceptual_candidate_images"])
        for class_name in CLASS_NAMES:
            writer.writerow(
                [class_name, class_counts[class_name], candidate_images_by_class[class_name]]
            )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--phash-max-distance", type=int, default=5)
    parser.add_argument("--ssim-min", type=float, default=0.95)
    parser.add_argument("--ssim-size", type=int, default=256)
    args = parser.parse_args()
    result = audit_dataset(
        args.dataset_root,
        args.output_dir,
        phash_max_distance=args.phash_max_distance,
        ssim_min=args.ssim_min,
        ssim_size=args.ssim_size,
    )
    print(
        f"Audited {result['total_images']} readable images across "
        f"{len(result['class_counts'])} classes."
    )
    print(f"Corrupt images: {result['corrupt_images']}")
    print(f"Exact duplicate groups: {result['exact_duplicate_groups']}")
    print(
        f"Perceptual candidate pairs: "
        f"{result['perceptual_similarity']['candidate_pairs']}"
    )
    print(f"Reports written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
