from pathlib import Path

import pytest
from PIL import Image

from src.data.audit import create_stratified_splits, scan_dataset


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, format="JPEG")


def test_scan_dataset_counts_valid_and_corrupt_images(tmp_path: Path) -> None:
    _save_image(tmp_path / "ok" / "valid.jpg", (255, 255, 255))
    (tmp_path / "ok" / "corrupt.jpg").write_text("not an image", encoding="utf-8")
    _save_image(tmp_path / "scratch" / "valid.jpg", (0, 0, 0))

    report, records = scan_dataset(tmp_path, expected_classes=("ok", "scratch"))

    assert report["total_images"] == 3
    assert report["valid_images"] == 2
    assert report["corrupt_images"] == 1
    assert len(records) == 2
    assert report["class_counts"]["ok"] == {"found": 2, "valid": 1, "corrupt": 1}


def test_stratified_splits_are_deterministic_and_balanced() -> None:
    records = [
        {"image_path": f"{label}/{index:03}.jpg", "label": label, "sha256": f"{label}-{index}"}
        for label in ("ok", "crack")
        for index in range(20)
    ]
    first = create_stratified_splits(records, seed=7)
    second = create_stratified_splits(records, seed=7)

    assert first == second
    assert {name: len(items) for name, items in first.items()} == {
        "train": 28, "val": 6, "test": 6,
    }
    for split, expected in {"train": 14, "val": 3, "test": 3}.items():
        assert sum(item["label"] == "ok" for item in first[split]) == expected


def test_exact_duplicates_block_split_creation() -> None:
    records = [
        {"image_path": "ok/one.jpg", "label": "ok", "sha256": "same"},
        {"image_path": "ok/two.jpg", "label": "ok", "sha256": "same"},
        {"image_path": "ok/three.jpg", "label": "ok", "sha256": "other"},
    ]
    with pytest.raises(ValueError, match="Exact duplicate"):
        create_stratified_splits(records)


def test_perceptual_candidates_flag_similar_images_across_classes(tmp_path: Path) -> None:
    from PIL import ImageEnhance

    from src.data.audit import _find_perceptual_candidates

    first = Image.new("RGB", (128, 128), (110, 110, 110))
    for x in range(16, 112, 16):
        for y in range(16, 112, 16):
            first.putpixel((x, y), (220, 220, 220))
    second = ImageEnhance.Brightness(first).enhance(1.03)
    (tmp_path / "ok").mkdir(parents=True)
    (tmp_path / "crack").mkdir(parents=True)
    first.save(tmp_path / "ok" / "one.jpg", quality=95)
    second.save(tmp_path / "crack" / "two.jpg", quality=95)
    records = [
        {"image_path": "ok/one.jpg", "label": "ok"},
        {"image_path": "crack/two.jpg", "label": "crack"},
    ]

    candidates = _find_perceptual_candidates(tmp_path, records)

    assert len(candidates) == 1
    assert candidates[0]["cross_class"] is True
    assert candidates[0]["ssim"] >= 0.95
