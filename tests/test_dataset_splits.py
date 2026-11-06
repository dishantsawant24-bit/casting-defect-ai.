import csv
from collections import Counter

from scripts.create_splits import CLASS_NAMES, create_splits


def test_create_splits_keeps_candidate_components_in_training(tmp_path):
    manifest = tmp_path / "image_manifest.csv"
    candidates = tmp_path / "perceptual_candidates.csv"
    output = tmp_path / "splits"

    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["image_path", "label", "sha256", "width", "height", "format", "mode"]
        )
        for class_name in CLASS_NAMES:
            for index in range(100):
                path = f"{class_name}/{class_name}_{index:03d}.jpg"
                writer.writerow(
                    [path, class_name, f"{class_name}-{index}", 512, 512, "JPEG", "RGB"]
                )

    with candidates.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "image_path_1", "label_1", "image_path_2", "label_2",
            "phash_distance", "ssim", "cross_class",
        ])
        writer.writerow([
            "ok/ok_000.jpg", "ok", "crack/crack_000.jpg", "crack", 1, 0.98, True,
        ])
        writer.writerow([
            "crack/crack_000.jpg", "crack", "crack/crack_001.jpg", "crack", 2, 0.97, False,
        ])

    result = create_splits(manifest, candidates, output, seed=42)

    assert result["candidate_edges_crossing_splits"] == 0
    assert result["candidate_images"] == 3
    assert result["candidate_connected_groups"] == 1
    for split_name, expected_count in (("train", 840), ("val", 180), ("test", 180)):
        with (output / f"{split_name}.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == expected_count
        assert Counter(row["label"] for row in rows) == {
            name: expected_count // 12 for name in CLASS_NAMES
        }
    with (output / "train.csv").open(encoding="utf-8") as stream:
        train_paths = {row["image_path"] for row in csv.DictReader(stream)}
    assert {
        "ok/ok_000.jpg", "crack/crack_000.jpg", "crack/crack_001.jpg"
    }.issubset(train_paths)
