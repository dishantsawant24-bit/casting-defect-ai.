"""Create balanced splits while keeping perceptual candidate groups in training."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLASS_NAMES = [
    "ok", "scratch", "crack", "porosity", "pinhole", "inclusion",
    "shrinkage", "surface_roughness", "dent", "cold_shut", "flash", "mixed_defects",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def candidate_groups(pairs: list[dict[str, str]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    for pair in pairs:
        root_a = find(pair["image_path_1"])
        root_b = find(pair["image_path_2"])
        if root_a != root_b:
            if root_a < root_b:
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(parent):
        groups[find(path)].append(path)
    ordered = sorted((sorted(paths) for paths in groups.values()), key=lambda paths: paths[0])
    return {
        path: f"similar_group_{index:02d}"
        for index, paths in enumerate(ordered, start=1)
        for path in paths
    }


def create_splits(
    manifest_path: Path,
    candidates_path: Path,
    output_dir: Path,
    train_per_class: int = 70,
    val_per_class: int = 15,
    test_per_class: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    if train_per_class + val_per_class + test_per_class != 100:
        raise ValueError("For this balanced dataset, split counts must sum to 100 per class.")
    manifest = read_csv(manifest_path)
    pairs = read_csv(candidates_path)
    groups = candidate_groups(pairs)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        if row["label"] not in CLASS_NAMES:
            raise ValueError(f"Unexpected class in manifest: {row['label']}")
        by_class[row["label"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [], "val": [], "test": [],
    }
    for class_name in CLASS_NAMES:
        rows = sorted(by_class[class_name], key=lambda row: row["image_path"])
        if len(rows) != 100:
            raise ValueError(f"Expected exactly 100 images in {class_name}; found {len(rows)}.")
        flagged = [row for row in rows if row["image_path"] in groups]
        clean = [row for row in rows if row["image_path"] not in groups]
        if len(flagged) > train_per_class:
            raise ValueError(
                f"{class_name} has {len(flagged)} candidate images, exceeding the training quota. "
                "Review similarity groups and adjust split ratios manually."
            )
        if len(clean) < train_per_class - len(flagged) + val_per_class + test_per_class:
            raise ValueError(f"Not enough non-candidate images to split class {class_name}.")
        rng.shuffle(clean)
        train_singletons = train_per_class - len(flagged)
        for row in flagged:
            split_rows["train"].append({
                **row,
                "similarity_group": groups[row["image_path"]],
                "selection_reason": "candidate_component_kept_in_train",
            })
        for row in clean[:train_singletons]:
            split_rows["train"].append({
                **row,
                "similarity_group": f"single:{row['image_path']}",
                "selection_reason": "train_singleton",
            })
        val_start = train_singletons
        for row in clean[val_start:val_start + val_per_class]:
            split_rows["val"].append({
                **row,
                "similarity_group": f"single:{row['image_path']}",
                "selection_reason": "non_candidate_singleton",
            })
        test_start = val_start + val_per_class
        for row in clean[test_start:test_start + test_per_class]:
            split_rows["test"].append({
                **row,
                "similarity_group": f"single:{row['image_path']}",
                "selection_reason": "non_candidate_singleton",
            })

    counts: dict[str, Any] = {}
    assignments: dict[str, str] = {}
    fields = [
        "image_path", "label", "sha256", "similarity_group", "selection_reason",
    ]
    for split_name, rows in split_rows.items():
        rows.sort(key=lambda row: row["image_path"])
        for row in rows:
            assignments[row["image_path"]] = split_name
        counts[split_name] = {
            "total": len(rows),
            "per_class": dict(sorted(Counter(row["label"] for row in rows).items())),
        }
        with (output_dir / f"{split_name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row[field] for field in fields} for row in rows)

    crossing = [
        pair for pair in pairs
        if assignments.get(pair["image_path_1"]) != assignments.get(pair["image_path_2"])
    ]
    if crossing:
        raise RuntimeError(f"Similarity candidates cross split boundaries: {len(crossing)} pairs.")
    summary = {
        "seed": seed,
        "ratios": {
            "train": train_per_class / 100,
            "val": val_per_class / 100,
            "test": test_per_class / 100,
        },
        "counts": counts,
        "candidate_pairs": len(pairs),
        "candidate_images": len(groups),
        "candidate_connected_groups": len(set(groups.values())),
        "candidate_edges_crossing_splits": len(crossing),
        "strategy": (
            "All images in a perceptual-similarity connected component are placed in training. "
            "Validation and test are sampled only from images not in any candidate pair."
        ),
        "warning": (
            "Similarity flags are screening candidates, not confirmed duplicates or label errors. "
            "This does not replace grouping by physical part, batch, or capture session IDs."
        ),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = create_splits(args.manifest, args.candidates, args.output_dir, seed=args.seed)
    for split_name, info in result["counts"].items():
        print(f"{split_name}: {info['total']} images")
    print(
        "Near-match candidate edges crossing splits: "
        f"{result['candidate_edges_crossing_splits']}"
    )
    print(f"Split files written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
