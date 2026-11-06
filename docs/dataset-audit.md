# Dataset audit and split generation

The full casting dataset is intentionally **not stored in this repository**. Keep the ZIP, raw images, local split manifests, model weights, and generated reports on your own machine unless redistribution rights are explicitly established.

## Expected folder layout

Extract the dataset so that the 12 class folders sit directly inside one root directory:

```text
defect_dataset/
  ok/
  scratch/
  crack/
  porosity/
  pinhole/
  inclusion/
  shrinkage/
  surface_roughness/
  dent/
  cold_shut/
  flash/
  mixed_defects/
```

Non-image files such as `dataset_stats.json` are allowed. Supported image types are JPG/JPEG, PNG, BMP, TIFF, and WEBP.

## Run locally

Install the project dependencies, then run:

```bash
python -m src.data.audit --dataset-root /path/to/defect_dataset --output-dir outputs/dataset_audit --seed 42
```

On Windows PowerShell, replace the dataset path with the extracted folder path, for example:

```powershell
python -m src.data.audit --dataset-root \"D:\\\\archive (2)\\\\output\\\\defect_dataset\" --output-dir outputs/dataset_audit --seed 42
```

The command writes:

- `dataset_audit.json`: image counts, decodability, dimensions, formats, class-folder issues, and exact duplicate checks.
- `class_counts.csv`: found, valid, and corrupt image counts per class.
- `splits/train.csv`, `splits/val.csv`, and `splits/test.csv`: relative image paths, labels, and SHA-256 hashes.

The default split is stratified per class at 70% train, 15% validation, and 15% test with seed 42. You can change the ratios with `--train-ratio`, `--val-ratio`, and `--test-ratio`; they must sum to 1.0. Exact duplicate images block split generation so duplicate copies cannot silently leak between splits. Corrupt images are reported and excluded from the manifests.

## Audit of the maintainer-supplied archive (2026-11-06)

The ZIP supplied for this review contained 1,200 JPEG images across 12 classes, with 100 images in each class. All 1,200 images decoded successfully; all were 512 × 512 RGB JPEGs. No exact byte-identical duplicate images were detected. One JSON class-count file was also present.

A seed-42, 70/15/15 split was generated locally: 840 train, 180 validation, and 180 test images. This is a **new split**, not evidence that it matches the membership of any earlier model-training run. Do not use it to claim independent accuracy for an existing checkpoint unless its training-image membership is known and non-overlapping.

Exact duplicate detection uses SHA-256 and does not detect visually similar or near-duplicate images. Automated file integrity checks do not verify whether every human-assigned class label is correct. Review labels and provenance before publishing model-performance claims.
