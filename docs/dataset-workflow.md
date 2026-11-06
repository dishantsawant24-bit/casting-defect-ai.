# Local dataset workflow

The dataset is intentionally not committed to Git. Keep the original ZIP and extracted images on your machine.

## 1. Install optional evaluation tooling

From the repository root:

```bash
python -m pip install -r requirements-evaluation.txt
```

## 2. Audit the extracted class folders

The dataset-root path must point to the directory containing the 12 class folders (ok, scratch, crack, and so on):

```bash
python scripts/audit_dataset.py --dataset-root "D:/archive (2)/output/defect_dataset" --output-dir outputs/dataset_audit
```

The audit writes dataset_audit.json, class_counts.csv, image_manifest.csv, and perceptual_candidates.csv.

## 3. Generate conservative splits

```bash
python scripts/create_splits.py --manifest outputs/dataset_audit/image_manifest.csv --candidates outputs/dataset_audit/perceptual_candidates.csv --output-dir data/splits --seed 42
```

The default split is 70 train / 15 validation / 15 test images per class. Every connected group of perceptually similar candidate images is retained in training; validation and test are sampled only from images not involved in a candidate pair. The generator fails instead of silently creating an imbalanced split if those constraints cannot be met.

## 4. Train a fresh model

```bash
python scripts/train_resnet18.py --dataset-root "D:/archive (2)/output/defect_dataset" --splits-dir data/splits
```

By default, the training script uses the public ImageNet-pretrained ResNet18 weights provided by Torchvision. This may download the weights from the PyTorch hosting service; there is no paid API or subscription. Use --no-pretrained to initialise from random weights instead.

## 5. Evaluate once on the held-out test set

```bash
python scripts/evaluate_resnet18.py --checkpoint models/resnet18_best.pth --split-csv data/splits/test.csv --dataset-root "D:/archive (2)/output/defect_dataset" --output-dir outputs/evaluation
```

The evaluator checks the checkpoint class order and writes an aggregate JSON report, a confusion matrix CSV, and image-level predictions CSV. Keep the test set out of training and model selection.

## Important limits

- Similarity candidates are flags for human review, not automatic proof of duplicates or wrong labels.
- The generated split does not group by physical part, production batch, or camera session because those IDs were not provided.
- Do not score the old checkpoint on the new test manifest as independent performance: the old checkpoint's training split was not provided.
- Do not upload customer or restricted images, the dataset ZIP, or model weights without confirmed redistribution rights.
