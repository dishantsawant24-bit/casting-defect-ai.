# Casting Defect AI

An operator-friendly demonstration app for image-based casting surface-defect inspection.

## What it does

- Accepts a casting/component image.
- Runs a locally supplied ResNet18 classifier checkpoint.
- Reports **PASS** for the `ok` class and **FAIL** for a predicted defect class.
- Displays the predicted class and model confidence.
- Fails safely when the model checkpoint is missing or incompatible.

## Defect classes

`ok`, `scratch`, `crack`, `porosity`, `pinhole`, `inclusion`, `shrinkage`, `surface_roughness`, `dent`, `cold_shut`, `flash`, `mixed_defects`.

> Important: the class order must match the order used while training the supplied checkpoint. Confirm it before using predictions for any quality decision.

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Model checkpoint

Place the trained ResNet18 checkpoint at:

`models/resnet18_best.pth`

The model file is intentionally not included in this repository. Do not commit private datasets, credentials, or large model weights without reviewing their ownership and repository-storage requirements.

The inference loader accepts a PyTorch state dictionary or a checkpoint containing a `state_dict` key. It expects a 12-class ResNet18 model. If your checkpoint uses a different architecture, preprocessing, or class order, update `src/inference.py` to match the original training pipeline rather than guessing.

## Current limitations

- This repository scaffold does not contain trained weights or a dataset.
- No accuracy claim is made until evaluation is run against a separate, labelled test set.
- This is a demonstration aid, not a certified or safety-critical inspection system.
- Always confirm predictions with an approved inspection process.

## Tests

```bash
pytest -q
```

The tests cover decision logic and safe handling of invalid inputs. They do not validate model accuracy.

## Project structure

```
.
├── app.py
├── src/
│   ├── __init__.py
│   └── inference.py
├── tests/
│   └── test_inference.py
├── models/              # Add your approved checkpoint locally
├── requirements.txt
└── .gitignore
```
