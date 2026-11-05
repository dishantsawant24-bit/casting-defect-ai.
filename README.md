# Casting Defect AI

A local-first Streamlit application for image-based casting surface inspection using a trained 12-class ResNet18 classifier.

## What the application does

- Accepts a casting image through file upload or a connected camera.
- Applies a lightweight frontal-face exclusion heuristic before classification.
- Runs a locally supplied PyTorch ResNet18 checkpoint.
- Reports **PASS**, **FAIL**, or **REVIEW** based on predicted class and confidence.
- Displays predicted defect class, confidence, recent inspection metadata, and a recent-result chart.
- Stores inspection metadata in a local SQLite database; image bytes are not retained by the application.

This is a development and demonstration tool, not a certified inspection system. Do not use it as the sole basis for safety-critical or final production acceptance decisions.

## Defect classes

The configured class order is: `ok`, `scratch`, `crack`, `porosity`, `pinhole`, `inclusion`, `shrinkage`, `surface_roughness`, `dent`, `cold_shut`, `flash`, `mixed_defects`.

The predictor checks the class order stored in the checkpoint, when present. Do not reorder these labels without verifying the original training mapping.

## Model checkpoint

The source repository intentionally excludes model weights. Place your approved checkpoint at `models/resnet18_best.pth`. The checkpoint is excluded because it is a 44.8 MB binary artifact and redistribution rights should be confirmed before public distribution.

The loader uses PyTorch's restricted `weights_only=True` loader, validates the stored class order and class count, and fails closed when the checkpoint is missing or incompatible. The checkpoint metadata observed during project review included epoch 9 and best-validation-accuracy metadata of approximately 78.89%; these are checkpoint metadata, not independently reproduced evaluation results.

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the approved checkpoint at `models/resnet18_best.pth`, then run:

```bash
python -m streamlit run app/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

On Windows, `run_streamlit.bat` is also provided.

## Packaging for Windows

Install `requirements-packaging.txt`, place the model checkpoint locally, then run:

```powershell
pyinstaller CastingInspectionAI.spec
```

The PyInstaller bundle includes the checkpoint. Treat the generated executable as containing extractable model weights. Distribute it only if you have rights to share them.

## Decisions and limitations

- `ok` with confidence at or above 70% becomes PASS.
- A defect class at or above 70% becomes FAIL.
- Any prediction below 70% becomes REVIEW.
- An unreadable image or image rejected by the face-exclusion heuristic becomes INVALID.
- The face gate is not a casting-object detector and may produce false positives or false negatives.
- Dataset provenance, original training code, and a complete held-out evaluation report are not included in this source release. No verified test-accuracy claim is made.
- Original sample images are not included because redistribution rights have not been established.

## Local data and privacy

Images are written to a temporary directory only while an inspection runs and are removed when processing exits. The SQLite database stores timestamp, sanitized image filename, decision, predicted class, and confidence—not image bytes. Use **CLEAR LOCAL HISTORY** to delete stored records.

The application performs inference locally and does not call an external prediction API. The launcher binds to `127.0.0.1` and disables Streamlit usage statistics. See [docs/privacy.md](docs/privacy.md).

## Development and tests

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app launcher.py src tests
ruff check app launcher.py src tests
pytest -q
```

The checkpoint is not included, so tests that require its actual weights must be run locally after placing the approved file.

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md). Never upload credentials, customer images, private datasets, local databases, logs, or unapproved model weights.
