# Contributing

Thank you for considering a contribution to Casting Defect AI.

The project is an evolving demonstration of image-based casting defect inspection. Contributions should improve reproducibility, usability, reliability, documentation, or responsible evaluation.

## Before starting

1. Search existing issues and pull requests to avoid duplicate work.
2. For a substantial change, open an issue first to discuss the proposed approach.
3. Do not share confidential production images, personal data, credentials, or proprietary datasets in issues or pull requests.
4. Do not add model checkpoints or datasets unless their ownership, licence, size, and redistribution rights have been confirmed.

## Development setup

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ruff
```

## Checks before opening a pull request

```bash
python -m compileall -q app.py src tests
ruff check app.py src tests
pytest -q
```

## Pull request expectations

- Keep changes focused and explain why they are needed.
- Add or update tests for changed behaviour.
- Update the README and model documentation when interfaces, preprocessing, class labels, or model requirements change.
- Record the exact dataset, checkpoint, class mapping, and evaluation protocol for model-related changes.
- Never report accuracy, speed, or performance without reproducible measurements.
- Never introduce fake predictions or turn an inference error into a PASS result.
- Do not commit secrets, virtual environments, cache files, private datasets, or unapproved model artifacts.

## Review process

A maintainer may ask for clarification, tests, documentation, or a smaller change set before merging. Review decisions should be based on correctness, clarity, reproducibility, and project scope.

## Licensing

The repository's licence must be checked before contributing. Until a licence is added, do not assume that the code is available for unrestricted reuse or redistribution. By submitting a contribution, you confirm that you have the right to submit it and that it does not contain third-party or confidential material.
