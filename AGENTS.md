# Repository instructions for coding agents

## Project intent

Build a practical, transparent demonstration for image-based casting defect inspection. Prefer the smallest reliable implementation over unnecessary infrastructure.

## Non-negotiable correctness rules

- Never fabricate model predictions, accuracy, evaluation results, dataset provenance, or completed features.
- Never convert a missing, invalid, or uncertain model output into PASS.
- Keep class names and class-index mapping aligned with the verified training artefacts. Do not reorder labels by assumption.
- Keep preprocessing aligned with the actual training pipeline. Verify image size, normalization, channels, and transforms before changing them.
- Do not replace a trained checkpoint with randomly initialized weights.
- Do not commit secrets, credentials, private datasets, production/customer images, or large model files without explicit approval and rights verification.
- Do not make safety-critical claims. This is a demonstration aid, not a certified inspection system.

## Change workflow

1. Inspect relevant files and repository history before editing.
2. Make the smallest focused change.
3. Add or update tests.
4. Run syntax checks, Ruff, and pytest when the environment allows.
5. Update documentation for user-visible or model-pipeline changes.
6. Report exactly what was changed and which checks were actually run.

## Model integration

The current repository scaffold expects a 12-class ResNet18 checkpoint at `models/resnet18_best.pth`. This expectation is provisional until the owner's original project ZIP is reviewed. Do not assume the checkpoint key format, class order, image transform, or architecture without examining the original training code and artefacts.

## Git discipline

- Use a feature branch for substantive changes.
- Do not force-push or rewrite history.
- Do not delete user files without explaining why and obtaining approval.
- Keep pull requests focused and descriptive.
