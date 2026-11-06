# Project roadmap

Last reviewed: 2026-11-06. A checked item means the work is present in the public repository or was verified during the local project review. A checked smoke test is not a claim of production validation.

## Completed

- [x] Integrate the original Streamlit inspection workflow with image upload and camera capture.
- [x] Integrate the 12-class ResNet18 inference path and the project's PASS / FAIL / REVIEW decision threshold.
- [x] Preserve the project's 224 × 224 image resize and ImageNet normalization in the inference pipeline.
- [x] Add temporary local image handling and SQLite inspection metadata history; the app does not retain image bytes.
- [x] Review the supplied checkpoint locally with restricted PyTorch loading and strict ResNet18 state-dictionary matching.
- [x] Perform a five-image local smoke check. This is only a functional spot-check, not a representative accuracy evaluation.
- [x] Add unit tests, linting, syntax checks, and GitHub Actions CI. The current main-branch CI run passes.
- [x] Add contribution, security, privacy, model-card, and maintenance documentation.
- [x] Add the MIT licence for project source code and document that it does not grant rights to model weights or datasets.

## Highest-priority work

1. [ ] Confirm the provenance and redistribution rights for the model checkpoint, training dataset, and any sample images. Keep unapproved assets out of the public repository.
2. [ ] Recover or document the exact training configuration, class-index mapping, and training-time transforms. The original training code and dataset were not present in the supplied project archive.
3. [ ] Add a reproducible held-out evaluation command that reports per-class precision, recall, F1, support, and a confusion matrix.
4. [ ] Run the complete Streamlit application end to end with the approved local checkpoint, including upload, camera, invalid input, missing checkpoint, and incompatible checkpoint scenarios.
5. [ ] Record the Python, PyTorch, torchvision, OpenCV, and Streamlit versions used in a verified working environment.
6. [ ] Add redistributable sample images only when source, consent, and licence are documented.

## Release and community

- [ ] Publish a first tagged release only after the full application has been run end to end and the setup instructions have been verified from a clean environment.
- [ ] Invite a small number of genuine manufacturing / computer-vision users to try the project and report issues.
- [ ] Triage incoming issues and review external pull requests on their merits.
- [ ] Maintain dependency-update pull requests after checking compatibility, not by blindly merging every automated update.

## Known release limitations

- The trained checkpoint is not included in the public repository; users need an approved local copy at `models/resnet18_best.pth`.
- The dataset and original training code are not included.
- The approximately 78.89% value found in checkpoint metadata is not an independently reproduced held-out metric and must not be presented as verified model accuracy.
- The current five-image smoke check is too small to establish generalization, production readiness, or a reliable defect-detection rate.
- The Haar frontal-face gate is a heuristic, not a casting-object detector.
