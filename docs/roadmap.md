# Project roadmap

Last reviewed: 2026-11-06. A checked item means the work is present in the public repository or was verified during the local project review. A checked smoke test is not a claim of production validation.

## Completed

- [x] Integrate the original Streamlit inspection workflow with image upload and camera capture.
- [x] Integrate the 12-class ResNet18 inference path and the project's PASS / FAIL / REVIEW decision threshold.
- [x] Preserve the project's 224 × 224 image resize and ImageNet normalization in the inference pipeline.
- [x] Add temporary local image handling and SQLite inspection metadata history; the app does not retain image bytes.
- [x] Review the supplied checkpoint locally with restricted PyTorch loading and strict ResNet18 state-dictionary matching.
- [x] Perform a five-image local smoke check. This is only a functional spot-check, not a representative accuracy evaluation.
- [x] Audit the supplied 1,200-image dataset: 100 readable images per class, 512 × 512 RGB JPEGs, no corrupt images or exact duplicate groups.
- [x] Add perceptual-similarity screening and candidate-aware 70 / 15 / 15 split generation. The audited dataset's 111 candidate pairs do not cross the generated split boundaries.
- [x] Add local ResNet18 training and held-out evaluation commands, with manifest and checkpoint hashes in the reports.
- [x] Add unit tests, linting, syntax checks, and GitHub Actions CI.
- [x] Add contribution, security, privacy, model-card, dataset-audit, and maintenance documentation.
- [x] Add the MIT licence for project source code and document that it does not grant rights to model weights or datasets.

## Highest-priority work

1. [ ] Manually review the perceptual-similarity candidate pairs and verify class labels with someone familiar with the casting defect taxonomy.
2. [ ] Confirm the provenance and redistribution rights for the dataset, model checkpoint, and any sample images. Keep unapproved assets out of the public repository.
3. [ ] Train a fresh checkpoint on the generated train split, use validation only for model selection, and publish held-out test metrics from the new test split.
4. [ ] Run the complete Streamlit application end to end with the newly trained checkpoint, including upload, camera, invalid input, missing checkpoint, and incompatible checkpoint scenarios.
5. [ ] Record the Python, PyTorch, torchvision, OpenCV, and Streamlit versions used in the verified working environment.
6. [ ] Add redistributable sample images only when source, consent, and licence are documented.

## Release and community

- [ ] Publish a first tagged release only after the full application has been run end to end and the setup instructions have been verified from a clean environment.
- [ ] Invite a small number of genuine manufacturing / computer-vision users to try the project and report issues.
- [ ] Triage incoming issues and review external pull requests on their merits.
- [ ] Maintain dependency-update pull requests after checking compatibility, not by blindly merging every automated update.

## Known release limitations

- The trained checkpoint is not included in the public repository; users need an approved local copy at `models/resnet18_best.pth`.
- The dataset is not included in the repository, and permission to redistribute it has not been established.
- The approximately 78.89% value found in the old checkpoint metadata is not an independently reproduced held-out metric and must not be presented as verified model accuracy. The old checkpoint must not be scored on the newly generated test split because its original split manifests were not provided.
- The current five-image smoke check is too small to establish generalization, production readiness, or a reliable defect-detection rate.
- The perceptual-similarity flags are candidates for manual review, not confirmed duplicates or label errors; no physical-part or capture-session identifiers were supplied.
- The Haar frontal-face gate is a heuristic, not a casting-object detector.
