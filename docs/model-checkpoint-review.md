# Model checkpoint review

This note records a limited local integrity and inference smoke check performed against the checkpoint supplied in the owner's private project archive. The checkpoint and example images are not distributed in this repository.

## Integrity and compatibility

- File: resnet18_best.pth
- File size: 44,810,546 bytes (about 44.8 MB)
- Loader: PyTorch weights-only mode (weights_only=True)
- Checkpoint dictionary keys: model_state_dict, num_classes, class_names, best_val_accuracy, epoch
- Architecture loaded: torchvision ResNet18 with a 12-output fully connected layer
- Strict state-dictionary load: passed with zero missing and zero unexpected keys
- Stored class count: 12
- Stored class order: ok, scratch, crack, porosity, pinhole, inclusion, shrinkage, surface_roughness, dent, cold_shut, flash, mixed_defects
- Stored epoch metadata: 9
- Stored best-validation-accuracy metadata: 0.7888888888888889 (about 78.89%)

The validation-accuracy value above is checkpoint metadata. This repository does not contain the original training script, dataset, split manifests, or full evaluation report, so the value has not been independently reproduced.

## Five-image smoke check

The five private archive examples were run through the stored ResNet18 weights using the original 224x224 resize, ImageNet normalization, softmax, and class order.

| Private example filename | Predicted class | Confidence | 70% decision |
|---|---|---:|---|
| sample_crack.jpg | crack | 96.75% | FAIL |
| sample_dent.jpg | dent | 92.40% | FAIL |
| sample_ok.jpg | ok | 50.18% | REVIEW |
| sample_porosity.jpg | porosity | 96.06% | FAIL |
| sample_scratch.jpg | scratch | 60.30% | REVIEW |

These are smoke-test outputs on five supplied examples, not a held-out test evaluation. Filenames were used only as descriptive labels for this check; no overall accuracy, precision, recall, F1, or production performance claim should be inferred from them.

## Release requirements still outstanding

- Confirm ownership and redistribution rights for the checkpoint.
- Obtain the original training code and configuration if reproducibility is required.
- Obtain dataset provenance and split manifests.
- Run a full evaluation on a separately held-out labelled dataset and publish per-class metrics and a confusion matrix.
- Validate across cameras, lighting conditions, casting geometries, and production environments before any operational use.
