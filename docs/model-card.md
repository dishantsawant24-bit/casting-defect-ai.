# Model card: Casting Defect AI

**Status:** Provisional scaffold. The original trained checkpoint and training project have not yet been integrated or independently verified.

## Intended use

A demonstration workflow for exploring image-based surface-defect classification on cast components. It is intended for development, learning, and controlled evaluation.

## Out-of-scope use

- Final acceptance or rejection of production components without a validated quality process.
- Safety-critical decisions.
- Claims of certification, regulatory compliance, or guaranteed defect detection.
- Use on materials, geometries, lighting conditions, cameras, or defect categories not represented in the verified training/evaluation data.

## Configured labels

The current scaffold lists: `ok`, `scratch`, `crack`, `porosity`, `pinhole`, `inclusion`, `shrinkage`, `surface_roughness`, `dent`, `cold_shut`, `flash`, and `mixed_defects`.

This list and its order are provisional until confirmed against the original training class mapping.

## Model architecture

The current inference scaffold is written for a 12-output ResNet18 model. This is not yet evidence that the owner's existing checkpoint uses this exact architecture or preprocessing.

## Data

Dataset source, collection conditions, labelling method, class balance, permissions, and train/validation/test split are **not yet documented**. They must be added after reviewing the original project and verifying redistribution rights.

## Evaluation

No verified evaluation metrics are published in this repository at this stage. Accuracy, precision, recall, F1, per-class performance, confusion matrix, and calibration must be measured on a held-out labelled test set before any performance claims are made.

## Known limitations and risks

- Lighting, viewpoint, surface finish, image quality, part geometry, and dataset shift may affect predictions.
- A confidence score is not a guarantee of correctness.
- Similar-looking defects may be confused.
- The current decision rule maps the predicted `ok` class to PASS and all configured defect labels to FAIL; its suitability requires validation with domain experts.

## Update policy

Any change to the architecture, checkpoint, preprocessing, labels, thresholds, or dataset should be recorded with its rationale and associated reproducible evaluation.
