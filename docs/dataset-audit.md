# Casting Defect AI — dataset audit

## Summary

The uploaded dataset contains 1,200 readable JPEG images organised into 12 balanced class folders (100 images per class). All images are RGB and 512 × 512 pixels. The audit found no corrupt images, byte-identical duplicate groups, or EXIF metadata.

## Perceptual similarity screening

The audit computes a DCT perceptual hash from a 32 × 32 grayscale thumbnail. Candidate pairs are those with pHash Hamming distance ≤ 5 and structural similarity (SSIM) ≥ 0.95, with SSIM computed on 256 × 256 grayscale thumbnails.

The scan found:

- 111 candidate image pairs
- 103 distinct images participating in candidate pairs
- 27 connected components
- 93 candidate pair links connecting different folder labels

These are **similarity candidates**, not confirmed duplicate images or confirmed label errors. High similarity can arise from images of the same casting geometry, a repeated capture setup, or different defects on similar parts. Review candidates manually with someone familiar with the defect taxonomy before changing any labels.

## Conservative split

The split generator keeps every image in a perceptual-candidate connected component in the training partition. It then selects remaining images from non-candidate singleton files to reach a balanced 70 / 15 / 15 split per class:

| Split | Images | Per class |
|---|---:|---:|
| Train | 840 | 70 |
| Validation | 180 | 15 |
| Test | 180 | 15 |

No detected candidate pair crosses a split boundary. This is more conservative than a random image-level split. It is not a substitute for grouping by physical-part ID, casting batch, or capture session when those identifiers are available.

## Existing checkpoint limitation

The previously supplied ResNet18 checkpoint records validation accuracy of approximately 78.89% at epoch 9 in its metadata. The original data/splits manifests used to train it were not supplied, so it may already have seen images that are now in the generated test manifest. **Do not report a score from that old checkpoint on this new test split as independent test performance.**

For a defensible evaluation, train a fresh model from train.csv, use val.csv only for model selection, and evaluate the selected checkpoint once against test.csv.

## Data handling and rights

The original archive and all image files remain local and are excluded from the public repository. No dataset licence or permission to redistribute the images was included with the archive. The audit report, split-generation scripts, and aggregate findings do not contain image payloads.
