# Project roadmap

This roadmap reflects the current repository scaffold. Items are not complete until implemented and verified.

## Foundation

- [x] Create a documented Streamlit demonstration scaffold.
- [x] Add provisional 12-class ResNet18 inference interface.
- [x] Add basic decision-logic and image-input tests.
- [ ] Verify the real checkpoint architecture and state-dictionary format.
- [ ] Verify the training-time image transforms and class-index mapping.
- [ ] Integrate the owner's original project after reviewing its source and artefacts.

## Reliability and evaluation

- [ ] Add representative, redistributable sample images with documented provenance.
- [ ] Create a reproducible held-out evaluation script.
- [ ] Report per-class precision, recall, F1, support, and confusion matrix.
- [ ] Test corrupt images, unsupported formats, missing checkpoint, and incompatible checkpoint.
- [ ] Document known limitations and operating conditions.

## Open-source maintenance

- [x] Add contribution guidance and pull-request checklist.
- [x] Add issue templates and security reporting guidance.
- [x] Add CI and dependency-update configuration.
- [ ] Select and add an open-source licence after maintainer approval.
- [ ] Publish a first tagged release after model integration and validation.
- [ ] Triage real user feedback and external contributions as they arrive.

## Deployment

- [ ] Add reproducible deployment instructions after the runtime environment is confirmed.
- [ ] Add a privacy-conscious inspection history only after the storage and retention model is agreed.
