from pathlib import Path
import pytest
from src.inference.predict import CLASS_NAMES, CONFIDENCE_REVIEW_THRESHOLD, ModelUnavailableError, inspection_decision, load_model, predict


def test_class_order_has_twelve_unique_labels():
    assert len(CLASS_NAMES) == 12
    assert len(set(CLASS_NAMES)) == 12


def test_high_confidence_ok_is_pass():
    assert inspection_decision("ok", 0.95) == "PASS"


def test_high_confidence_defect_is_fail():
    assert inspection_decision("crack", 0.91) == "FAIL"


def test_low_confidence_requires_review():
    low = CONFIDENCE_REVIEW_THRESHOLD - 0.01
    assert inspection_decision("ok", low) == "REVIEW"
    assert inspection_decision("crack", low) == "REVIEW"


def test_unknown_class_and_invalid_confidence_are_rejected():
    with pytest.raises(ValueError):
        inspection_decision("unknown", 0.9)
    with pytest.raises(ValueError):
        inspection_decision("ok", 1.1)


def test_missing_checkpoint_fails_safely(tmp_path: Path):
    with pytest.raises(ModelUnavailableError, match="missing"):
        load_model(tmp_path / "missing.pth")


def test_missing_image_fails_safely(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        predict(tmp_path / "missing.jpg")
