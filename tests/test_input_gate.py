import cv2
import numpy as np
from src.inference.input_gate import check_input


def test_missing_image_is_rejected(tmp_path):
    assert check_input(tmp_path / "missing.jpg")["valid"] is False


def test_unreadable_image_is_rejected(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_text("not an image", encoding="utf-8")
    assert check_input(path)["valid"] is False


def test_plain_image_passes_face_gate(tmp_path):
    path = tmp_path / "plain.png"
    assert cv2.imwrite(str(path), np.zeros((160, 160, 3), dtype=np.uint8))
    result = check_input(path)
    assert result["valid"] is True
    assert result["face_detected"] is False
