"""Basic face-exclusion input gate for casting image inspection."""
from __future__ import annotations

from pathlib import Path

import cv2


def check_input(image_path: str | Path, min_face_size: int = 60) -> dict[str, bool | str]:
    """Reject unreadable images and images containing an obvious frontal face.

    This is only a lightweight guardrail; it is not a casting-object detector.
    When the OpenCV build lacks the Haar cascade API, image readability is still
    checked and the gate reports that face screening was skipped.
    """
    path = Path(image_path)
    if not path.is_file():
        return {"valid": False, "reason": "Image file not found.", "face_detected": False}

    image = cv2.imread(str(path))
    if image is None:
        return {"valid": False, "reason": "Image could not be read.", "face_detected": False}

    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return {
            "valid": True,
            "reason": "OpenCV face-screening API unavailable; face screening skipped.",
            "face_detected": False,
        }

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return {
                "valid": True,
                "reason": "Face detector unavailable; face screening skipped.",
                "face_detected": False,
            }
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(min_face_size), int(min_face_size)),
        )
    except (AttributeError, cv2.error, OSError):
        return {
            "valid": True,
            "reason": "Face detector failed; face screening skipped.",
            "face_detected": False,
        }

    if len(faces):
        return {
            "valid": False,
            "reason": "A human face was detected. Upload an image of a casting component.",
            "face_detected": True,
        }
    return {
        "valid": True,
        "reason": "No obvious frontal face detected.",
        "face_detected": False,
    }
