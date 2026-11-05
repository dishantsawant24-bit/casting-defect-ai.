"""ResNet18 inference for the Casting Defect AI inspection application."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "resnet18_best.pth"
CLASS_NAMES = [
    "ok", "scratch", "crack", "porosity", "pinhole", "inclusion",
    "shrinkage", "surface_roughness", "dent", "cold_shut", "flash", "mixed_defects",
]
CONFIDENCE_REVIEW_THRESHOLD = 0.70
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ModelUnavailableError(RuntimeError):
    """Raised when a trained checkpoint is absent or incompatible."""


def inspection_decision(class_name: str, confidence: float) -> str:
    """Map high-confidence OK to PASS, defect to FAIL, low confidence to REVIEW."""
    if class_name not in CLASS_NAMES:
        raise ValueError(f"Unknown model class: {class_name}")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    if confidence < CONFIDENCE_REVIEW_THRESHOLD:
        return "REVIEW"
    return "PASS" if class_name == "ok" else "FAIL"


@lru_cache(maxsize=2)
def _load_model_cached(checkpoint_path: str) -> tuple[torch.nn.Module, torch.device]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise ModelUnavailableError(
            "Trained model checkpoint is missing. Place resnet18_best.pth in the models folder."
        )
    try:
        checkpoint: Any = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ModelUnavailableError(
            "Checkpoint could not be loaded safely as a PyTorch weights-only file."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ModelUnavailableError("Checkpoint format is invalid.")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ModelUnavailableError("Checkpoint does not contain model_state_dict weights.")
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes is not None and list(checkpoint_classes) != CLASS_NAMES:
        raise ModelUnavailableError("Checkpoint class order does not match configured labels.")
    if int(checkpoint.get("num_classes", len(CLASS_NAMES))) != len(CLASS_NAMES):
        raise ModelUnavailableError("Checkpoint class count is not 12.")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ModelUnavailableError("Checkpoint weights do not match the expected ResNet18.") from exc
    model.to(DEVICE)
    model.eval()
    return model, DEVICE


def load_model(checkpoint_path: str | Path = MODEL_PATH) -> tuple[torch.nn.Module, torch.device]:
    """Load and cache the trained model."""
    return _load_model_cached(str(Path(checkpoint_path).expanduser().resolve()))


def predict(image_path: str | Path) -> dict[str, float | str]:
    """Classify an image and return result, defect class, and confidence."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError("Inspection image was not found.")
    model, device = load_model()
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Inspection image is not a supported, readable image.") from exc
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 2 or tuple(logits.shape) != (1, len(CLASS_NAMES)):
            raise RuntimeError("Model output does not contain 12 class scores.")
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
    class_name = CLASS_NAMES[int(index.item())]
    confidence_value = float(confidence.item())
    return {
        "result": inspection_decision(class_name, confidence_value),
        "defect": class_name,
        "confidence": confidence_value,
    }
