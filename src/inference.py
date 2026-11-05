"""Safe inference utilities for the Casting Defect AI demo.

The class order and preprocessing here must match the training pipeline used
to produce the checkpoint. Verify both before relying on predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

CLASS_NAMES = [
    "ok",
    "scratch",
    "crack",
    "porosity",
    "pinhole",
    "inclusion",
    "shrinkage",
    "surface_roughness",
    "dent",
    "cold_shut",
    "flash",
    "mixed_defects",
]

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "resnet18_best.pth"

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


class ModelUnavailableError(RuntimeError):
    """Raised when a usable trained checkpoint is unavailable."""


def inspection_decision(class_name: str) -> str:
    """Return PASS only for the explicitly recognised OK class."""
    if class_name not in CLASS_NAMES:
        raise ValueError(f"Unknown model class: {class_name}")
    return "PASS" if class_name == "ok" else "FAIL"


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict) or not state:
        raise ModelUnavailableError("Checkpoint does not contain a valid state dictionary.")

    # Support checkpoints saved from DataParallel.
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def load_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT):
    """Load a 12-class ResNet18 checkpoint on CPU or CUDA."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ModelUnavailableError(
            f"Model checkpoint not found: {path}. Add the trained checkpoint before inspecting images."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)

        state_dict = _extract_state_dict(checkpoint)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(
            "Could not load this checkpoint into the expected 12-class ResNet18. "
            "Check its architecture, class count, and training configuration."
        ) from exc

    return model, device


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Predict one PIL image and return a structured inspection result."""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a valid PIL image.")

    rgb_image = image.convert("RGB")
    tensor = _PREPROCESS(rgb_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(CLASS_NAMES):
            raise ValueError("Model output must contain one score for each of the 12 configured classes.")
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)

    predicted_class = CLASS_NAMES[int(class_index.item())]
    return {
        "decision": inspection_decision(predicted_class),
        "defect": "None" if predicted_class == "ok" else predicted_class.replace("_", " ").title(),
        "class_name": predicted_class,
        "confidence": float(confidence.item()),
    }


def open_image_file(uploaded_file) -> Image.Image:
    """Decode an uploaded image and return a detached RGB copy."""
    try:
        image = Image.open(uploaded_file)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a supported, readable image.") from exc
