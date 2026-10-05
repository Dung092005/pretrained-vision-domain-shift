"""Optional checkpoint-backed inference helpers for the Streamlit demo.

This module deliberately imports PyTorch only when an actual checkpoint is
loaded. That keeps the archived-results dashboard lightweight on Streamlit
Community Cloud when no model files are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
SKIPPED_CHECKPOINT_DIRECTORIES = {".git", ".venv", "venv", "officehome_data", "officehome_download_cache"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be used for Office-Home inference."""


@dataclass(frozen=True)
class CheckpointCandidate:
    """A checkpoint discovered under the project directory."""

    path: Path
    model_name: str | None
    mode: str | None

    @property
    def label(self) -> str:
        details = []
        if self.model_name:
            details.append(self.model_name)
        if self.mode:
            details.append(self.mode)
        descriptor = " / ".join(details) if details else "unidentified configuration"
        return f"{self.path.name} — {descriptor}"


def _is_within_skipped_directory(path: Path) -> bool:
    return any(part.lower() in SKIPPED_CHECKPOINT_DIRECTORIES for part in path.parts)


def _infer_metadata(path: Path) -> tuple[str | None, str | None]:
    text = path.as_posix().lower()
    model_name = "resnet50" if "resnet50" in text else "resnet18" if "resnet18" in text else None
    mode = "finetune_last_block" if "finetune_last_block" in text else "linear_probe" if "linear_probe" in text else None
    return model_name, mode


def discover_checkpoints(project_root: Path) -> list[CheckpointCandidate]:
    """Return all small, local checkpoint candidates without downloading anything."""
    candidates: list[CheckpointCandidate] = []
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        if _is_within_skipped_directory(path.relative_to(project_root)):
            continue
        model_name, mode = _infer_metadata(path)
        candidates.append(CheckpointCandidate(path=path, model_name=model_name, mode=mode))
    return sorted(candidates, key=lambda item: item.path.as_posix().lower())


def compatible_candidates(
    candidates: Iterable[CheckpointCandidate], model_name: str, mode: str
) -> list[CheckpointCandidate]:
    """Filter checkpoints that identify the currently selected experiment configuration."""
    return [
        candidate
        for candidate in candidates
        if candidate.model_name == model_name and candidate.mode == mode
    ]


def _load_torch_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torchvision import models
    except ModuleNotFoundError as error:
        raise CheckpointError(
            "PyTorch and torchvision are required for checkpoint inference. "
            "Install compatible CPU packages in demo/requirements.txt before deploying inference."
        ) from error
    return torch, nn, models


def _extract_state_dict(payload: Any) -> tuple[dict[str, Any], list[str] | None]:
    if isinstance(payload, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                stored_classes = payload.get("classes")
                return state_dict, list(stored_classes) if isinstance(stored_classes, list) else None
        if all(isinstance(key, str) for key in payload):
            return payload, None
    raise CheckpointError("The checkpoint does not contain a recognizable model state dictionary.")


def load_model(
    checkpoint_path: str,
    model_name: str,
    class_names: list[str],
) -> tuple[Any, Any]:
    """Load a CPU ResNet checkpoint matching the archived Office-Home setup."""
    torch, nn, models = _load_torch_dependencies()
    constructors = {"resnet18": models.resnet18, "resnet50": models.resnet50}
    if model_name not in constructors:
        raise CheckpointError(f"Unsupported architecture: {model_name}")

    model = constructors[model_name](weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - only exercised with user-supplied files
        raise CheckpointError(f"Could not read checkpoint: {error}") from error

    state_dict, stored_classes = _extract_state_dict(payload)
    if stored_classes is not None and stored_classes != class_names:
        raise CheckpointError("Checkpoint class order does not match results/class_to_idx.json.")

    cleaned_state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except RuntimeError as error:
        raise CheckpointError(
            "Checkpoint is incompatible with the selected ResNet architecture or 65-class classifier."
        ) from error

    model.eval()
    return model, torch


def predict_image(model: Any, torch: Any, image: Any, class_names: list[str]) -> list[dict[str, float | str]]:
    """Apply the saved evaluation preprocessing and return the top five classes."""
    try:
        from torchvision import transforms
    except ModuleNotFoundError as error:
        raise CheckpointError("torchvision is required for the archived evaluation preprocessing.") from error

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(5, len(class_names)))

    return [
        {"class": class_names[int(index)], "probability": float(value)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]
