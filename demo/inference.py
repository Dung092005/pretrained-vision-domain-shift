"""Validated, CPU-only inference helpers for the Office-Home Streamlit demo.

The project keeps the four trained checkpoints in the public supplementary
archive instead of Git. Checkpoints are fetched one at a time with an HTTP
range request, verified by SHA-256, and released after inference. Importing
this module itself does not import PyTorch or download anything.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DOMAIN_ORDER = ("Real_World", "Product", "Art", "Clipart")
ProgressCallback = Callable[[int, int], None]


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be downloaded, validated, or loaded."""


class SampleError(RuntimeError):
    """Raised when a curated Office-Home sample cannot be resolved safely."""


@dataclass(frozen=True)
class CheckpointSpec:
    """Identity and verified byte-range location of one selected best checkpoint."""

    key: str
    architecture: str
    mode: str
    source: str
    seed: int
    best_epoch: int
    config_hash: str
    url: str
    archive_member: str
    range_start: int
    size_bytes: int
    sha256: str

    @property
    def range_end(self) -> int:
        return self.range_start + self.size_bytes - 1

    @property
    def label(self) -> str:
        architecture = "ResNet18" if self.architecture == "resnet18" else "ResNet50"
        mode = "Linear Probe" if self.mode == "linear_probe" else "Fine-tune Last Block"
        return f"{architecture} — {mode}"


@dataclass(frozen=True)
class SampleSpec:
    """A reproducible image sample resolved from the public Office-Home mirror."""

    class_name: str
    label: int
    domain: str
    relative_path: str
    row_index: int
    split: str
    dataset: str
    dataset_revision: str
    config: str
    source_split: str
    rows_endpoint: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON file {path.name}: {error}") from error


def load_checkpoint_specs(manifest_path: Path) -> dict[str, CheckpointSpec]:
    """Load the committed checkpoint manifest without downloading model bytes."""
    payload = _read_json(manifest_path)
    records = payload.get("checkpoints") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        raise ValueError("Checkpoint manifest must contain a 'checkpoints' object.")

    required = {
        "architecture",
        "mode",
        "source",
        "seed",
        "best_epoch",
        "config_hash",
        "url",
        "archive_member",
        "range_start",
        "size_bytes",
        "sha256",
    }
    specs: dict[str, CheckpointSpec] = {}
    for key, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"Checkpoint '{key}' is not an object.")
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Checkpoint '{key}' is missing: {', '.join(sorted(missing))}")
        spec = CheckpointSpec(key=str(key), **{field: record[field] for field in required})
        if spec.architecture not in {"resnet18", "resnet50"}:
            raise ValueError(f"Checkpoint '{key}' has an unsupported architecture.")
        if spec.mode not in {"linear_probe", "finetune_last_block"}:
            raise ValueError(f"Checkpoint '{key}' has an unsupported training mode.")
        if not spec.url.startswith("https://"):
            raise ValueError(f"Checkpoint '{key}' must use an HTTPS URL.")
        if spec.range_start < 0 or spec.size_bytes <= 0:
            raise ValueError(f"Checkpoint '{key}' has an invalid byte range.")
        if len(spec.sha256) != 64 or any(char not in "0123456789abcdef" for char in spec.sha256.lower()):
            raise ValueError(f"Checkpoint '{key}' has an invalid SHA-256 value.")
        specs[spec.key] = spec

    expected = {
        "resnet18_linear_probe",
        "resnet18_finetune_last_block",
        "resnet50_linear_probe",
        "resnet50_finetune_last_block",
    }
    if set(specs) != expected:
        raise ValueError("Checkpoint manifest must describe exactly the four archived configurations.")
    return specs


def _normalise_domain(value: str) -> str:
    key = "".join(character for character in str(value).lower() if character.isalnum())
    domains = {
        "realworld": "Real_World",
        "product": "Product",
        "art": "Art",
        "clipart": "Clipart",
    }
    try:
        return domains[key]
    except KeyError as error:
        raise SampleError(f"Unexpected Office-Home domain returned by the mirror: {value!r}") from error


def load_sample_specs(manifest_path: Path) -> dict[str, dict[str, SampleSpec]]:
    """Load the small curated sample manifest; no dataset image is loaded yet."""
    payload = _read_json(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Sample manifest must be a JSON object.")
    source = payload.get("source")
    classes = payload.get("classes")
    if not isinstance(source, dict) or not isinstance(classes, dict):
        raise ValueError("Sample manifest must contain 'source' and 'classes' objects.")
    source_fields = {"dataset", "dataset_revision", "config", "split", "rows_endpoint"}
    missing_source = source_fields.difference(source)
    if missing_source:
        raise ValueError(f"Sample manifest source is missing: {', '.join(sorted(missing_source))}")

    samples: dict[str, dict[str, SampleSpec]] = {}
    for class_name, records in classes.items():
        if not isinstance(records, list):
            raise ValueError(f"Samples for {class_name} must be a list.")
        by_domain: dict[str, SampleSpec] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"A sample for {class_name} is not an object.")
            required = {"label", "domain", "relative_path", "row_index", "split"}
            missing = required.difference(record)
            if missing:
                raise ValueError(f"Sample {class_name} is missing: {', '.join(sorted(missing))}")
            domain = _normalise_domain(str(record["domain"]))
            spec = SampleSpec(
                class_name=str(class_name),
                label=int(record["label"]),
                domain=domain,
                relative_path=str(record["relative_path"]),
                row_index=int(record["row_index"]),
                split=str(record["split"]),
                dataset=str(source["dataset"]),
                dataset_revision=str(source["dataset_revision"]),
                config=str(source["config"]),
                source_split=str(source["split"]),
                rows_endpoint=str(source["rows_endpoint"]),
            )
            if spec.row_index < 0 or spec.label < 0:
                raise ValueError(f"Sample {class_name}/{domain} has invalid metadata.")
            if domain in by_domain:
                raise ValueError(f"Sample manifest has duplicate {class_name}/{domain} entries.")
            by_domain[domain] = spec
        if set(by_domain) != set(DOMAIN_ORDER):
            raise ValueError(f"Samples for {class_name} must include all four domains.")
        samples[str(class_name)] = by_domain
    return samples


def default_cache_dir() -> Path:
    """Return a writable, non-repository cache that Community Cloud can discard."""
    override = os.environ.get("OFFICEHOME_DEMO_CACHE_DIR")
    root = Path(override) if override else Path(tempfile.gettempdir()) / "officehome-domain-shift-demo"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit_progress(callback: ProgressCallback | None, completed: int, total: int) -> None:
    if callback is not None:
        callback(completed, total)


def _valid_checkpoint_cache(path: Path, spec: CheckpointSpec) -> bool:
    return path.is_file() and path.stat().st_size == spec.size_bytes and _hash_file(path) == spec.sha256


def _request_range(spec: CheckpointSpec, temporary_path: Path, progress_callback: ProgressCallback | None) -> None:
    headers = {
        "Range": f"bytes={spec.range_start}-{spec.range_end}",
        "User-Agent": "OfficeHome-Streamlit-Demo/1.0",
    }
    request = Request(spec.url, headers=headers)
    with urlopen(request, timeout=90) as response, temporary_path.open("wb") as output:
        status = response.getcode()
        content_range = response.headers.get("Content-Range", "")
        expected_range = f"bytes {spec.range_start}-{spec.range_end}/"
        if status != 206 or not content_range.startswith(expected_range):
            raise CheckpointError(
                "The checkpoint host did not return the expected byte range; "
                "the archive may have changed."
            )
        downloaded = 0
        _emit_progress(progress_callback, downloaded, spec.size_bytes)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            _emit_progress(progress_callback, downloaded, spec.size_bytes)
    if downloaded != spec.size_bytes:
        raise CheckpointError(
            f"Checkpoint download was incomplete ({downloaded:,} of {spec.size_bytes:,} bytes)."
        )


def ensure_checkpoint(
    spec: CheckpointSpec,
    cache_dir: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Return a SHA-256 validated local checkpoint, downloading only this configuration.

    The supplementary Google Drive item is a stored ZIP archive. ``range_start``
    points at the first byte of the uncompressed ``best.pt`` member, allowing a
    direct range download rather than downloading the whole archive.
    """
    base = (cache_dir or default_cache_dir()) / "checkpoints"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{spec.key}.pt"
    if _valid_checkpoint_cache(target, spec):
        _emit_progress(progress_callback, spec.size_bytes, spec.size_bytes)
        return target
    if target.exists():
        target.unlink()

    failure: Exception | None = None
    for attempt in range(1, 4):
        temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.part")
        try:
            _request_range(spec, temporary, progress_callback)
            if _hash_file(temporary) != spec.sha256:
                raise CheckpointError("Checkpoint SHA-256 does not match the committed manifest.")
            os.replace(temporary, target)
            return target
        except (CheckpointError, HTTPError, URLError, OSError) as error:
            failure = error
            temporary.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt)
    raise CheckpointError(f"Could not prepare {spec.label}: {failure}") from failure


def _valid_image_cache(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _get_json(url: str, params: Mapping[str, str | int]) -> dict[str, Any]:
    query = urlencode(params)
    request = Request(f"{url}?{query}", headers={"User-Agent": "OfficeHome-Streamlit-Demo/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as error:
        raise SampleError(f"Could not resolve the curated sample from the public mirror: {error}") from error


def _download_sample_image(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "OfficeHome-Streamlit-Demo/1.0"})
    try:
        with urlopen(request, timeout=90) as response, destination.open("wb") as output:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except (HTTPError, URLError, OSError) as error:
        raise SampleError(f"Could not download the selected sample image: {error}") from error


def ensure_sample(sample: SampleSpec, cache_dir: Path | None = None) -> Path:
    """Resolve one curated image and verify its label/domain before displaying it."""
    base = (cache_dir or default_cache_dir()) / "samples"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"{sample.row_index:05d}_{sample.domain}_{sample.class_name}.jpg"
    if _valid_image_cache(target):
        return target
    if target.exists():
        target.unlink()

    payload = _get_json(
        sample.rows_endpoint,
        {
            "dataset": sample.dataset,
            "config": sample.config,
            "split": sample.source_split,
            "offset": sample.row_index,
            "length": 1,
        },
    )
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 1:
        raise SampleError("The public mirror did not return the expected sample row.")
    row = rows[0]
    row_data = row.get("row") if isinstance(row, dict) else None
    if not isinstance(row_data, dict) or int(row.get("row_idx", -1)) != sample.row_index:
        raise SampleError("The public mirror returned a different sample row.")
    if int(row_data.get("label", -1)) != sample.label:
        raise SampleError("The public mirror returned a sample with a different class label.")
    if _normalise_domain(str(row_data.get("domain", ""))) != sample.domain:
        raise SampleError("The public mirror returned a sample from a different domain.")
    image = row_data.get("image")
    image_url = image.get("src") if isinstance(image, dict) else None
    if not isinstance(image_url, str) or not image_url.startswith("https://"):
        raise SampleError("The public mirror did not provide an image URL for this sample.")

    temporary = target.with_name(f"{target.name}.{uuid.uuid4().hex}.part")
    try:
        _download_sample_image(image_url, temporary)
        if not _valid_image_cache(temporary):
            raise SampleError("The downloaded sample is not a readable image.")
        os.replace(temporary, target)
        return target
    except (SampleError, OSError) as error:
        temporary.unlink(missing_ok=True)
        raise SampleError(str(error)) from error


def load_image(path: Path) -> Any:
    """Open a cached sample as a detached RGB Pillow image."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except (OSError, ValueError) as error:
        raise SampleError(f"Could not open sample image {path.name}: {error}") from error


def _load_torch_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torchvision import models
    except ModuleNotFoundError as error:
        raise CheckpointError(
            "PyTorch and torchvision are required for inference. "
            "Install the CPU wheels listed in demo/requirements.txt."
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


def _validate_checkpoint_metadata(payload: Any, spec: CheckpointSpec, class_names: Sequence[str]) -> None:
    if not isinstance(payload, dict):
        return
    expected = {
        "source": spec.source,
        "model_name": spec.architecture,
        "mode": spec.mode,
        "seed": spec.seed,
        "best_epoch": spec.best_epoch,
        "config_hash": spec.config_hash,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise CheckpointError(
                f"Checkpoint metadata mismatch for {key}: expected {value!r}, got {payload.get(key)!r}."
            )
    stored_classes = payload.get("classes")
    if not isinstance(stored_classes, list) or stored_classes != list(class_names):
        raise CheckpointError("Checkpoint class order does not match results/class_to_idx.json.")


def load_model(checkpoint_path: Path, spec: CheckpointSpec, class_names: Sequence[str]) -> tuple[Any, Any]:
    """Construct exactly one 65-class ResNet and load one validated best checkpoint."""
    if len(class_names) != 65:
        raise CheckpointError("Expected the archived 65-class Office-Home label mapping.")
    torch, nn, models = _load_torch_dependencies()
    constructors = {"resnet18": models.resnet18, "resnet50": models.resnet50}
    model = constructors[spec.architecture](weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - exercised only with external files
        raise CheckpointError(f"Could not read checkpoint {checkpoint_path.name}: {error}") from error
    _validate_checkpoint_metadata(payload, spec, class_names)
    state_dict, stored_classes = _extract_state_dict(payload)
    if stored_classes is not None and stored_classes != list(class_names):
        raise CheckpointError("Checkpoint class order does not match results/class_to_idx.json.")

    cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
    classifier = cleaned.get("fc.weight")
    if classifier is None or tuple(classifier.shape)[0] != len(class_names):
        raise CheckpointError("Checkpoint classifier does not have 65 outputs.")
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as error:
        raise CheckpointError(
            "Checkpoint is incompatible with the selected ResNet architecture or classifier."
        ) from error
    model.eval()
    return model, torch


def predict_image(model: Any, torch: Any, image: Any, class_names: Sequence[str]) -> list[dict[str, float | str]]:
    """Use the archived deterministic test transform and return top-5 probabilities."""
    try:
        from torchvision import transforms
    except ModuleNotFoundError as error:
        raise CheckpointError("torchvision is required for the archived test preprocessing.") from error

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
        {"class": str(class_names[int(index)]), "probability": float(value)}
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def predict_images_for_configuration(
    spec: CheckpointSpec,
    images: Mapping[str, Any],
    class_names: Sequence[str],
    cache_dir: Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, list[dict[str, float | str]]]:
    """Predict one or more images with one model, then release that model from RAM."""
    model: Any | None = None
    torch: Any | None = None
    try:
        checkpoint_path = ensure_checkpoint(spec, cache_dir=cache_dir, progress_callback=progress_callback)
        model, torch = load_model(checkpoint_path, spec, class_names)
        return {name: predict_image(model, torch, image, class_names) for name, image in images.items()}
    finally:
        if model is not None:
            del model
        if torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
            torch.cuda.empty_cache()
        gc.collect()
        try:
            import ctypes

            libc = ctypes.CDLL(None)
            trim = getattr(libc, "malloc_trim", None)
            if trim is not None:
                trim(0)
        except (AttributeError, OSError, TypeError):
            pass
