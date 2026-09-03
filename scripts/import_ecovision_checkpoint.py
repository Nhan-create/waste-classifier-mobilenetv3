"""Import the pinned MIT EcoVision weights into this project's checkpoint format."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import torch

from src.data.schema import CLASS_NAMES
from src.models.mobilenetv3 import MODEL_ID, build_model
from src.training.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint

SOURCE_MODEL = "AmadFR/ecovision_mobilenetv3"
SOURCE_REVISION = "7c2daeea3f684058ae8a1c9656c50fc7309fc36c"
SOURCE_SHA256 = "0d60ec3944396b9e989a5e082ad848d6820a92e461230d620d7a53e39ccd52a1"
SOURCE_DATASET = "sumn2u/garbage-classification-v2"
SOURCE_URL = (
    "https://huggingface.co/AmadFR/ecovision_mobilenetv3/resolve/"
    f"{SOURCE_REVISION}/ecovision_mobilenetv3_ST.pth?download=true"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(destination: Path) -> Path:
    """Download the pinned source weights without loading them as executable code."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = Request(SOURCE_URL, headers={"User-Agent": "waste-classifier/1.0"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(destination)
    except (OSError, URLError) as error:
        raise RuntimeError(f"Could not download pinned model: {error}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def import_checkpoint(source: Path, output: Path) -> Path:
    """Validate a pinned state dict and wrap it in the local metadata contract."""

    actual_sha256 = _sha256(source)
    if actual_sha256 != SOURCE_SHA256:
        raise RuntimeError(
            "Source checkpoint checksum mismatch: "
            f"expected {SOURCE_SHA256}, received {actual_sha256}"
        )

    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise RuntimeError("Source checkpoint is not a tensor-only state dictionary")

    model = build_model(MODEL_ID, len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(state, strict=True)
    dataset_fingerprint = hashlib.sha256(SOURCE_DATASET.encode("utf-8")).hexdigest()
    metadata = CheckpointMetadata(
        format_version=1,
        model_name=MODEL_ID,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        input_size=224,
        normalization={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        epoch=10,
        metrics={},
        dataset_fingerprint=dataset_fingerprint,
        training_config={
            "origin": "third_party_import",
            "source_model": SOURCE_MODEL,
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "source_license": "MIT",
            "reported_training_dataset": SOURCE_DATASET,
        },
    )
    save_checkpoint(output, metadata, model.state_dict())
    load_checkpoint(output, torch.device("cpu"), expected_classes=CLASS_NAMES)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Optional already-downloaded source .pth; otherwise download the pin.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ecovision/best.pt"),
    )
    arguments = parser.parse_args()
    downloaded_source: Path | None = None
    try:
        if arguments.output.is_file():
            load_checkpoint(
                arguments.output,
                torch.device("cpu"),
                expected_classes=CLASS_NAMES,
            )
            output = arguments.output
        else:
            source = arguments.source
            if source is None:
                downloaded_source = arguments.output.with_name(
                    "ecovision_mobilenetv3_ST.pth"
                )
                source = download_source(downloaded_source)
            output = import_checkpoint(source, arguments.output)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Import failed: {error}")
        return 2
    finally:
        if downloaded_source is not None and downloaded_source.exists():
            downloaded_source.unlink()
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "source_model": SOURCE_MODEL,
                "source_revision": SOURCE_REVISION,
                "class_names": CLASS_NAMES,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
