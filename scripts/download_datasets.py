"""Download the two pinned Kaggle dataset versions through KaggleHub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import kagglehub

PINNED_DATASETS: dict[str, str] = {
    "vn_trash": "mrgetshjtdone/vn-trash-classification/versions/1",
    "garbage_v2": "sumn2u/garbage-classification-v2/versions/12",
}


def download_pinned_datasets(output_root: Path) -> dict[str, Path]:
    """Download sources without reading, printing, or persisting credentials."""

    output_root.mkdir(parents=True, exist_ok=True)
    roots: dict[str, Path] = {}
    for name, handle in PINNED_DATASETS.items():
        destination = output_root / name
        destination.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = kagglehub.dataset_download(handle, output_dir=destination)
        except Exception as error:
            raise RuntimeError(
                f"Could not download {handle}; configure KaggleHub "
                "authentication outside the repository"
            ) from error
        roots[name] = Path(downloaded)
    return roots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/sources"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        roots = download_pinned_datasets(arguments.output_root)
    except RuntimeError as error:
        print(str(error))
        return 2
    print(json.dumps({name: str(path) for name, path in roots.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
