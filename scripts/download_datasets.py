"""Download the two pinned Kaggle dataset versions through KaggleHub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import kagglehub
import yaml

PINNED_DATASETS: dict[str, str] = {
    "vn_trash": "mrgetshjtdone/vn-trash-classification/versions/1",
    "garbage_v2": "sumn2u/garbage-classification-v2/versions/12",
}


def _layout_candidates(root: Path) -> list[Path]:
    if not root.is_dir():
        return [root]
    nested = [path for path in root.rglob("*") if path.is_dir()]
    return sorted([root, *nested], key=lambda path: (len(path.parts), path.as_posix()))


def _resolve_layout_root(source_name: str, downloaded_root: Path) -> Path:
    candidates = _layout_candidates(downloaded_root)
    if source_name == "vn_trash":
        match = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "Train").is_dir()
                and (candidate / "Test").is_dir()
            ),
            None,
        )
    else:
        match = next(
            (candidate for candidate in candidates if (candidate / "original").is_dir()),
            None,
        )
    return match or downloaded_root


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
        roots[name] = _resolve_layout_root(name, Path(downloaded))
    return roots


def write_resolved_preprocessing_config(
    base_config_path: Path,
    destination: Path,
    roots: dict[str, Path],
) -> Path:
    """Write a local config without changing the committed source contract."""

    try:
        config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
        sources = config["dataset"]["sources"]
    except (FileNotFoundError, KeyError, TypeError) as error:
        raise RuntimeError(f"Invalid preprocessing config: {base_config_path}") from error
    if set(roots) != set(PINNED_DATASETS):
        raise RuntimeError(
            f"Resolved roots must be {sorted(PINNED_DATASETS)}, received {sorted(roots)}"
        )
    for name, root in roots.items():
        sources[name]["root"] = str(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/sources"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/preprocessing_config.yaml"),
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("data/sources/preprocessing_config.local.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        roots = download_pinned_datasets(arguments.output_root)
        config_path = write_resolved_preprocessing_config(
            arguments.base_config,
            arguments.config_output,
            roots,
        )
    except RuntimeError as error:
        print(str(error))
        return 2
    print(
        json.dumps(
            {
                "sources": {name: str(path) for name, path in roots.items()},
                "config": str(config_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
