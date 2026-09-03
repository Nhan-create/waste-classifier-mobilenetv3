from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.data.schema import CLASS_NAMES
from src.preprocessing.run_pipeline import run_pipeline


def test_pipeline_builds_valid_ten_class_dataset(tmp_path: Path) -> None:
    source_root = tmp_path / "garbage"
    sequence = 0
    for class_name in CLASS_NAMES:
        for index in range(3):
            path = source_root / "original" / class_name / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.random.default_rng(sequence).integers(
                0,
                256,
                (32, 32, 3),
                dtype=np.uint8,
            )
            sequence += 1
            Image.fromarray(pixels, mode="RGB").save(path)

    mapping_path = Path("data/metadata/label_mapping.csv").resolve()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "image_size": 224,
                "batch_size": 32,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_workers": 0,
                "dataset": {
                    "version": "v1",
                    "mapping_path": str(mapping_path),
                    "sources": {
                        "garbage_v2": {
                            "root": str(source_root),
                            "layout": "garbage_v2",
                            "slug": "sumn2u/garbage-classification-v2",
                            "version": 12,
                        }
                    },
                    "split": {
                        "train": 0.70,
                        "val": 0.15,
                        "test": 0.15,
                        "seed": 42,
                    },
                    "duplicates": {"phash_hamming_threshold": 4},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = run_pipeline(
        config_path=config_path,
        raw_root=tmp_path / "raw",
        metadata_root=tmp_path / "metadata" / "v1",
        processed_root=tmp_path / "processed" / "v1",
        report_root=tmp_path / "outputs" / "v1",
    )

    assert result.validation.is_valid
    assert len(result.fingerprint) == 64
    assert result.class_weights.is_file()
    assert result.eda_json.is_file()
    for split_name in ("train", "val", "test"):
        assert {
            path.name
            for path in (result.processed_root / split_name).iterdir()
            if path.is_dir()
        } == set(CLASS_NAMES)
