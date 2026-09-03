from pathlib import Path

import pytest

from src.data.schema import (
    CLASS_NAMES,
    LabelMappingError,
    canonicalize_label,
    load_label_mapping,
    map_to_unified_label,
)

MAPPING_PATH = Path("data/metadata/label_mapping.csv")


def test_canonical_class_order_is_fixed() -> None:
    assert CLASS_NAMES == (
        "battery",
        "biological",
        "cardboard",
        "clothes",
        "glass",
        "metal",
        "paper",
        "plastic",
        "shoes",
        "trash",
    )


def test_vn_mapping_covers_all_source_labels() -> None:
    mapping = load_label_mapping(MAPPING_PATH)
    expected = {
        "Alu": "metal",
        "Carton": "cardboard",
        "Foam_box": "plastic",
        "Milk_box": "cardboard",
        "Other": "trash",
        "PET": "plastic",
        "Paper": "paper",
        "Paper_cup": "paper",
        "Plastic_cup": "plastic",
    }

    actual = {
        label: map_to_unified_label("vn_trash", label, mapping)
        for label in expected
    }

    assert actual == expected


def test_garbage_v2_accepts_only_canonical_labels() -> None:
    mapping = load_label_mapping(MAPPING_PATH)

    assert map_to_unified_label("garbage_v2", "GLASS", mapping) == "glass"
    with pytest.raises(LabelMappingError, match="food waste"):
        map_to_unified_label("garbage_v2", "food waste", mapping)


def test_unknown_vn_label_is_rejected() -> None:
    mapping = load_label_mapping(MAPPING_PATH)

    with pytest.raises(LabelMappingError, match="Mystery"):
        map_to_unified_label("vn_trash", "Mystery", mapping)


@pytest.mark.parametrize(
    ("source", "expected"),
    [(" Milk box ", "milk_box"), ("Foam-box", "foam_box"), ("PAPER", "paper")],
)
def test_label_normalization_is_stable(source: str, expected: str) -> None:
    assert canonicalize_label(source) == expected
