from src.data.schema import CLASS_NAMES
from src.ui.presentation import DISPLAY_NAMES_VI, STYLE_BY_CLASS, present_label


def test_every_class_has_the_approved_vietnamese_name() -> None:
    assert DISPLAY_NAMES_VI == {
        "battery": "Pin",
        "biological": "Rác hữu cơ",
        "cardboard": "Bìa carton",
        "clothes": "Quần áo",
        "glass": "Thủy tinh",
        "metal": "Kim loại",
        "paper": "Giấy",
        "plastic": "Nhựa",
        "shoes": "Giày dép",
        "trash": "Rác khác",
    }
    assert set(DISPLAY_NAMES_VI) == set(CLASS_NAMES)
    assert set(STYLE_BY_CLASS) == set(CLASS_NAMES)


def test_known_class_has_display_metadata() -> None:
    result = present_label("glass")

    assert result.display_name == "Thủy tinh"
    assert result.icon
    assert result.color.startswith("#")


def test_unknown_class_uses_neutral_fallback() -> None:
    result = present_label("future_class")

    assert result.display_name == "Không xác định"
    assert result.icon == "?"
    assert result.color == "#64748B"
