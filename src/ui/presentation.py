"""Pure display metadata for canonical waste class identifiers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelPresentation:
    display_name: str
    icon: str
    color: str


DISPLAY_NAMES_VI: dict[str, str] = {
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

STYLE_BY_CLASS: dict[str, tuple[str, str]] = {
    "battery": ("🔋", "#B45309"),
    "biological": ("🌿", "#15803D"),
    "cardboard": ("▤", "#92400E"),
    "clothes": ("♙", "#7E22CE"),
    "glass": ("◇", "#0369A1"),
    "metal": ("⬡", "#475569"),
    "paper": ("▱", "#2563EB"),
    "plastic": ("♻", "#0F766E"),
    "shoes": ("◒", "#9D174D"),
    "trash": ("▧", "#52525B"),
}

UNKNOWN_PRESENTATION = LabelPresentation("Không xác định", "?", "#64748B")


def present_label(class_id: str) -> LabelPresentation:
    """Return user-facing metadata without changing the stable class ID."""

    display_name = DISPLAY_NAMES_VI.get(class_id)
    style = STYLE_BY_CLASS.get(class_id)
    if display_name is None or style is None:
        return UNKNOWN_PRESENTATION
    icon, color = style
    return LabelPresentation(display_name, icon, color)
