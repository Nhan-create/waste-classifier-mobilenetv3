"""The single MobileNetV3 architecture definition used by all consumers."""

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

MODEL_ID = "mobilenet_v3_large"


def build_model(
    model_name: str = MODEL_ID,
    num_classes: int = 10,
    pretrained: bool = True,
) -> nn.Module:
    """Build MobileNetV3-Large and replace its final classifier with logits."""

    if model_name != MODEL_ID:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected {MODEL_ID!r}"
        )
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, received {num_classes}")
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    final_layer = model.classifier[3]
    model.classifier[3] = nn.Linear(final_layer.in_features, num_classes)
    return model
