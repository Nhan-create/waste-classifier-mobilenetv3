import pytest
import torch
from torch import nn

from src.models.mobilenetv3 import MODEL_ID, build_model


def test_factory_returns_ten_logits_without_softmax() -> None:
    model = build_model(MODEL_ID, num_classes=10, pretrained=False).eval()

    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))

    assert logits.shape == (2, 10)
    assert model.classifier[3].out_features == 10
    assert not any(isinstance(module, nn.Softmax) for module in model.modules())


def test_factory_rejects_unknown_model_and_invalid_class_count() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("mobilenet_v3_small", num_classes=10, pretrained=False)
    with pytest.raises(ValueError, match="num_classes"):
        build_model(MODEL_ID, num_classes=1, pretrained=False)
