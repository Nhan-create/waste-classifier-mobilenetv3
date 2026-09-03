import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mobilenetv3 import MODEL_ID, build_model
from src.training.engine import configure_phase, run_epoch


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16), nn.ReLU())
        self.classifier = nn.Linear(16, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def test_phase_one_freezes_features_and_phase_two_unfreezes() -> None:
    model = build_model(MODEL_ID, 10, pretrained=False)

    configure_phase(model, phase=1)
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())

    configure_phase(model, phase=2)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_one_cpu_optimizer_step_returns_finite_metrics() -> None:
    torch.manual_seed(3)
    model = TinyClassifier()
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 3])),
        batch_size=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    result = run_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        optimizer=optimizer,
        device=torch.device("cpu"),
        scaler=None,
        gradient_clip=1.0,
        batch_size=4,
    )

    assert math.isfinite(result.loss)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
