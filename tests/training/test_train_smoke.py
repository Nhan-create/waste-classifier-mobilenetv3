from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.schema import CLASS_NAMES
from src.models.mobilenetv3 import MODEL_ID, build_model
from src.training.checkpoint import CheckpointMetadata, load_checkpoint
from src.training.train import fit


def tiny_config(phase2_epochs: int = 1) -> dict:
    return {
        "training": {
            "batch_size": 2,
            "label_smoothing": 0.1,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "early_stopping_patience": 7,
            "phase1": {"epochs": 1, "head_lr": 0.001},
            "phase2": {
                "epochs": phase2_epochs,
                "backbone_lr": 0.0001,
                "head_lr": 0.0003,
            },
        }
    }


def checkpoint_metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        format_version=1,
        model_name=MODEL_ID,
        num_classes=10,
        class_names=CLASS_NAMES,
        input_size=32,
        normalization={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        epoch=0,
        metrics={},
        dataset_fingerprint="b" * 64,
        training_config={"seed": 42},
    )


def synthetic_loaders() -> dict[str, DataLoader]:
    torch.manual_seed(11)
    dataset = TensorDataset(
        torch.randn(2, 3, 32, 32),
        torch.tensor([0, 1]),
    )
    return {
        "train": DataLoader(dataset, batch_size=2),
        "val": DataLoader(dataset, batch_size=2),
    }


def test_fit_writes_best_and_resumable_last_checkpoint(tmp_path: Path) -> None:
    model = build_model(MODEL_ID, 10, pretrained=False)

    result = fit(
        model=model,
        loaders=synthetic_loaders(),
        class_weights=torch.ones(10),
        config=tiny_config(),
        base_metadata=checkpoint_metadata(),
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert result.best_path.is_file()
    assert result.last_path.is_file()
    assert len(result.history) == 2
    best = load_checkpoint(result.best_path, torch.device("cpu"))
    last = load_checkpoint(result.last_path, torch.device("cpu"))
    assert best.resume_state is None
    assert best.metadata.metrics["val_macro_f1"] == max(
        row["val_macro_f1"] for row in result.history
    )
    assert {
        "optimizer",
        "scheduler",
        "scaler",
        "next_epoch",
        "phase",
        "global_epoch",
        "best_macro_f1",
        "patience_counter",
        "history",
    } <= set(last.resume_state)


def test_resume_continues_at_saved_phase_epoch(tmp_path: Path) -> None:
    first = fit(
        model=build_model(MODEL_ID, 10, pretrained=False),
        loaders=synthetic_loaders(),
        class_weights=torch.ones(10),
        config=tiny_config(phase2_epochs=1),
        base_metadata=checkpoint_metadata(),
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    resumed = fit(
        model=build_model(MODEL_ID, 10, pretrained=False),
        loaders=synthetic_loaders(),
        class_weights=torch.ones(10),
        config=tiny_config(phase2_epochs=2),
        base_metadata=checkpoint_metadata(),
        output_dir=tmp_path,
        device=torch.device("cpu"),
        resume_path=first.last_path,
    )

    last = load_checkpoint(resumed.last_path, torch.device("cpu"))
    assert len(resumed.history) == 3
    assert last.resume_state["phase"] == 2
    assert last.resume_state["next_epoch"] == 2
    assert last.resume_state["global_epoch"] == 3
