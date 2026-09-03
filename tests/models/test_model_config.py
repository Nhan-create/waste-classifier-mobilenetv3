from pathlib import Path

import yaml


def test_model_config_contains_approved_training_defaults() -> None:
    config = yaml.safe_load(
        Path("configs/model_config.yaml").read_text(encoding="utf-8")
    )

    assert config["model"] == {
        "name": "mobilenet_v3_large",
        "pretrained": True,
        "num_classes": 10,
    }
    assert config["input"]["size"] == 224
    assert config["input"]["mean"] == [0.485, 0.456, 0.406]
    assert config["input"]["std"] == [0.229, 0.224, 0.225]
    assert config["training"]["seed"] == 42
    assert config["training"]["batch_size"] == 32
    assert config["training"]["label_smoothing"] == 0.1
    assert config["training"]["weight_decay"] == 0.0001
    assert config["training"]["phase1"] == {"epochs": 5, "head_lr": 0.001}
    assert config["training"]["phase2"] == {
        "epochs": 25,
        "backbone_lr": 0.0001,
        "head_lr": 0.0003,
    }
    assert config["training"]["gradient_clip"] == 1.0
    assert config["training"]["early_stopping_patience"] == 7
    assert config["inference"]["confidence_threshold"] == 0.55
    assert config["checkpoint"]["format_version"] == 1
