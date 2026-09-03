import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.data.schema",
        "src.data.validation",
        "src.models.mobilenetv3",
        "src.training.train",
        "src.evaluation.evaluate",
        "src.inference.predict",
        "src.ui.main_window",
        "scripts.download_datasets",
    ],
)
def test_public_modules_import(module: str) -> None:
    importlib.import_module(module)
