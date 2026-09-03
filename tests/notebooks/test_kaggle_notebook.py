from pathlib import Path

import nbformat

NOTEBOOK_PATH = Path("notebooks/train_mobilenetv3_kaggle.ipynb")


def read_source() -> tuple[nbformat.NotebookNode, str]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    return notebook, source


def test_kaggle_notebook_has_required_order_and_no_credentials() -> None:
    notebook, source = read_source()
    markers = [
        "run_pipeline",
        "validation.is_valid",
        "src.training.train",
        "src.evaluation.evaluate",
        "waste-classifier-output.zip",
    ]
    positions = [source.index(marker) for marker in markers]

    assert positions == sorted(positions)
    assert "kaggle.json" not in source
    assert "KAGGLE_KEY" not in source
    assert "KAGGLE_USERNAME" not in source
    assert notebook.metadata.kernelspec.name == "python3"


def test_kaggle_notebook_pins_sources_and_packages_auditable_outputs() -> None:
    notebook, source = read_source()

    assert "mrgetshjtdone/vn-trash-classification/versions/1" in source
    assert "sumn2u/garbage-classification-v2/versions/12" in source
    for artifact in (
        "best.pt",
        "metrics.json",
        "per_class_metrics.csv",
        "confusion_matrix_raw.png",
        "confusion_matrix_normalized.png",
        "split_manifest.csv",
        "label_mapping.csv",
        "resolved_config.yaml",
    ):
        assert artifact in source
    assert all(cell.execution_count is None for cell in notebook.cells if cell.cell_type == "code")
    assert all(not cell.outputs for cell in notebook.cells if cell.cell_type == "code")
