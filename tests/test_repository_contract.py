from pathlib import Path

from src.data.schema import CLASS_NAMES


def test_documentation_matches_the_pinned_ten_class_system() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    preprocessing = Path("Tienxulidulieu.md").read_text(encoding="utf-8")
    documentation = readme + preprocessing

    for class_id in CLASS_NAMES:
        assert f"`{class_id}`" in readme
    assert "mrgetshjtdone/vn-trash-classification/versions/1" in documentation
    assert "sumn2u/garbage-classification-v2/versions/12" in documentation
    assert "SHA-256" in documentation
    assert "MD5" not in documentation
    assert "self_collected" not in documentation
    assert "repository không chứa dữ liệu hoặc checkpoint đã huấn luyện" in readme


def test_ci_runs_the_full_headless_quality_gate() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'python-version: "3.11"' in workflow
    assert "QT_QPA_PLATFORM: offscreen" in workflow
    assert "MPLBACKEND: Agg" in workflow
    assert "ruff check src tests app.py scripts" in workflow
    assert "python -m compileall src app.py scripts" in workflow
    assert "pytest -q" in workflow
    assert "kaggle" not in workflow.lower()


def test_generated_models_data_secrets_and_runtime_history_are_ignored() -> None:
    ignore = Path(".gitignore").read_text(encoding="utf-8")

    for pattern in (
        "/data/raw/",
        "/data/interim/",
        "/data/processed/",
        "/data/sources/",
        "/artifacts/",
        "*.pt",
        "*.pth",
        "*.sqlite3*",
        "kaggle.json",
        ".ipynb_checkpoints/",
        "*.zip",
    ):
        assert pattern in ignore
    assert "!/data/metadata/label_mapping.csv" in ignore
