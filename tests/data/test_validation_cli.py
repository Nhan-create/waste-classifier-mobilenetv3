import subprocess
import sys
from pathlib import Path


def test_validation_cli_fails_with_missing_manifest_path(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "missing.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.data.validation",
            "--dataset-root",
            str(tmp_path / "processed"),
            "--manifest",
            str(missing_manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert str(missing_manifest) in completed.stdout
