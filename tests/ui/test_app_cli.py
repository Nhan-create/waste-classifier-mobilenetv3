import subprocess
import sys
from pathlib import Path


def test_app_help_lists_runtime_options() -> None:
    completed = subprocess.run(
        [sys.executable, "app.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--history-db" in completed.stdout
    assert "--camera-index" in completed.stdout


def test_missing_checkpoint_fails_before_qt_is_started(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"

    completed = subprocess.run(
        [sys.executable, "app.py", "--checkpoint", str(missing)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert str(missing) in completed.stderr
