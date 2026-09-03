import subprocess
import sys


def test_train_cli_help_lists_data_resume_and_device_controls() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.training.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--data-root" in completed.stdout
    assert "--manifest" in completed.stdout
    assert "--resume" in completed.stdout
    assert "--batch-size" in completed.stdout
    assert "--deterministic" in completed.stdout
