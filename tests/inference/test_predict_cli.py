import subprocess
import sys


def test_predict_cli_help_lists_checkpoint_image_and_top_k() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.inference.predict", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--image" in completed.stdout
    assert "--top-k" in completed.stdout
    assert "--confidence-threshold" in completed.stdout
