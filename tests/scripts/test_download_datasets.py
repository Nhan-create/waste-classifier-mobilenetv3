from pathlib import Path

import kagglehub

from scripts.download_datasets import PINNED_DATASETS, download_pinned_datasets


def test_downloader_uses_exact_pinned_handles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(handle: str, *, output_dir: Path) -> str:
        calls.append((handle, output_dir))
        return str(output_dir)

    monkeypatch.setattr(kagglehub, "dataset_download", fake_download)

    result = download_pinned_datasets(tmp_path)

    assert PINNED_DATASETS == {
        "vn_trash": "mrgetshjtdone/vn-trash-classification/versions/1",
        "garbage_v2": "sumn2u/garbage-classification-v2/versions/12",
    }
    assert calls == [
        (
            "mrgetshjtdone/vn-trash-classification/versions/1",
            tmp_path / "vn_trash",
        ),
        (
            "sumn2u/garbage-classification-v2/versions/12",
            tmp_path / "garbage_v2",
        ),
    ]
    assert result == {
        "vn_trash": tmp_path / "vn_trash",
        "garbage_v2": tmp_path / "garbage_v2",
    }


def test_downloader_error_does_not_request_or_print_a_secret(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_download(handle: str, *, output_dir: Path) -> str:
        raise OSError("authentication unavailable")

    monkeypatch.setattr(kagglehub, "dataset_download", fail_download)

    try:
        download_pinned_datasets(tmp_path)
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("expected a wrapped download failure")

    assert "configure KaggleHub authentication outside the repository" in message
    assert "token" not in message.lower()
