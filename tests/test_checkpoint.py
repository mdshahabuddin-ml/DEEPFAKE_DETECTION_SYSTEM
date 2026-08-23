"""Tests for the checkpoint auto-download hook."""

from pathlib import Path

import pytest

from src.checkpoint import (
    CheckpointDownloadError, _download_from_url, ensure_checkpoint,
)


def test_existing_checkpoint_is_returned(tmp_path):
    existing = tmp_path / "model.pth"
    existing.write_bytes(b"weights")
    assert ensure_checkpoint(existing) == existing


def test_no_source_configured_returns_none(tmp_path):
    missing = tmp_path / "missing.pth"
    # No URL / HF repo configured -> None (not an error).
    assert ensure_checkpoint(missing, url="", hf_repo="") is None


def test_download_from_file_url(tmp_path):
    """Exercise the streaming downloader offline via a file:// URL."""
    source = tmp_path / "source.pth"
    source.write_bytes(b"x" * 4096)
    dest = tmp_path / "downloaded.pth"

    url = source.resolve().as_uri()  # file:///.../source.pth
    out = _download_from_url(url, dest)

    assert out == dest
    assert dest.exists()
    assert dest.read_bytes() == b"x" * 4096
    # The temporary .part file must be cleaned up.
    assert not dest.with_suffix(dest.suffix + ".part").exists()


def test_download_bad_url_raises(tmp_path):
    dest = tmp_path / "nope.pth"
    with pytest.raises(CheckpointDownloadError):
        _download_from_url((tmp_path / "does_not_exist.pth").resolve().as_uri(), dest)
