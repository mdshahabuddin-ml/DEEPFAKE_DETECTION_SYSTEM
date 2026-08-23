"""
src/checkpoint.py
Auto-download the model checkpoint on startup when it is missing locally.

Two sources are supported (configured via environment variables / config.py):
  1. A direct HTTPS URL   -> DEEPFAKE_CHECKPOINT_URL
  2. A Hugging Face repo  -> DEEPFAKE_HF_REPO (+ DEEPFAKE_HF_FILENAME / REVISION)

`ensure_checkpoint()` is called by the inference engine before loading weights,
so every entry point (Gradio, Streamlit, FastAPI, CLI) benefits automatically.

The URL downloader uses only the standard library. The Hugging Face path uses
the optional `huggingface_hub` package and fails with a clear message if it is
not installed.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.error import URLError

from src.config import (
    CHECKPOINT_PATH, CHECKPOINT_URL, HF_FILENAME, HF_REPO, HF_REVISION,
)
from src.logger import get_logger

logger = get_logger(__name__)


class CheckpointDownloadError(RuntimeError):
    """Raised when an auto-download source is configured but fails."""


def _download_from_url(url: str, dest: Path) -> Path:
    """Stream a file from `url` to `dest` atomically (via a .part temp file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading checkpoint from URL: %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "deepfake-detector"})
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            next_log = 10  # log at each ~10% step
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct >= next_log:
                        logger.info("  ... %d%% (%d/%d bytes)", pct, downloaded, total)
                        next_log = pct - (pct % 10) + 10
    except (URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise CheckpointDownloadError(f"URL download failed: {exc}") from exc

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise CheckpointDownloadError("Downloaded checkpoint is empty.")

    tmp.replace(dest)
    logger.info("Checkpoint saved to %s", dest)
    return dest


def _download_from_hf(repo_id: str, filename: str, dest: Path,
                      revision: Optional[str] = None) -> Path:
    """Fetch `filename` from a Hugging Face Hub repo and copy it to `dest`."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise CheckpointDownloadError(
            "huggingface_hub is not installed. Run `pip install huggingface_hub` "
            "or use DEEPFAKE_CHECKPOINT_URL instead."
        ) from exc

    logger.info("Downloading checkpoint from HF Hub: %s/%s%s", repo_id, filename,
                f"@{revision}" if revision else "")
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    except Exception as exc:  # pragma: no cover - network / repo errors
        raise CheckpointDownloadError(f"HF Hub download failed: {exc}") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, dest)
    logger.info("Checkpoint saved to %s", dest)
    return dest


def ensure_checkpoint(path: Path = CHECKPOINT_PATH, *,
                      url: str = CHECKPOINT_URL,
                      hf_repo: str = HF_REPO,
                      hf_filename: str = HF_FILENAME,
                      hf_revision: Optional[str] = HF_REVISION) -> Optional[Path]:
    """
    Ensure a checkpoint exists at `path`, downloading it if configured.

    Returns the path if it exists (already present or freshly downloaded), or
    None if the file is missing and no download source is configured. A
    configured-but-failing source raises CheckpointDownloadError.
    """
    path = Path(path)
    if path.exists():
        return path

    if url:
        return _download_from_url(url, path)
    if hf_repo:
        return _download_from_hf(hf_repo, hf_filename, path, revision=hf_revision)

    logger.info("No checkpoint at %s and no download source configured "
                "(set DEEPFAKE_CHECKPOINT_URL or DEEPFAKE_HF_REPO).", path)
    return None


if __name__ == "__main__":
    # Manual trigger: `python -m src.checkpoint`
    result = ensure_checkpoint()
    if result and result.exists():
        print(f"Checkpoint ready at: {result}")
    else:
        raise SystemExit(
            "No checkpoint available. Set DEEPFAKE_CHECKPOINT_URL or "
            "DEEPFAKE_HF_REPO (and place/download the .pth), then retry."
        )
