"""
scripts/download_asvspoof.py
Automated download of ASVspoof 2019 LA dataset from Edinburgh DataShare.

Usage:
    python scripts/download_asvspoof.py

Downloads LA.zip (~7 GB) to data/asvspoof2019/ and extracts it.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "asvspoof2019"
ZIP_PATH = DOWNLOAD_DIR / "LA.zip"
EXTRACT_DIR = DOWNLOAD_DIR / "LA"

DOWNLOAD_URL = (
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
    "?sequence=3&isAllowed=y"
)


def download_with_progress(url, dest):
    print(f"Downloading from:\n  {url}")
    print(f"To:\n  {dest}\n")

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        if total:
            print(f"File size: {total / (1024**3):.2f} GB")
        else:
            print("File size: unknown (streaming)")

        downloaded = 0
        start_time = time.time()
        chunk_size = 1024 * 1024

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                elapsed = time.time() - start_time
                speed = downloaded / elapsed if elapsed > 0 else 0
                if total:
                    pct = downloaded / total * 100
                    eta = (total - downloaded) / speed if speed > 0 else 0
                    mb_down = downloaded / (1024**2)
                    mb_total = total / (1024**2)
                    print(f"\r  {mb_down:.0f}/{mb_total:.0f} MB ({pct:.1f}%) | {speed/(1024**2):.1f} MB/s | ETA: {eta:.0f}s", end="", flush=True)
                else:
                    mb = downloaded / (1024**2)
                    print(f"\r  {mb:.0f} MB | {speed/(1024**2):.1f} MB/s", end="", flush=True)

    print(f"\n\nDownload complete: {dest}")
    sz = os.path.getsize(dest)
    print(f"Size: {sz / (1024**3):.2f} GB")


def extract_zip(zip_path, extract_dir):
    import zipfile
    print(f"\nExtracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"Files in archive: {len(members)}")
        top_dirs = set()
        for m in members:
            parts = m.split("/")
            if len(parts) > 1:
                top_dirs.add(parts[0])
        print(f"Top-level entries: {sorted(top_dirs)}")
        zf.extractall(extract_dir)
    print(f"Extraction complete: {extract_dir}")


def main():
    os.chdir(PROJECT_ROOT)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        proto_dir = EXTRACT_DIR / "ASVspoof2019_LA_cm_protocols"
        if not proto_dir.exists():
            proto_dir = EXTRACT_DIR / "LA" / "ASVspoof2019_LA_cm_protocols"
        if proto_dir.exists():
            print(f"Dataset already extracted at {EXTRACT_DIR}")
            print("Protocol files found - ready for preparation!")
            return
        else:
            print(f"Extracted dir exists but no protocols found. Will re-extract.")

    if ZIP_PATH.exists():
        size_gb = ZIP_PATH.stat().st_size / (1024**3)
        print(f"LA.zip already exists ({size_gb:.2f} GB)")
    else:
        print("=" * 60)
        print("ASVspoof 2019 LA Dataset Download")
        print("=" * 60)
        print()
        print("If automated download fails, download LA.zip manually from:")
        print("  https://datashare.ed.ac.uk/handle/10283/3336")
        print(f"and save to: {ZIP_PATH}")
        print()

        try:
            download_with_progress(DOWNLOAD_URL, ZIP_PATH)
        except Exception as e:
            print(f"\nAutomated download failed: {e}")
            print("\nPlease download LA.zip manually from:")
            print("  https://datashare.ed.ac.uk/handle/10283/3336")
            print(f"Save to: {ZIP_PATH}")
            print("Then re-run this script to extract.")
            sys.exit(1)

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1024**3:
        extract_zip(ZIP_PATH, EXTRACT_DIR)
    else:
        print("ZIP file too small or missing - cannot extract")

    print("\n" + "=" * 60)
    print("Done! Next step: python scripts/prepare_asvspoof.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
