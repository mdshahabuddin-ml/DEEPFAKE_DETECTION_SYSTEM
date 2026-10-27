"""
scripts/test_webcam_pipeline.py

Tests the FIXED live-webcam frame pipeline on known-real (and known-fake)
still frames, without requiring camera hardware.

Each image is fed through the exact worker code path
(src.live_detection.process_video_frame):
  Haar largest-face detect -> 20% margin crop -> quality gate
  (min 80px, Laplacian blur >= 100) -> BGR->RGB + Resize(299) +
  ImageNet norm -> Xception sigmoid, with rolling-average smoothing
  (window 8) across consecutive frames exactly as a live feed would.

Reports per-frame raw probability, smoothed probability, threshold and
final verdict (REAL / UNCERTAIN / FAKE / SKIPPED), plus summary rates.

Usage:
    .venv/Scripts/python.exe -u scripts/test_webcam_pipeline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_val_transforms  # noqa: E402
from src.inference import FaceDetector  # noqa: E402
from src.live_detection import (  # noqa: E402
    LiveSession,
    LiveSessionConfig,
    classify_live,
)
from src.model import build_model, load_checkpoint  # noqa: E402

CHECKPOINT = PROJECT_ROOT / "models" / "best_xception_celebdf.pth"
REAL_DIR = PROJECT_ROOT / "data" / "indian_eval" / "real"
FAKE_DIR = PROJECT_ROOT / "data" / "indian_eval" / "fake"
IMG_SIZE = 299
THRESHOLD = 0.47
MARGIN = 0.10


def main():
    import src.live_detection as ld

    print("Loading model:", CHECKPOINT.name, flush=True)
    device = "cpu"
    model = build_model("xception", pretrained=False, dropout=0.5,
                        device=device)
    load_checkpoint(model, str(CHECKPOINT), device=device)
    model.eval()
    transform = get_val_transforms(IMG_SIZE)
    face_detector = FaceDetector(backend="auto")

    cfg = LiveSessionConfig(threshold=THRESHOLD, uncertain_margin=MARGIN,
                            infer_every_n=1, smooth_window=8)
    real_paths = sorted(REAL_DIR.glob("*.png"))
    fake_paths = sorted(FAKE_DIR.glob("*.png"))
    assert real_paths and fake_paths, "eval image dirs missing"

    for tag, paths in [("KNOWN-REAL", real_paths), ("KNOWN-FAKE", fake_paths)]:
        session = LiveSession(cfg)
        session.running = True  # enable history without threads
        t0 = time.time()
        print(f"\n=== {tag} ({len(paths)} frames) ===", flush=True)
        print(f"{'frame':>24} {'raw':>7} {'smooth':>7} "
              f"{'verdict':>9}  note", flush=True)
        counts = {"REAL": 0, "UNCERTAIN": 0, "FAKE": 0, "SKIPPED": 0,
                  "NO_FACE": 0}
        shown = 0
        for i, p in enumerate(paths):
            bgr = cv2.imread(str(p))
            if bgr is None:
                counts["SKIPPED"] += 1
                continue
            r = ld.process_video_frame(session, bgr, model, device,
                                       transform, face_detector, i)
            sm = r["smoothed"]
            if r["box"] is None:
                verdict, note = "NO_FACE", "no face"
                counts["NO_FACE"] += 1
            elif r["reject"]:
                verdict, note = "SKIPPED", r["reject"]
                counts["SKIPPED"] += 1
            else:
                verdict, _, _ = classify_live(sm, THRESHOLD, MARGIN)
                counts[verdict] += 1
                note = ""
            raw_s = f"{r['raw_prob']:.3f}" if r["raw_prob"] is not None else "  --  "
            sm_s = f"{sm:.3f}" if sm is not None else "  --  "
            if shown < 12:
                print(f"{p.name:>24} {raw_s:>7} {sm_s:>7} "
                      f"{verdict:>9}  {note}", flush=True)
                shown += 1
        n_scored = counts["REAL"] + counts["UNCERTAIN"] + counts["FAKE"]
        print(f"... ({len(paths) - shown} more frames)", flush=True)
        print(f"Summary @{THRESHOLD}/margin {MARGIN}: {counts} "
              f"in {time.time() - t0:.1f}s", flush=True)
        if tag == "KNOWN-REAL" and n_scored:
            fpr = counts["FAKE"] / n_scored
            print(f"Known-real FPR (FAKE bogus verdicts): {fpr:.3f} | "
                  f"UNCERTAIN rate: {counts['UNCERTAIN'] / n_scored:.3f} | "
                  f"frame reject/no-face: "
                  f"{(counts['SKIPPED'] + counts['NO_FACE'])}/{len(paths)}",
                  flush=True)
        if tag == "KNOWN-FAKE" and n_scored:
            print(f"Known-fake recall: "
                  f"{counts['FAKE'] / n_scored:.3f}", flush=True)


if __name__ == "__main__":
    main()
