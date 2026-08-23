"""
src/realtime_audio.py
Real-time audio deepfake detection from your microphone.

Continuously records short overlapping windows from the default mic, runs the
trained audio detector on each window, and prints a live REAL / FAKE readout
with a rolling-averaged fake probability.

Run from the project root:
    python -m src.realtime_audio
    python -m src.realtime_audio --checkpoint models/best_audio_efficientnet.pth --threshold 0.5

Requires the optional `sounddevice` package (pip install sounddevice).
Press Ctrl+C to stop.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    from src.audio_infer import (
        load_audio_detector, predict_from_array,
        DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_CHECKPOINT,
    )
    from src.audio_dataset import SAMPLE_RATE, DURATION
except ModuleNotFoundError:  # pragma: no cover
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.audio_infer import (
        load_audio_detector, predict_from_array,
        DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_CHECKPOINT,
    )
    from src.audio_dataset import SAMPLE_RATE, DURATION


def bar(prob, width=24):
    fill = int(width * min(max(prob, 0.0), 1.0))
    return "[" + "#" * fill + "-" * (width - fill) + "]"


def run(args):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(
            "Real-time mic detection needs the 'sounddevice' package:\n"
            "    pip install sounddevice"
        )

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint}\n"
            f"Train a model first (python -m src.audio_train), or pass --checkpoint."
        )

    print(f"Loading {args.model} audio detector ...")
    model, device = load_audio_detector(args.model, str(checkpoint))

    window_samples = int(SAMPLE_RATE * DURATION)
    hop_samples = int(SAMPLE_RATE * args.hop)
    ring = np.zeros(window_samples, dtype=np.float32)
    prob_history = deque(maxlen=args.smooth)

    print(f"Listening on the default mic ({SAMPLE_RATE} Hz, {DURATION:.0f}s window). "
          f"Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=hop_samples) as stream:
            while True:
                block, _ = stream.read(hop_samples)
                block = block[:, 0]
                # Slide the analysis window: drop oldest hop, append newest block.
                ring = np.concatenate([ring[len(block):], block])

                # Skip near-silence so we don't score room noise as speech.
                if np.sqrt(np.mean(ring ** 2)) < args.silence:
                    print("\r  ...silence...".ljust(60), end="", flush=True)
                    continue

                prob = predict_from_array(ring, model, device)
                prob_history.append(prob)
                smoothed = float(np.mean(prob_history))

                label = "FAKE" if smoothed >= args.threshold else "REAL"
                marker = "🔴" if label == "FAKE" else "🟢"
                print(f"\r  {marker} {label:4s} {bar(smoothed)} "
                      f"{smoothed * 100:5.1f}%   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time mic audio deepfake detection")
    p.add_argument("--model", default=DEFAULT_AUDIO_MODEL, choices=["efficientnet", "cnn"])
    p.add_argument("--checkpoint", default=DEFAULT_AUDIO_CHECKPOINT)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hop", type=float, default=1.0,
                   help="Seconds of new audio between predictions (window slides by this)")
    p.add_argument("--smooth", type=int, default=5,
                   help="Rolling-average window (number of predictions)")
    p.add_argument("--silence", type=float, default=0.01,
                   help="RMS threshold below which a window is treated as silence")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
