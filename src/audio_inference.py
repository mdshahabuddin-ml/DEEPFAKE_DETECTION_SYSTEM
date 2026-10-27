"""
src/audio_inference.py
Audio deepfake (AI voice / cloned speech) inference.

Conventions match the rest of the app (see src/inference.py,
app/predict.py, src/database.py):

    result = detector.predict_audio(path)
    result["prediction"]        # "REAL" | "FAKE"
    result["probability_fake"]  # raw fake probability in [0, 1]
    result["fake_probability"]  # alias of probability_fake
    result["confidence"]        # confidence in the chosen label, [0, 1]
    result["inference_time"]    # seconds (float)
    result["file_name"]         # basename, filled in by app/predict.py

Logged to SQLite history with ``media_type="audio"`` via
``app.predict.store_result(result, "audio")`` so the existing History and
Dashboard pages pick it up with no changes.

Pipeline
--------
1. Load with librosa, resample to 16 kHz mono, peak-normalize.
2. Optional spectral-gating denoise (``noisereduce``). Music separation
   via ``demucs`` is an optional extra: skipped silently when not installed.
3. Clips shorter than 4 s are loop-padded to one window; longer clips are
   split into overlapping 4 s windows (2 s hop) and window probabilities
   are averaged. This keeps short clips (< 7 s) reliable.
4. Features: log-mel spectrogram (classifier input, same shape as
   ``src/audio_dataset.waveform_to_input``) plus MFCCs / pitch-adjacent
   spectral stats kept as diagnostics. The classifier input stays log-mel
   so existing ``models/best_audio_*.pth`` checkpoints remain compatible;
   see the "swap point" note in :class:`AudioDeepfakeDetector`.
5. If no trained checkpoint exists at ``models/audio_detector.pth`` the
   detector raises :class:`AudioModelNotFoundError` and the Streamlit page
   shows a warning instead of returning meaningless numbers.

Datasets for training (see src/train_audio.py):
    - ASVspoof 2019 LA, WaveFake, In-the-Wild.
    - Arrange as ``<root>/real/*`` + ``<root>/fake/*`` (also accepts
      bonafide/genuine/human and spoof/synthetic/tts/vc folders).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.config import (
    DEFAULT_THRESHOLD,
    DEVICE,
    MODELS_DIR,
    confidence_from_prob,
    label_from_prob,
)
from src.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (kept in sync with src/audio_dataset.py)
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_WINDOW_SEC: float = 4.0
AUDIO_HOP_SEC: float = 2.0
MIN_VALID_SEC: float = 0.5
SILENCE_RMS: float = 1e-5

AUDIO_CHECKPOINT: Path = MODELS_DIR / "audio_detector.pth"
# Existing checkpoints this module can also load (same log-mel CNN family).
LEGACY_CHECKPOINTS: List[Path] = [
    MODELS_DIR / "best_audio_efficientnet.pth",
    MODELS_DIR / "best_audio_cnn.pth",
]

ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

N_MFCC: int = 20


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AudioModelNotFoundError(FileNotFoundError):
    """Raised when no trained audio checkpoint is available."""


class AudioInferenceError(ValueError):
    """Raised for unreadable / silent / too-short audio (friendly message)."""


# ---------------------------------------------------------------------------
# Audio loading / preprocessing
# ---------------------------------------------------------------------------
def load_waveform_16k(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio as 16 kHz mono float32. Raises AudioInferenceError."""
    import librosa

    path = Path(path)
    if path.suffix.lower() not in ALLOWED_AUDIO_EXT:
        raise AudioInferenceError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Use one of: {sorted(ALLOWED_AUDIO_EXT)}."
        )
    try:
        y, _ = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioInferenceError(
            f"Could not read audio file '{path.name}'. "
            "The file may be corrupt or not a real audio file."
        ) from exc
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise AudioInferenceError(
            f"'{path.name}' contains no audio samples."
        )
    duration = float(y.size / AUDIO_SAMPLE_RATE)
    if duration < MIN_VALID_SEC:
        raise AudioInferenceError(
            f"'{path.name}' is only {duration:.2f}s long. "
            f"Please upload at least {MIN_VALID_SEC:.1f}s of speech."
        )
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if rms < SILENCE_RMS or float(np.max(np.abs(y))) < 1e-4:
        raise AudioInferenceError(
            f"'{path.name}' looks silent (no detectable speech energy). "
            "Please upload a clip with audible speech."
        )
    return y, AUDIO_SAMPLE_RATE


def normalize_waveform(y: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1] (no-op for all-zero input)."""
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


def loop_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    """Tile a short clip until it reaches target_len (then truncate)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= target_len:
        return y[:target_len]
    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)
    reps = int(np.ceil(target_len / len(y)))
    return np.tile(y, reps)[:target_len].astype(np.float32)


def make_windows(
    y: np.ndarray,
    sr: int = AUDIO_SAMPLE_RATE,
    window_sec: float = AUDIO_WINDOW_SEC,
    hop_sec: float = AUDIO_HOP_SEC,
) -> List[tuple[float, float, np.ndarray]]:
    """Split into overlapping fixed windows.

    Clips shorter than one window are loop-padded to a single window.
    Returns [(start_sec, end_sec, samples)].
    """
    win = int(sr * window_sec)
    hop = int(sr * hop_sec)
    if len(y) < win:
        return [(0.0, window_sec, loop_pad(y, win))]
    out: List[tuple[float, float, np.ndarray]] = []
    start = 0
    while start + win <= len(y):
        out.append((start / sr, (start + win) / sr, y[start:start + win]))
        start += hop
    if start < len(y):  # trailing partial window -> take last full window
        out.append(((len(y) - win) / sr, len(y) / sr, y[len(y) - win:]))
    return out


# ---------------------------------------------------------------------------
# Denoising (optional extras — never crash when deps are missing)
# ---------------------------------------------------------------------------
def denoise_waveform(
    y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE
) -> tuple[np.ndarray, str]:
    """Spectral-gating denoise. Returns (waveform, note).

    Falls back to the original waveform when ``noisereduce`` is not
    installed; the note explains what happened for the UI.
    """
    try:
        import noisereduce as nr
    except ImportError:
        return y, "denoise skipped (pip install noisereduce)"
    try:
        clean = nr.reduce_noise(y=y.astype(np.float32), sr=sr)
        return np.asarray(clean, dtype=np.float32), "denoised with noisereduce"
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("noisereduce failed, using raw audio: %s", exc)
        return y, "denoise failed, used raw audio"


def separate_music_optional(
    y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE
) -> tuple[np.ndarray, str]:
    """Music/vocal separation via demucs — optional extra.

    Returns the original waveform untouched when ``demucs`` is not
    installed (the common case). Callers should treat this as best-effort.
    """
    try:
        import demucs  # noqa: F401
    except ImportError:
        return y, "music separation skipped (demucs not installed)"
    # Full demucs separation is out of scope for the default path; the
    # hook exists so a future swap can plug it in here.
    return y, "demucs available but not applied by default"


# ---------------------------------------------------------------------------
# Features: log-mel (classifier input) + MFCC diagnostics
# ---------------------------------------------------------------------------
def compute_features(y: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> dict:
    """Return log-mel (model input) plus MFCC summary stats.

    The classifier consumes the log-mel spectrogram via
    ``src.audio_dataset.waveform_to_input`` so existing checkpoints stay
    compatible. MFCC means/stds capture pitch / spectral / voice-quality
    cues and are returned for display and for future model swaps.
    """
    import librosa

    from src.audio_dataset import waveform_to_input

    model_input = waveform_to_input(y.astype(np.float32), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return {
        "logmel_input": model_input,  # (1, N_MELS, N_FRAMES) float32
        "mfcc": mfcc.astype(np.float32),
        "mfcc_mean": mfcc.mean(axis=1).astype(np.float32),
        "mfcc_std": mfcc.std(axis=1).astype(np.float32),
    }


def resolve_checkpoint(explicit: Optional[str | Path] = None) -> Path:
    """Return the checkpoint to use, or raise AudioModelNotFoundError."""
    if explicit is not None and Path(explicit).exists():
        return Path(explicit)
    if AUDIO_CHECKPOINT.exists():
        return AUDIO_CHECKPOINT
    for legacy in LEGACY_CHECKPOINTS:
        if legacy.exists():
            logger.info("Using legacy audio checkpoint: %s", legacy)
            return legacy
    # Auto-download (DEEPFAKE_AUDIO_CHECKPOINT_URL / DEEPFAKE_AUDIO_HF_REPO,
    # including Streamlit secrets) before giving up.
    try:
        from src.checkpoint import ensure_audio_checkpoint
        downloaded = ensure_audio_checkpoint(AUDIO_CHECKPOINT)
        if downloaded is not None and downloaded.exists():
            return downloaded
    except Exception as exc:  # noqa: BLE001 - fall through to the error below
        logger.warning("Audio checkpoint auto-download failed: %s", exc)
    raise AudioModelNotFoundError(
        "No trained audio checkpoint found.\n"
        f"Expected: {AUDIO_CHECKPOINT}\n"
        "Train one with: python -m src.train_audio "
        "--data_root data/audio --epochs 15\n"
        "Or configure DEEPFAKE_AUDIO_CHECKPOINT_URL / DEEPFAKE_AUDIO_HF_REPO "
        "(env or Streamlit secrets) to download it automatically."
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class AudioDeepfakeDetector:
    """Clip-level AI-voice detector with swappable backend.

    Default backend: small spectral CNN on log-mel spectrograms
    (``src/audio_model.py``: efficientnet_b0 or from-scratch CNN).

    SWAP POINT for wav2vec2 / AASIST: pass
    ``checkpoint_path`` pointing at the pretrained anti-spoofing weights and
    ``model_name="wav2vec2"`` (or ``"aasist"``). The loader below detects a
    wav2vec2-style state dict (keys containing ``wav2vec2``/``encoder``) and
    raises an informative error telling you to install ``transformers`` and
    wire the HF model in ``_load_wav2vec2_backend`` — the windowing,
    averaging, result-dict and history-logging code is backend-agnostic and
    does not need to change.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = "auto",
    ):
        self.device = device or DEVICE
        self.threshold = float(threshold)
        self.checkpoint = resolve_checkpoint(checkpoint_path)
        self.model_name = model_name
        self.backend = "spectral-cnn"
        self.model = self._load_backend(self.checkpoint, model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Audio detector ready: backend=%s checkpoint=%s device=%s",
            self.backend, self.checkpoint, self.device,
        )

    # -- backend loading -------------------------------------------------
    def _load_backend(self, checkpoint: Path, model_name: str):
        from src.audio_model import build_audio_model

        try:
            from src.checkpoint import safe_torch_load
            state = safe_torch_load(str(checkpoint), map_location="cpu")
            keys = (
                list(state.get("model_state_dict", state).keys())
                if isinstance(state, dict) else []
            )
        except Exception as exc:
            raise AudioModelNotFoundError(
                f"Could not read audio checkpoint {checkpoint}: {exc}"
            ) from exc

        if any("wav2vec2" in k or k.startswith("encoder.") for k in keys):
            return self._load_wav2vec2_backend(checkpoint)

        saved_name = None
        if isinstance(state, dict):
            saved_name = state.get("model_name") or state.get("model")
        for candidate in (
            ([model_name] if model_name not in ("auto", None) else [])
            + ([saved_name] if saved_name in ("efficientnet", "cnn") else [])
            + ["efficientnet", "cnn"]
        ):
            try:
                model = build_audio_model(
                    candidate, pretrained=False, device="cpu"
                )
                sd = state.get("model_state_dict", state)
                model.load_state_dict(sd)
                self.backend = f"spectral-cnn:{candidate}"
                self.model_name = candidate
                return model
            except Exception:
                continue
        raise AudioModelNotFoundError(
            f"Checkpoint {checkpoint} did not match any known spectral-CNN "
            "architecture ('efficientnet' / 'cnn')."
        )

    def _load_wav2vec2_backend(self, checkpoint: Path):  # pragma: no cover
        raise AudioModelNotFoundError(
            f"Checkpoint {checkpoint} looks like a wav2vec2/AASIST "
            "anti-spoofing model. To use it: pip install transformers, then "
            "implement _load_wav2vec2_backend() in src/audio_inference.py "
            "to build the HF model and load these weights. Windowing, "
            "averaging and history logging stay unchanged."
        )

    # -- inference -------------------------------------------------------
    @torch.no_grad()
    def _window_prob(self, window: np.ndarray) -> float:
        from src.audio_dataset import waveform_to_input

        arr = waveform_to_input(window.astype(np.float32))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.model(tensor)).item())

    def predict_audio(
        self, path: str | Path, denoise: bool = False,
        window_sec: float = AUDIO_WINDOW_SEC,
        hop_sec: Optional[float] = None,
    ) -> dict:
        """Run full-clip inference. Keys match src/inference.py style.

        ``window_sec`` (clipping window, 1-8 s) controls the analysis
        windows; hop defaults to half the window. The log-mel model input
        is fixed-size, so any window length works (short windows are
        padded, long ones truncated).
        """
        t0 = time.perf_counter()
        path = Path(path)

        window_sec = min(8.0, max(1.0, float(window_sec)))
        if hop_sec is None:
            hop_sec = window_sec / 2.0

        y, sr = load_waveform_16k(path)
        y = normalize_waveform(y)
        denoise_note = "not requested"
        if denoise:
            y, denoise_note = denoise_waveform(y, sr)

        duration = float(len(y) / sr)
        windows = make_windows(y, sr, window_sec=window_sec, hop_sec=hop_sec)
        probs = [self._window_prob(w) for _, _, w in windows]
        mean_prob = float(np.mean(probs)) if probs else 0.0

        label = label_from_prob(mean_prob, self.threshold)
        confidence = confidence_from_prob(mean_prob, self.threshold)
        elapsed = round(time.perf_counter() - t0, 4)

        segments = [
            {"start": round(s, 2), "end": round(e, 2),
             "probability_fake": float(p)}
            for (s, e, _), p in zip(windows, probs)
        ]
        return {
            "path": str(path),
            "file_name": path.name,
            "label": label,
            "prediction": label,
            "probability_fake": mean_prob,
            "fake_probability": mean_prob,
            "confidence": float(confidence),
            "inference_time": elapsed,
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
            "n_windows": len(windows),
            "window_probs": [float(p) for p in probs],
            "segments": segments,
            "denoised": bool(denoise),
            "denoise_note": denoise_note,
            "window_sec": float(window_sec),
            "hop_sec": float(hop_sec),
            "model": self.model_name,
            "backend": self.backend,
            "threshold": float(self.threshold),
        }


# ---------------------------------------------------------------------------
# Singleton (mirrors src/inference.get_detector)
# ---------------------------------------------------------------------------
_AUDIO_DETECTOR: Optional[AudioDeepfakeDetector] = None


def is_audio_checkpoint_available() -> bool:
    try:
        resolve_checkpoint()
        return True
    except AudioModelNotFoundError:
        return False


def get_audio_detector(force_reload: bool = False) -> AudioDeepfakeDetector:
    global _AUDIO_DETECTOR
    if _AUDIO_DETECTOR is None or force_reload:
        _AUDIO_DETECTOR = AudioDeepfakeDetector()
    return _AUDIO_DETECTOR
