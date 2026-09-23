"""
src/audio_dataset.py
Audio dataset + feature extraction for audio deepfake (spoof) detection.

The model works on log-mel spectrograms treated as single-channel images.
Audio is loaded with librosa (handles wav / flac / mp3 / m4a / ogg), resampled
to a fixed sample rate, and converted to a fixed-size log-mel spectrogram so
every clip becomes a tensor of shape (1, N_MELS, N_FRAMES).

Supported data layouts (label 0 = REAL/bonafide, 1 = FAKE/spoof):

  1. Folder layout:
        <root>/real/*.wav   (or bonafide/, genuine/)
        <root>/fake/*.wav   (or spoof/, synthetic/)

  2. Split CSV with columns: path,label,split   (split in {train,val,test})
"""

import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import librosa

# ---------------------------------------------------------------------------
# Audio / feature configuration -- keep in sync with audio_infer.py & the app
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 4.0                       # seconds per clip (pad/truncate to this)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
N_FRAMES = int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH))  # target time steps

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}

# Common folder names mapped to labels (0 = real, 1 = fake)
_REAL_NAMES = {"real", "bonafide", "genuine", "human", "original"}
_FAKE_NAMES = {"fake", "spoof", "synthetic", "tts", "vc", "generated", "clone"}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_waveform(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono float32 waveform at `sr` Hz."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def fix_waveform_length(y: np.ndarray, sr: int = SAMPLE_RATE,
                        duration: float = DURATION) -> np.ndarray:
    """Pad (with zeros) or center-truncate the waveform to a fixed length."""
    target = int(sr * duration)
    if len(y) < target:
        return np.pad(y, (0, target - len(y)), mode="constant")
    if len(y) > target:
        start = (len(y) - target) // 2
        return y[start:start + target]
    return y


def waveform_to_logmel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convert a waveform to a log-mel spectrogram of shape (N_MELS, time)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    return logmel.astype(np.float32)


def fix_frames(logmel: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """Pad or truncate the time axis of a spectrogram to `n_frames`."""
    t = logmel.shape[1]
    if t < n_frames:
        return np.pad(logmel, ((0, 0), (0, n_frames - t)), mode="constant",
                      constant_values=logmel.min())
    if t > n_frames:
        return logmel[:, :n_frames]
    return logmel


def standardize(logmel: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-std normalization per clip (robust to level changes)."""
    mean = logmel.mean()
    std = logmel.std()
    return (logmel - mean) / (std + 1e-6)


def waveform_to_input(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full pipeline: waveform -> normalized (1, N_MELS, N_FRAMES) float32 array."""
    y = fix_waveform_length(y, sr=sr)
    logmel = waveform_to_logmel(y, sr=sr)
    logmel = fix_frames(logmel)
    logmel = standardize(logmel)
    return logmel[np.newaxis, :, :]  # add channel dim


def file_to_input_tensor(path, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a file and return a (1, 1, N_MELS, N_FRAMES) batch-ready tensor."""
    y = load_waveform(path, sr=sr)
    arr = waveform_to_input(y, sr=sr)
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AudioDeepfakeDataset(Dataset):
    """Returns (spectrogram_tensor[1, N_MELS, N_FRAMES], label)."""

    def __init__(self, samples: List[Tuple[str, int]], augment: bool = False,
                 sr: int = SAMPLE_RATE):
        self.samples = samples
        self.augment = augment
        self.sr = sr

    def __len__(self):
        return len(self.samples)

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        # Light, label-preserving augmentation for the training split.
        if random.random() < 0.5:
            y = y + np.random.normal(0, 0.005, size=y.shape).astype(np.float32)
        if random.random() < 0.5:
            y = y * random.uniform(0.8, 1.2)  # gain jitter
        return y

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            y = load_waveform(path, sr=self.sr)
        except Exception:
            y = np.zeros(int(self.sr * DURATION), dtype=np.float32)
        if self.augment:
            y = self._augment_waveform(y)
        arr = waveform_to_input(y, sr=self.sr)
        return torch.from_numpy(arr), label


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------
def load_folder_samples(root) -> List[Tuple[str, int]]:
    """Scan <root>/<class>/*.<audio> where class name implies real/fake."""
    root = Path(root)
    samples: List[Tuple[str, int]] = []
    if not root.exists():
        print(f"[WARNING] Audio data root not found: {root}")
        return samples

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        name = sub.name.lower()
        if name in _REAL_NAMES:
            label = 0
        elif name in _FAKE_NAMES:
            label = 1
        else:
            print(f"[WARNING] Skipping unrecognized class folder: {sub.name}")
            continue
        for f in sorted(sub.rglob("*")):
            if f.suffix.lower() in AUDIO_EXTENSIONS:
                samples.append((str(f), label))

    n_real = sum(1 for _, l in samples if l == 0)
    n_fake = sum(1 for _, l in samples if l == 1)
    print(f"Audio folder: {n_real} real, {n_fake} fake")
    return samples


def stratified_split(samples, val_split=0.15, test_split=0.15, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)
    train, val, test = [], [], []
    for group in by_label.values():
        rng.shuffle(group)
        n = len(group)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def load_split_csv(split_csv) -> dict:
    df = pd.read_csv(split_csv)
    required = {"path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")
    out = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        split = str(row["split"]).strip().lower()
        if split in out:
            out[split].append((str(row["path"]), int(row["label"])))
    return out


def make_dataloaders(train_samples, val_samples, test_samples,
                     batch_size=32, num_workers=2):
    train_ds = AudioDeepfakeDataset(train_samples, augment=True)
    val_ds = AudioDeepfakeDataset(val_samples, augment=False)
    test_ds = AudioDeepfakeDataset(test_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def get_audio_dataloaders(data_root=None, split_csv=None, batch_size=32,
                          num_workers=2, val_split=0.15, test_split=0.15, seed=42):
    if split_csv:
        splits = load_split_csv(split_csv)
        train, val, test = splits["train"], splits["val"], splits["test"]
    else:
        samples = load_folder_samples(data_root)
        if not samples:
            raise SystemExit(
                f"No audio files found under {data_root}. Expected real/ and fake/ "
                f"subfolders, or pass --split_csv."
            )
        train, val, test = stratified_split(samples, val_split, test_split, seed)
    return make_dataloaders(train, val, test, batch_size=batch_size,
                            num_workers=num_workers)



class CachedAudioDeepfakeDataset(Dataset):
    """Loads pre-computed spectrogram .npy files for fast training."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_path, label = self.samples[idx]
        try:
            spec = np.load(cache_path)
            if spec.dtype == np.float16:
                spec = spec.astype(np.float32)
        except Exception:
            spec = np.zeros((1, 128, 256), dtype=np.float32)
        return torch.from_numpy(spec), label


def load_cached_split_csv(cache_index_path):
    """Load cache_index.csv with columns: cache_path, label, split."""
    df = pd.read_csv(cache_index_path)
    required = {"cache_path", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cache index missing columns: {sorted(missing)}")
    out = {"train": [], "val": [], "test": []}
    for _, row in df.iterrows():
        split = str(row["split"]).strip().lower()
        if split in out:
            out[split].append((str(row["cache_path"]), int(row["label"])))
    return out


def make_cached_dataloaders(train_samples, val_samples, test_samples,
                            batch_size=32, num_workers=0):
    train_ds = CachedAudioDeepfakeDataset(train_samples)
    val_ds = CachedAudioDeepfakeDataset(val_samples)
    test_ds = CachedAudioDeepfakeDataset(test_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    print(f"Cached - Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


def get_cached_audio_dataloaders(cache_index, batch_size=32, num_workers=0):
    """Load dataloaders from pre-cached spectrogram .npy files."""
    splits = load_cached_split_csv(cache_index)
    return make_cached_dataloaders(
        splits["train"], splits["val"], splits["test"],
        batch_size=batch_size, num_workers=num_workers,
    )