"""Configuration for multimodal fusion module."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MultimodalConfig:
    """Configuration for multimodal analysis."""
    # Consistency analysis
    sync_threshold: float = 0.7
    lip_sync_threshold: float = 0.6

    # Fusion weights (experimental)
    video_weight: float = 0.35
    audio_weight: float = 0.35
    temporal_weight: float = 0.15
    sync_weight: float = 0.15

    # Thresholds
    suspicious_threshold: float = 0.5
    high_confidence_threshold: float = 0.8

    # Device
    device: str = "auto"

    @classmethod
    def from_env(cls) -> "MultimodalConfig":
        return cls(
            sync_threshold=float(os.getenv("MULTI_SYNC_THRESHOLD", "0.7")),
            device=os.getenv("MULTI_DEVICE", "auto"),
        )
