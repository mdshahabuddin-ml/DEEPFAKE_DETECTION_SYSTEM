"""Configuration for temporal analysis module."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalConfig:
    """Configuration for temporal analysis."""
    # Frame sampling
    target_fps: float = 10.0
    max_frames: int = 100
    min_frames: int = 5
    frame_size: tuple = (224, 224)

    # Analysis
    window_size: int = 5
    stride: int = 1
    anomaly_threshold: float = 0.5
    consistency_threshold: float = 0.7

    # Model
    feature_model: str = "efficientnet"
    device: str = "auto"
    batch_size: int = 8

    # Output
    output_dir: str = "results/temporal_analysis"

    @classmethod
    def from_env(cls) -> "TemporalConfig":
        return cls(
            target_fps=float(os.getenv("TEMPORAL_FPS", "10.0")),
            max_frames=int(os.getenv("TEMPORAL_MAX_FRAMES", "100")),
            device=os.getenv("TEMPORAL_DEVICE", "auto"),
        )
