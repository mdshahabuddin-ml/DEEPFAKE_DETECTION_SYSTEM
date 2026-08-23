"""
tests/conftest.py
Shared pytest fixtures and path setup.
"""

import sys
from pathlib import Path

# Make the project root importable regardless of where pytest is invoked.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest

from src.config import CHECKPOINT_PATH

# Skip marker for tests that need a trained model checkpoint.
requires_model = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason=f"Model checkpoint not found at {CHECKPOINT_PATH}",
)


@pytest.fixture
def dummy_bgr_image():
    """A small synthetic BGR image (no real face)."""
    return (np.random.rand(240, 240, 3) * 255).astype("uint8")
