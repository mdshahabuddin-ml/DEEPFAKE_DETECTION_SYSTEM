"""Tests for the robustness benchmark corruption functions."""

import numpy as np

from src.robustness import CORRUPTIONS


def test_all_corruptions_return_valid_images():
    img = (np.random.rand(100, 100, 3) * 255).astype("uint8")
    for name, fn in CORRUPTIONS.items():
        out = fn(img)
        assert out is not None
        assert out.dtype == np.uint8
        assert out.ndim == 3 and out.shape[2] == 3
        assert out.size > 0, f"{name} produced an empty image"


def test_clean_is_identity():
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    assert np.array_equal(CORRUPTIONS["clean"](img), img)
