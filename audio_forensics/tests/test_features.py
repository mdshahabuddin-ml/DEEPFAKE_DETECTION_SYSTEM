"""Tests for audio feature extraction module.

These tests verify the feature extraction pipeline works correctly
without modifying any existing system tests.
"""

import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from audio_forensics.config import AudioForensicsConfig
from audio_forensics.features import (
    AudioFeatureExtractor,
    FeatureExtractionError,
    FeatureSet,
)


def create_test_audio(
    duration: float = 2.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Create a test audio signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


def create_speech_like_audio(
    duration: float = 3.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Create speech-like audio with harmonics and modulation."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Fundamental frequency with vibrato
    f0 = 150 + 10 * np.sin(2 * np.pi * 5 * t)

    # Harmonics
    audio = (
        0.4 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * f0 * 2 * t) +
        0.2 * np.sin(2 * np.pi * f0 * 3 * t) +
        0.1 * np.sin(2 * np.pi * f0 * 4 * t)
    )

    # Add some noise
    audio += 0.02 * np.random.randn(len(audio))

    return audio.astype(np.float32)


def test_feature_extractor_initialization():
    """Test feature extractor initialization."""
    print("Testing feature extractor initialization...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    assert extractor is not None
    assert extractor.config is config

    print("  [PASS] Feature extractor initialized")


def test_log_mel_extraction():
    """Test Log-Mel spectrogram extraction."""
    print("Testing Log-Mel extraction...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_test_audio(duration=2.0)
    features = extractor.extract(audio, 16000, ["log_mel"])

    assert features.log_mel is not None
    assert features.log_mel.ndim == 2
    assert features.log_mel.shape[0] == config.n_mels

    print(f"  [PASS] Log-Mel shape: {features.log_mel.shape}")


def test_mfcc_extraction():
    """Test MFCC extraction."""
    print("Testing MFCC extraction...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_test_audio(duration=2.0)
    features = extractor.extract(audio, 16000, ["mfcc"])

    assert features.mfcc is not None
    assert features.mfcc.ndim == 2
    assert features.mfcc.shape[0] == config.n_mfcc

    print(f"  [PASS] MFCC shape: {features.mfcc.shape}")


def test_mfcc_delta_extraction():
    """Test MFCC delta extraction."""
    print("Testing MFCC delta extraction...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_test_audio(duration=2.0)
    features = extractor.extract(audio, 16000, ["mfcc", "mfcc_delta", "mfcc_delta2"])

    assert features.mfcc is not None
    assert features.mfcc_delta is not None
    assert features.mfcc_delta2 is not None

    # Delta should have same shape as MFCC
    assert features.mfcc_delta.shape == features.mfcc.shape
    assert features.mfcc_delta2.shape == features.mfcc.shape

    print(f"  [PASS] MFCC delta shapes match")


def test_spectral_features():
    """Test spectral feature extraction."""
    print("Testing spectral features...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_speech_like_audio(duration=2.0)
    features = extractor.extract(
        audio, 16000,
        ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]
    )

    assert features.spectral_centroid is not None
    assert features.spectral_bandwidth is not None
    assert features.spectral_rolloff is not None

    # Check values are reasonable
    assert np.all(features.spectral_centroid > 0)
    assert np.all(features.spectral_bandwidth > 0)

    print(f"  [PASS] Spectral features extracted")


def test_temporal_features():
    """Test temporal feature extraction."""
    print("Testing temporal features...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_speech_like_audio(duration=2.0)
    features = extractor.extract(audio, 16000, ["zero_crossing_rate", "rms_energy"])

    assert features.zero_crossing_rate is not None
    assert features.rms_energy is not None

    assert len(features.zero_crossing_rate) > 0
    assert len(features.rms_energy) > 0

    # ZCR should be between 0 and 1
    assert np.all(features.zero_crossing_rate >= 0)
    assert np.all(features.zero_crossing_rate <= 1)

    print(f"  [PASS] Temporal features extracted")


def test_feature_set_to_array():
    """Test FeatureSet to_array conversion."""
    print("Testing FeatureSet to_array...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_test_audio(duration=2.0)
    features = extractor.extract(audio, 16000)

    # Convert to array
    feature_array = features.to_array()

    assert isinstance(feature_array, np.ndarray)
    assert feature_array.dtype == np.float32
    assert len(feature_array) > 0

    print(f"  [PASS] Feature array shape: {feature_array.shape}")


def test_feature_set_to_dict():
    """Test FeatureSet to_dict conversion."""
    print("Testing FeatureSet to_dict...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_test_audio(duration=2.0)
    features = extractor.extract(audio, 16000)

    # Convert to dict
    feature_dict = features.to_dict()

    assert isinstance(feature_dict, dict)
    assert len(feature_dict) > 0

    print(f"  [PASS] Feature dict keys: {list(feature_dict.keys())[:5]}...")


def test_empty_audio():
    """Test handling of empty audio."""
    print("Testing empty audio handling...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    try:
        extractor.extract(np.array([]), 16000)
        assert False, "Should raise FeatureExtractionError"
    except FeatureExtractionError:
        pass

    print("  [PASS] Empty audio handled correctly")


def test_invalid_sample_rate():
    """Test handling of invalid sample rate."""
    print("Testing invalid sample rate...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_test_audio(duration=1.0)

    try:
        extractor.extract(audio, 0)
        assert False, "Should raise FeatureExtractionError"
    except FeatureExtractionError:
        pass

    print("  [PASS] Invalid sample rate handled correctly")


def test_stereo_audio():
    """Test handling of stereo audio."""
    print("Testing stereo audio...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    # Create stereo audio
    mono = create_test_audio(duration=1.0)
    stereo = np.stack([mono, mono], axis=1)

    features = extractor.extract(stereo, 16000, ["log_mel"])

    assert features.log_mel is not None

    print("  [PASS] Stereo audio converted to mono")


def test_feature_extraction_with_speech_audio():
    """Test feature extraction with speech-like audio."""
    print("Testing feature extraction with speech audio...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio = create_speech_like_audio(duration=3.0)
    features = extractor.extract(audio, 16000)

    # Should have multiple features
    assert len(features.feature_names) > 0

    # Duration should be correct
    assert abs(features.duration - 3.0) < 0.1

    print(f"  [PASS] Extracted {len(features.feature_names)} features")


def test_batch_extraction():
    """Test batch feature extraction."""
    print("Testing batch extraction...")

    config = AudioForensicsConfig()
    extractor = AudioFeatureExtractor(config)

    audio_batch = [
        create_test_audio(duration=1.0),
        create_test_audio(duration=1.5),
        create_test_audio(duration=2.0),
    ]

    results = extractor.extract_batch(audio_batch, 16000, ["log_mel"])

    assert len(results) == 3
    for r in results:
        assert r.log_mel is not None

    print("  [PASS] Batch extraction works")


def run_all_tests():
    """Run all feature extraction tests."""
    print("=" * 60)
    print("AUDIO FEATURE EXTRACTION TESTS")
    print("=" * 60)

    tests = [
        test_feature_extractor_initialization,
        test_log_mel_extraction,
        test_mfcc_extraction,
        test_mfcc_delta_extraction,
        test_spectral_features,
        test_temporal_features,
        test_feature_set_to_array,
        test_feature_set_to_dict,
        test_empty_audio,
        test_invalid_sample_rate,
        test_stereo_audio,
        test_feature_extraction_with_speech_audio,
        test_batch_extraction,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
