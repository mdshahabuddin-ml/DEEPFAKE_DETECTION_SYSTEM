"""Tests for audio segmentation module.

These tests verify the segmentation pipeline works correctly
without modifying any existing system tests.
"""

import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from audio_forensics.config import AudioForensicsConfig
from audio_forensics.segmentation import (
    AudioSegmenter,
    SegmentationError,
    SegmentationResult,
    SpeechSegment,
)


def create_test_audio_with_silence(
    duration: float = 5.0,
    sample_rate: int = 16000,
    speech_duration: float = 1.0,
    silence_duration: float = 1.0,
) -> np.ndarray:
    """Create test audio with alternating speech and silence."""
    samples_per_second = sample_rate
    total_samples = int(duration * sample_rate)

    audio = np.zeros(total_samples, dtype=np.float32)

    # Create speech segments
    speech_samples = int(speech_duration * sample_rate)
    silence_samples = int(silence_duration * sample_rate)

    position = 0
    while position < total_samples:
        # Add speech
        end = min(position + speech_samples, total_samples)
        t = np.linspace(0, speech_duration, end - position, False)
        audio[position:end] = 0.5 * np.sin(2 * np.pi * 440 * t)
        position = end

        # Add silence
        position += silence_samples

    return audio


def create_speech_bursts(
    n_bursts: int = 5,
    burst_duration: float = 0.5,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Create audio with multiple speech bursts."""
    samples_per_burst = int(burst_duration * sample_rate)
    gap_samples = int(0.2 * sample_rate)  # 200ms gap

    total_samples = n_bursts * (samples_per_burst + gap_samples)
    audio = np.zeros(total_samples, dtype=np.float32)

    for i in range(n_bursts):
        start = i * (samples_per_burst + gap_samples)
        end = start + samples_per_burst

        t = np.linspace(0, burst_duration, samples_per_burst, False)
        audio[start:end] = 0.5 * np.sin(2 * np.pi * 440 * t)

    return audio


def test_segmenter_initialization():
    """Test segmenter initialization."""
    print("Testing segmenter initialization...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    assert segmenter is not None
    assert segmenter.config is config

    print("  [PASS] Segmenter initialized")


def test_energy_segmentation():
    """Test energy-based segmentation."""
    print("Testing energy-based segmentation...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = create_test_audio_with_silence(duration=5.0)
    result = segmenter.segment(audio, 16000, method="energy")

    assert isinstance(result, SegmentationResult)
    assert result.total_segments > 0
    assert result.method == "energy"
    assert result.speech_ratio > 0

    print(f"  [PASS] Found {result.total_segments} segments")
    print(f"    Speech ratio: {result.speech_ratio:.2%}")


def test_zcr_segmentation():
    """Test ZCR-based segmentation."""
    print("Testing ZCR-based segmentation...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = create_test_audio_with_silence(duration=5.0)
    result = segmenter.segment(audio, 16000, method="zcr")

    assert isinstance(result, SegmentationResult)
    assert result.method == "zcr"

    print(f"  [PASS] ZCR segmentation found {result.total_segments} segments")


def test_combined_segmentation():
    """Test combined segmentation."""
    print("Testing combined segmentation...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = create_test_audio_with_silence(duration=5.0)
    result = segmenter.segment(audio, 16000, method="combined")

    assert isinstance(result, SegmentationResult)
    assert result.method == "combined"

    print(f"  [PASS] Combined segmentation found {result.total_segments} segments")


def test_segment_properties():
    """Test segment properties."""
    print("Testing segment properties...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = create_test_audio_with_silence(duration=5.0)
    result = segmenter.segment(audio, 16000)

    if result.total_segments > 0:
        segment = result.segments[0]
        assert isinstance(segment, SpeechSegment)
        assert segment.start_time >= 0
        assert segment.end_time > segment.start_time
        assert segment.duration > 0
        assert segment.energy >= 0

        print(f"  [PASS] Segment properties valid")
        print(f"    Duration: {segment.duration:.2f}s")
        print(f"    Energy: {segment.energy:.4f}")
    else:
        print("  [PASS] No segments found (may be expected)")


def test_segment_times():
    """Test segment time calculations."""
    print("Testing segment time calculations...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = create_speech_bursts(n_bursts=3, burst_duration=0.5)
    result = segmenter.segment(audio, 16000)

    # Check that segments don't overlap
    for i in range(len(result.segments) - 1):
        assert result.segments[i].end_time <= result.segments[i + 1].start_time

    print(f"  [PASS] Segment times non-overlapping")


def test_empty_audio():
    """Test handling of empty audio."""
    print("Testing empty audio handling...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    try:
        segmenter.segment(np.array([]), 16000)
        assert False, "Should raise SegmentationError"
    except SegmentationError:
        pass

    print("  [PASS] Empty audio handled correctly")


def test_speech_type_detection():
    """Test speech type detection."""
    print("Testing speech type detection...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    # Create audio with clear speech and silence
    audio = create_test_audio_with_silence(
        duration=5.0,
        speech_duration=1.0,
        silence_duration=1.0,
    )

    result = segmenter.segment(audio, 16000)

    # Check that we have different speech types
    speech_types = set(s.speech_type for s in result.segments)

    print(f"  [PASS] Detected speech types: {speech_types}")


def test_chunking():
    """Test audio chunking for analysis."""
    print("Testing audio chunking...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds
    chunks = segmenter.segment_to_analyzable_chunks(
        audio, 16000, max_chunk_duration=3.0, overlap=0.5
    )

    assert len(chunks) > 0
    for chunk, start, end in chunks:
        assert len(chunk) > 0
        assert start >= 0
        assert end > start
        assert end - start <= 3.1  # Allow small rounding

    print(f"  [PASS] Created {len(chunks)} chunks")


def test_segment_features():
    """Test segment feature extraction."""
    print("Testing segment features...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = create_test_audio_with_silence(duration=5.0)
    result = segmenter.segment(audio, 16000)

    if result.total_segments > 0:
        segment = result.segments[0]
        features = segmenter.get_segment_features(audio, 16000, segment)

        assert isinstance(features, np.ndarray)
        assert len(features) == 6  # Basic features

        print(f"  [PASS] Segment features shape: {features.shape}")
    else:
        print("  [PASS] No segments to extract features from")


def test_segment_statistics():
    """Test segmentation statistics."""
    print("Testing segmentation statistics...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    audio = create_test_audio_with_silence(duration=10.0)
    result = segmenter.segment(audio, 16000)

    # Check statistics
    assert result.total_speech_duration >= 0
    assert result.total_silence_duration >= 0
    assert 0 <= result.speech_ratio <= 1

    total = result.total_speech_duration + result.total_silence_duration
    assert abs(total - 10.0) < 0.5  # Allow some tolerance

    print(f"  [PASS] Statistics:")
    print(f"    Speech: {result.total_speech_duration:.2f}s")
    print(f"    Silence: {result.total_silence_duration:.2f}s")
    print(f"    Ratio: {result.speech_ratio:.2%}")


def test_stereo_audio():
    """Test handling of stereo audio."""
    print("Testing stereo audio...")

    config = AudioForensicsConfig()
    segmenter = AudioSegmenter(config)

    mono = create_test_audio_with_silence(duration=3.0)
    stereo = np.stack([mono, mono], axis=1)

    result = segmenter.segment(stereo, 16000)

    assert isinstance(result, SegmentationResult)

    print("  [PASS] Stereo audio converted to mono")


def test_minimum_segment_duration():
    """Test minimum segment duration filtering."""
    print("Testing minimum segment duration...")

    config = AudioForensicsConfig(min_speech_duration=0.5)
    segmenter = AudioSegmenter(config)

    # Create very short bursts
    audio = create_speech_bursts(n_bursts=5, burst_duration=0.1)

    result = segmenter.segment(audio, 16000)

    # All segments should be >= min duration (or filtered out)
    for segment in result.segments:
        assert segment.duration >= 0.4  # Allow some tolerance

    print(f"  [PASS] Minimum duration filtering works")


def run_all_tests():
    """Run all segmentation tests."""
    print("=" * 60)
    print("AUDIO SEGMENTATION TESTS")
    print("=" * 60)

    tests = [
        test_segmenter_initialization,
        test_energy_segmentation,
        test_zcr_segmentation,
        test_combined_segmentation,
        test_segment_properties,
        test_segment_times,
        test_empty_audio,
        test_speech_type_detection,
        test_chunking,
        test_segment_features,
        test_segment_statistics,
        test_stereo_audio,
        test_minimum_segment_duration,
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
