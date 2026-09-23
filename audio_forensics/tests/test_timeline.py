"""Tests for audio forensic timeline module.

These tests verify the timeline generation works correctly
without modifying any existing system tests.
"""

import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from audio_forensics.config import AudioForensicsConfig
from audio_forensics.schemas import (
    AudioAnalysisResult,
    AudioProperties,
    AudioStatus,
    SegmentResult,
    SpeechType,
)
from audio_forensics.timeline import (
    AudioForensicTimeline,
    AudioForensicTimelineGenerator,
)


def create_mock_analysis_result(
    n_segments: int = 5,
    duration: float = 30.0,
) -> AudioAnalysisResult:
    """Create a mock analysis result for testing."""
    segments = []

    for i in range(n_segments):
        start_time = i * (duration / n_segments)
        end_time = (i + 1) * (duration / n_segments)

        # Alternate predictions
        if i % 3 == 0:
            prediction = AudioStatus.SUSPICIOUS
            probability = 0.7
        elif i % 3 == 1:
            prediction = AudioStatus.NORMAL
            probability = 0.2
        else:
            prediction = AudioStatus.LOW_SUSPICION
            probability = 0.4

        segment = SegmentResult(
            segment_id=i,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            speech_type=SpeechType.SPEECH,
            prediction=prediction,
            probability=probability,
            confidence=0.8,
        )
        segments.append(segment)

    return AudioAnalysisResult(
        file_path="test_audio.wav",
        file_hash="abc123",
        analysis_id="AUD-TEST123",
        properties=AudioProperties(
            duration=duration,
            sample_rate=16000,
            channels=1,
        ),
        overall_prediction=AudioStatus.SUSPICIOUS,
        overall_probability=0.5,
        overall_confidence=0.7,
        segments=segments,
        total_segments=n_segments,
        suspicious_segments=2,
    )


def create_mock_result_no_segments() -> AudioAnalysisResult:
    """Create mock result with no segments."""
    return AudioAnalysisResult(
        file_path="silent_audio.wav",
        analysis_id="AUD-SILENT",
        properties=AudioProperties(
            duration=5.0,
            sample_rate=16000,
        ),
        overall_prediction=AudioStatus.UNKNOWN,
        overall_probability=0.0,
        overall_confidence=0.0,
        segments=[],
        total_segments=0,
    )


def test_timeline_generator_initialization():
    """Test timeline generator initialization."""
    print("Testing timeline generator initialization...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    assert generator is not None
    assert generator.config is config

    print("  [PASS] Timeline generator initialized")


def test_create_timeline():
    """Test timeline creation."""
    print("Testing timeline creation...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_analysis_result(n_segments=5)
    timeline = generator.create_timeline(result)

    assert isinstance(timeline, AudioForensicTimeline)
    assert timeline.total_duration == 30.0
    assert len(timeline.segments) == 5
    assert len(timeline.summary) > 0

    print(f"  [PASS] Timeline created with {len(timeline.segments)} segments")


def test_timeline_segment_properties():
    """Test timeline segment properties."""
    print("Testing timeline segment properties...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_analysis_result(n_segments=3)
    timeline = generator.create_timeline(result)

    for segment in timeline.segments:
        assert "start_time" in segment
        assert "end_time" in segment
        assert "status" in segment
        assert "probability" in segment
        assert "color" in segment
        assert "label" in segment

        # Check color is valid hex
        assert segment["color"].startswith("#")
        assert len(segment["color"]) == 7

    print("  [PASS] Segment properties valid")


def test_timeline_summary():
    """Test timeline summary generation."""
    print("Testing timeline summary...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_analysis_result(n_segments=5)
    timeline = generator.create_timeline(result)

    summary = timeline.summary
    assert "total_segments" in summary
    assert "suspicious_segments" in summary
    assert "mean_probability" in summary
    assert "max_probability" in summary
    assert "anomaly_score" in summary

    assert summary["total_segments"] == 5
    assert 0 <= summary["anomaly_score"] <= 1

    print(f"  [PASS] Summary: anomaly_score={summary['anomaly_score']:.3f}")


def test_timeline_no_segments():
    """Test timeline with no segments."""
    print("Testing timeline with no segments...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_result_no_segments()
    timeline = generator.create_timeline(result)

    assert isinstance(timeline, AudioForensicTimeline)
    assert len(timeline.segments) == 1  # Single entry for whole audio
    assert timeline.segments[0]["status"] == "UNKNOWN"

    print("  [PASS] No segments handled correctly")


def test_timeline_to_dict():
    """Test timeline to_dict conversion."""
    print("Testing timeline to_dict...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_analysis_result(n_segments=3)
    timeline = generator.create_timeline(result)

    timeline_dict = timeline.to_dict()

    assert isinstance(timeline_dict, dict)
    assert "total_duration" in timeline_dict
    assert "segments" in timeline_dict
    assert "summary" in timeline_dict
    assert "colors" in timeline_dict

    print("  [PASS] Timeline converted to dict")


def test_ascii_timeline():
    """Test ASCII timeline generation."""
    print("Testing ASCII timeline...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_analysis_result(n_segments=5)
    ascii_timeline = generator.create_ascii_timeline(result)

    assert isinstance(ascii_timeline, str)
    assert "AUDIO FORENSIC TIMELINE" in ascii_timeline
    assert "Legend:" in ascii_timeline
    assert "Segment Details:" in ascii_timeline

    print(f"  [PASS] ASCII timeline generated ({len(ascii_timeline)} chars)")


def test_ascii_timeline_no_segments():
    """Test ASCII timeline with no segments."""
    print("Testing ASCII timeline no segments...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_result_no_segments()
    ascii_timeline = generator.create_ascii_timeline(result)

    assert isinstance(ascii_timeline, str)

    print("  [PASS] ASCII timeline handled no segments")


def test_html_timeline():
    """Test HTML timeline generation."""
    print("Testing HTML timeline...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_analysis_result(n_segments=3)
    html_timeline = generator.create_html_timeline(result)

    assert isinstance(html_timeline, str)
    assert "<!DOCTYPE html>" in html_timeline
    assert "timeline-container" in html_timeline
    assert "timeline-segment" in html_timeline

    print(f"  [PASS] HTML timeline generated ({len(html_timeline)} chars)")


def test_timeline_colors():
    """Test timeline color mapping."""
    print("Testing timeline colors...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    # Check all status colors exist
    expected_statuses = [
        "NORMAL", "LOW_SUSPICION", "SUSPICIOUS",
        "POTENTIAL_MANIPULATION", "REQUIRES_REVIEW", "UNKNOWN"
    ]

    for status in expected_statuses:
        assert status in generator.STATUS_COLORS
        assert generator.STATUS_COLORS[status].startswith("#")

    print("  [PASS] All status colors defined")


def test_timeline_status_labels():
    """Test timeline status labels."""
    print("Testing timeline status labels...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    # Check all status labels exist
    expected_statuses = [
        "NORMAL", "LOW_SUSPICION", "SUSPICIOUS",
        "POTENTIAL_MANIPULATION", "REQUIRES_REVIEW", "UNKNOWN"
    ]

    for status in expected_statuses:
        assert status in generator.STATUS_LABELS
        assert isinstance(generator.STATUS_LABELS[status], str)

    print("  [PASS] All status labels defined")


def test_timeline_probability_stats():
    """Test probability statistics in summary."""
    print("Testing probability stats...")

    config = AudioForensicsConfig()
    generator = AudioForensicTimelineGenerator(config)

    result = create_mock_analysis_result(n_segments=5)
    timeline = generator.create_timeline(result)

    summary = timeline.summary

    # Check probability stats are reasonable
    assert 0 <= summary["mean_probability"] <= 1
    assert 0 <= summary["max_probability"] <= 1
    assert summary["max_probability"] >= summary["mean_probability"]

    print(f"  [PASS] Mean prob: {summary['mean_probability']:.3f}, "
          f"Max prob: {summary['max_probability']:.3f}")


def run_all_tests():
    """Run all timeline tests."""
    print("=" * 60)
    print("AUDIO FORENSIC TIMELINE TESTS")
    print("=" * 60)

    tests = [
        test_timeline_generator_initialization,
        test_create_timeline,
        test_timeline_segment_properties,
        test_timeline_summary,
        test_timeline_no_segments,
        test_timeline_to_dict,
        test_ascii_timeline,
        test_ascii_timeline_no_segments,
        test_html_timeline,
        test_timeline_colors,
        test_timeline_status_labels,
        test_timeline_probability_stats,
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
