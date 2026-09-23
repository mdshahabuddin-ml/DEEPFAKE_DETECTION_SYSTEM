"""Tests for audio forensics orchestrator.

These tests verify the main orchestrator works correctly
without modifying any existing system tests.
"""

import os
import sys
import tempfile
import wave

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from audio_forensics.audio_forensics import (
    AudioForensics,
    AudioForensicsError,
    AudioForensicsResult,
    get_audio_forensics,
)
from audio_forensics.config import AudioForensicsConfig
from audio_forensics.schemas import AudioStatus


def create_test_audio(
    duration: float = 3.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> str:
    """Create a test WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    audio = (audio * 32767).astype(np.int16)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return temp_path


def create_speech_like_audio(
    duration: float = 5.0,
    sample_rate: int = 16000,
) -> str:
    """Create speech-like audio."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t)
    )

    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    audio *= envelope

    audio = (audio * 32767).astype(np.int16)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return temp_path


def test_forensics_initialization():
    """Test forensics system initialization."""
    print("Testing forensics initialization...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    assert forensics is not None

    info = forensics.get_info()
    assert "version" in info
    assert "pipeline" in info
    assert "config" in info

    print(f"  [PASS] Forensics initialized (version={info['version']})")


def test_forensics_analyze_file():
    """Test forensics analysis with file."""
    print("Testing forensics file analysis...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_speech_like_audio(duration=3.0)

    try:
        result = forensics.analyze(audio_path, case_id="TEST-001")

        assert isinstance(result, AudioForensicsResult)
        assert result.case_id == "TEST-001"
        assert result.prediction in [
            AudioStatus.NORMAL,
            AudioStatus.LOW_SUSPICION,
            AudioStatus.SUSPICIOUS,
            AudioStatus.POTENTIAL_MANIPULATION,
            AudioStatus.UNKNOWN,
        ]
        assert 0 <= result.probability <= 1
        assert result.duration > 2.9

        print(f"  [PASS] Analysis complete: {result.prediction.value}")

    finally:
        os.unlink(audio_path)


def test_forensics_analyze_array():
    """Test forensics analysis with array."""
    print("Testing forensics array analysis...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    # Create audio array
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    result = forensics.analyze(audio, sample_rate)

    assert isinstance(result, AudioForensicsResult)
    assert result.duration > 2.9

    print(f"  [PASS] Array analysis complete")


def test_forensics_result_summary():
    """Test forensics result summary."""
    print("Testing forensics result summary...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_test_audio(duration=2.0)

    try:
        result = forensics.analyze(audio_path, case_id="TEST-SUM")

        summary = result.get_summary()

        assert isinstance(summary, str)
        assert "AUDIO FORENSIC ANALYSIS REPORT" in summary
        assert "TEST-SUM" in summary

        print(f"  [PASS] Summary generated")

    finally:
        os.unlink(audio_path)


def test_forensics_result_to_dict():
    """Test forensics result to_dict."""
    print("Testing forensics result to_dict...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_test_audio(duration=2.0)

    try:
        result = forensics.analyze(audio_path, case_id="TEST-DICT")

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "case_id" in result_dict
        assert "analysis" in result_dict
        assert "timeline" in result_dict

        print(f"  [PASS] Result converted to dict")

    finally:
        os.unlink(audio_path)


def test_forensics_result_to_json():
    """Test forensics result to_json."""
    print("Testing forensics result to_json...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_test_audio(duration=2.0)

    try:
        result = forensics.analyze(audio_path, case_id="TEST-JSON")

        json_str = result.to_json()

        assert isinstance(json_str, str)
        assert "TEST-JSON" in json_str

        # Verify it's valid JSON
        import json
        parsed = json.loads(json_str)
        assert "case_id" in parsed

        print(f"  [PASS] Result converted to JSON")

    finally:
        os.unlink(audio_path)


def test_forensics_segments():
    """Test forensics segment analysis."""
    print("Testing forensics segments...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_speech_like_audio(duration=5.0)

    try:
        segments = forensics.analyze_segments(audio_path)

        assert isinstance(segments, list)
        if segments:
            assert "id" in segments[0]
            assert "start_time" in segments[0]
            assert "prediction" in segments[0]

        print(f"  [PASS] Found {len(segments)} segments")

    finally:
        os.unlink(audio_path)


def test_forensics_timeline():
    """Test forensics timeline."""
    print("Testing forensics timeline...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_speech_like_audio(duration=5.0)

    try:
        timeline = forensics.get_timeline(audio_path)

        assert isinstance(timeline, dict)
        assert "total_duration" in timeline
        assert "segments" in timeline

        print(f"  [PASS] Timeline generated")

    finally:
        os.unlink(audio_path)


def test_forensics_compare_segments():
    """Test forensics segment comparison."""
    print("Testing forensics segment comparison...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_speech_like_audio(duration=5.0)

    try:
        comparison = forensics.compare_segments(audio_path)

        assert isinstance(comparison, dict)

        if "n_segments" in comparison:
            assert comparison["n_segments"] > 0
            assert "mean_probability" in comparison
            assert "consistency" in comparison

        print(f"  [PASS] Segment comparison complete")

    finally:
        os.unlink(audio_path)


def test_forensics_error_handling():
    """Test forensics error handling."""
    print("Testing forensics error handling...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    # Try to analyze non-existent file
    try:
        result = forensics.analyze("nonexistent.wav")
        # Should return error result or raise
        assert result is not None
    except AudioForensicsError:
        pass  # Expected

    print("  [PASS] Error handling works")


def test_forensics_custom_config():
    """Test forensics with custom config."""
    print("Testing forensics custom config...")

    config = AudioForensicsConfig(
        fake_threshold=0.7,
        suspicious_threshold=0.4,
    )
    forensics = AudioForensics(config)

    info = forensics.get_info()

    assert info["config"]["fake_threshold"] == 0.7
    assert info["config"]["suspicious_threshold"] == 0.4

    print(f"  [PASS] Custom config applied")


def test_forensics_factory():
    """Test forensics factory function."""
    print("Testing forensics factory...")

    forensics = get_audio_forensics()

    assert forensics is not None
    assert isinstance(forensics, AudioForensics)

    print("  [PASS] Factory function works")


def test_forensics_properties():
    """Test forensics result properties."""
    print("Testing forensics result properties...")

    config = AudioForensicsConfig()
    forensics = AudioForensics(config)

    audio_path = create_test_audio(duration=2.0)

    try:
        result = forensics.analyze(audio_path, case_id="TEST-PROP")

        # Test all properties
        assert result.case_id == "TEST-PROP"
        assert result.prediction is not None
        assert isinstance(result.probability, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.file_hash, str)
        assert isinstance(result.duration, float)
        assert isinstance(result.segments, list)

        print(f"  [PASS] All properties accessible")

    finally:
        os.unlink(audio_path)


def run_all_tests():
    """Run all audio forensics tests."""
    print("=" * 60)
    print("AUDIO FORENSICS ORCHESTRATOR TESTS")
    print("=" * 60)

    tests = [
        test_forensics_initialization,
        test_forensics_analyze_file,
        test_forensics_analyze_array,
        test_forensics_result_summary,
        test_forensics_result_to_dict,
        test_forensics_result_to_json,
        test_forensics_segments,
        test_forensics_timeline,
        test_forensics_compare_segments,
        test_forensics_error_handling,
        test_forensics_custom_config,
        test_forensics_factory,
        test_forensics_properties,
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
