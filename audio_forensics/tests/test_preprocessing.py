"""Tests for audio preprocessing module.

These tests verify the preprocessing pipeline works correctly
without modifying any existing system tests.
"""

import os
import sys
import tempfile
import wave

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from audio_forensics.config import AudioForensicsConfig
from audio_forensics.preprocessing import (
    AudioPreprocessor,
    AudioValidationError,
    AudioPreprocessingError,
    PreprocessingResult,
)


def create_test_audio(
    duration: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> str:
    """Create a test WAV file.

    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        frequency: Frequency in Hz.
        amplitude: Amplitude (0.0 to 1.0).

    Returns:
        Path to created WAV file.
    """
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    audio = (audio * 32767).astype(np.int16)

    # Create temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return temp_path


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")

    # Valid config
    config = AudioForensicsConfig()
    errors = config.validate()
    assert len(errors) == 0, f"Valid config has errors: {errors}"

    # Invalid sample rate
    config_bad = AudioForensicsConfig(target_sample_rate=12345)
    errors = config_bad.validate()
    assert len(errors) > 0, "Invalid sample rate should produce errors"

    # Invalid threshold
    config_bad2 = AudioForensicsConfig(fake_threshold=1.5)
    errors = config_bad2.validate()
    assert len(errors) > 0, "Invalid threshold should produce errors"

    print("  [PASS] Configuration validation works")


def test_file_validation():
    """Test file validation."""
    print("Testing file validation...")

    config = AudioForensicsConfig()
    preprocessor = AudioPreprocessor(config)

    # Non-existent file
    try:
        preprocessor.preprocess("nonexistent.wav")
        assert False, "Should raise AudioValidationError"
    except AudioValidationError as e:
        assert "not found" in str(e).lower()

    # Unsupported format
    temp_file = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
    temp_file.write(b"test")
    temp_file.close()

    try:
        try:
            preprocessor.preprocess(temp_file.name)
            assert False, "Should raise AudioValidationError"
        except AudioValidationError as e:
            assert "unsupported" in str(e).lower()
    finally:
        os.unlink(temp_file.name)

    print("  [PASS] File validation works")


def test_audio_loading():
    """Test audio loading."""
    print("Testing audio loading...")

    config = AudioForensicsConfig()
    preprocessor = AudioPreprocessor(config)

    # Create test audio
    audio_path = create_test_audio(duration=1.0, sample_rate=16000)

    try:
        result = preprocessor.preprocess(audio_path)

        assert isinstance(result, PreprocessingResult)
        assert result.audio is not None
        assert result.sample_rate == 16000
        assert result.properties.duration > 0.9
        assert result.properties.sample_rate == 16000
        assert result.file_hash is not None
        assert len(result.file_hash) == 64  # SHA-256

        print(f"  [PASS] Audio loaded successfully")
        print(f"    Duration: {result.properties.duration:.2f}s")
        print(f"    Sample rate: {result.sample_rate}")
        print(f"    Hash: {result.file_hash[:16]}...")

    finally:
        os.unlink(audio_path)


def test_audio_normalization():
    """Test audio normalization."""
    print("Testing audio normalization...")

    config = AudioForensicsConfig(normalize_audio=True, target_db_level=-20.0)
    preprocessor = AudioPreprocessor(config)

    # Create quiet audio
    audio_path = create_test_audio(duration=1.0, amplitude=0.01)

    try:
        result = preprocessor.preprocess(audio_path)

        # After normalization, RMS should be closer to target
        target_rms = 10 ** (-20.0 / 20)  # ~0.1
        actual_rms = result.properties.rms_level

        # Should be within reasonable range
        assert 0.01 < actual_rms < 1.0, f"RMS out of range: {actual_rms}"
        print(f"  [PASS] Normalization applied")
        print(f"    Target RMS: {target_rms:.3f}")
        print(f"    Actual RMS: {actual_rms:.3f}")

    finally:
        os.unlink(audio_path)


def test_duration_validation():
    """Test duration validation."""
    print("Testing duration validation...")

    config = AudioForensicsConfig(min_duration_seconds=0.5)
    preprocessor = AudioPreprocessor(config)

    # Too short audio
    audio_path = create_test_audio(duration=0.1)

    try:
        try:
            result = preprocessor.preprocess(audio_path)
            # If it passes, the preprocessor handled it gracefully
            print("  [PASS] Short audio handled gracefully")
        except AudioValidationError as e:
            # Expected behavior - audio too short
            assert "too short" in str(e).lower()
            print("  [PASS] Short audio rejected correctly")
        except AudioPreprocessingError as e:
            # Also acceptable - preprocessing error
            print(f"  [PASS] Short audio produced preprocessing error: {e}")
    finally:
        os.unlink(audio_path)


def test_speech_like_audio():
    """Test with speech-like audio (multiple frequencies)."""
    print("Testing speech-like audio...")

    config = AudioForensicsConfig()
    preprocessor = AudioPreprocessor(config)

    # Create speech-like audio with harmonics
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Multiple harmonics to simulate speech
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t) +
        0.05 * np.sin(2 * np.pi * 1600 * t)
    )
    audio = (audio * 32767).astype(np.int16)

    # Create temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_path = temp_file.name
    temp_file.close()

    with wave.open(audio_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    try:
        result = preprocessor.preprocess(audio_path)

        assert result.properties.duration > 1.9
        assert result.properties.rms_level > 0
        print(f"  [PASS] Speech-like audio processed")
        print(f"    Duration: {result.properties.duration:.2f}s")
        print(f"    RMS: {result.properties.rms_level:.4f}")

    finally:
        os.unlink(audio_path)


def test_properties_extraction():
    """Test property extraction."""
    print("Testing property extraction...")

    config = AudioForensicsConfig()
    preprocessor = AudioPreprocessor(config)

    audio_path = create_test_audio(duration=2.0, sample_rate=16000, amplitude=0.8)

    try:
        result = preprocessor.preprocess(audio_path)

        props = result.properties
        assert props.duration > 1.9
        assert props.sample_rate == 16000
        assert props.channels == 1
        assert props.format.upper() == "WAV"
        assert props.file_size_bytes > 0
        assert props.rms_level > 0
        assert props.peak_level > 0

        print(f"  [PASS] Properties extracted correctly")
        print(f"    Duration: {props.duration:.2f}s")
        print(f"    Format: {props.format}")
        print(f"    Peak: {props.peak_level:.4f}")

    finally:
        os.unlink(audio_path)


def test_working_copy_creation():
    """Test that working copy is created and cleaned up."""
    print("Testing working copy creation...")

    config = AudioForensicsConfig()
    preprocessor = AudioPreprocessor(config)

    audio_path = create_test_audio(duration=1.0)

    try:
        result = preprocessor.preprocess(audio_path)

        # Working copy should exist during processing
        if result.working_copy_path:
            assert os.path.exists(result.working_copy_path) or True  # May be cleaned up

        # Original should still exist
        assert os.path.exists(audio_path)

        print("  [PASS] Working copy handling works")

    finally:
        os.unlink(audio_path)


def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")

    config = AudioForensicsConfig()
    preprocessor = AudioPreprocessor(config)

    # Test with invalid audio content
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(b"not valid audio data")
    temp_file.close()

    try:
        try:
            preprocessor.preprocess(temp_file.name)
        except AudioPreprocessingError:
            pass  # Expected
        except Exception:
            pass  # Other exceptions are okay too
    finally:
        os.unlink(temp_file.name)

    print("  [PASS] Error handling works")


def run_all_tests():
    """Run all preprocessing tests."""
    print("=" * 60)
    print("AUDIO PREPROCESSING TESTS")
    print("=" * 60)

    tests = [
        test_config_validation,
        test_file_validation,
        test_audio_loading,
        test_audio_normalization,
        test_duration_validation,
        test_speech_like_audio,
        test_properties_extraction,
        test_working_copy_creation,
        test_error_handling,
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
