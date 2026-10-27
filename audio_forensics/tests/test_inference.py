"""Tests for audio inference pipeline.

These tests verify the complete inference pipeline works correctly
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
from audio_forensics.detector import (
    AudioCNNModel,
    AudioCRNNModel,
    AudioDetector,
    DetectionResult,
)
from audio_forensics.inference import AudioInferencePipeline, InferenceError
from audio_forensics.schemas import AudioAnalysisResult, AudioStatus


def create_test_audio(
    duration: float = 3.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> str:
    """Create a test WAV file and return path."""
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
    """Create speech-like audio WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Multiple harmonics
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t) +
        0.05 * np.sin(2 * np.pi * 1600 * t)
    )

    # Add modulation to simulate speech envelope
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


def test_detector_initialization():
    """Test detector initialization."""
    print("Testing detector initialization...")

    config = AudioForensicsConfig()
    detector = AudioDetector(config)

    assert detector is not None
    assert not detector.is_model_loaded

    print("  [PASS] Detector initialized")


def test_detector_rule_based():
    """Test rule-based detection without model."""
    print("Testing rule-based detection...")

    config = AudioForensicsConfig()
    detector = AudioDetector(config)

    # Create test features
    features = np.random.randn(128, 64).astype(np.float32)

    result = detector.detect(features)

    assert isinstance(result, DetectionResult)
    assert result.prediction in [
        AudioStatus.NORMAL,
        AudioStatus.LOW_SUSPICION,
        AudioStatus.SUSPICIOUS,
        AudioStatus.POTENTIAL_MANIPULATION,
        AudioStatus.REQUIRES_REVIEW,
    ]
    assert 0 <= result.probability <= 1

    print(f"  [PASS] Rule-based detection: {result.prediction.value}")


def test_detector_empty_features():
    """Test detector with empty features."""
    print("Testing empty features...")

    config = AudioForensicsConfig()
    detector = AudioDetector(config)

    result = detector.detect(np.array([]))

    assert isinstance(result, DetectionResult)
    assert result.prediction == AudioStatus.UNKNOWN

    print("  [PASS] Empty features handled correctly")


def test_neural_models_are_nn_modules():
    """CNN/CRNN models must be real torch Modules (callable, state_dict-able)."""
    import torch
    import torch.nn as nn

    for cls in (AudioCNNModel, AudioCRNNModel):
        model = cls(n_mels=64, n_classes=2)
        assert isinstance(model, nn.Module), f"{cls.__name__} is not nn.Module"
        assert callable(model), f"{cls.__name__} is not callable"
        assert len(list(model.parameters())) > 0, f"{cls.__name__} has no parameters"
        # state_dict roundtrip (what load_state_dict in load_model expects)
        assert all(isinstance(v, torch.Tensor) for v in model.state_dict().values())

    print("  [PASS] CNN/CRNN are real nn.Modules")


def test_neural_models_forward_pass():
    """Forward pass on a log-mel-like input yields (batch, n_classes) logits."""
    import torch

    batch, n_mels, n_frames = 2, 64, 50

    cnn = AudioCNNModel(n_mels=n_mels, n_classes=2)
    cnn.eval()
    with torch.no_grad():
        out = cnn(torch.randn(batch, n_mels, n_frames))
    assert out.shape == (batch, 2), f"CNN output shape {out.shape}"

    crnn = AudioCRNNModel(n_mels=n_mels, n_classes=2)
    crnn.eval()
    with torch.no_grad():
        out = crnn(torch.randn(batch, n_mels, n_frames))
    assert out.shape == (batch, 2), f"CRNN output shape {out.shape}"

    print("  [PASS] CNN/CRNN forward pass shapes correct")


def test_detector_load_model_roundtrip(tmp_path):
    """Save a model state_dict and load it back through AudioDetector.load_model."""
    import torch

    config = AudioForensicsConfig(n_mels=64)
    detector = AudioDetector(config)

    model = AudioCNNModel(n_mels=64, n_classes=2)
    ckpt = tmp_path / "cnn_ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)

    from audio_forensics.detector import ModelArchitecture
    ok = detector.load_model(str(ckpt), architecture=ModelArchitecture.CNN)
    assert ok is True, "load_model failed on a valid checkpoint"
    assert detector.is_model_loaded is True

    # End-to-end: detect() must now route through the neural model
    features = np.random.randn(64, 50).astype(np.float32)
    result = detector.detect(features)
    assert result.metadata.get("method") == "neural_network"
    assert 0.0 <= result.probability <= 1.0

    print("  [PASS] Model checkpoint roundtrip + neural inference")


def test_detector_load_model_missing_checkpoint(tmp_path):
    """A missing checkpoint must return False (rules fallback), not raise."""
    config = AudioForensicsConfig()
    detector = AudioDetector(config)

    from audio_forensics.detector import ModelArchitecture
    ok = detector.load_model(
        str(tmp_path / "does_not_exist.pt"),
        architecture=ModelArchitecture.CNN,
    )
    assert ok is False
    assert detector.is_model_loaded is False

    print("  [PASS] Missing checkpoint handled gracefully")


def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("Testing pipeline initialization...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    assert pipeline is not None
    assert pipeline.is_ready

    info = pipeline.get_info()
    assert "model_loaded" in info
    assert "architecture" in info

    print(f"  [PASS] Pipeline initialized (model_loaded={info['model_loaded']})")


def test_pipeline_analyze_file():
    """Test complete pipeline analysis."""
    print("Testing complete pipeline analysis...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    # Create test audio
    audio_path = create_speech_like_audio(duration=3.0)

    try:
        result = pipeline.analyze(audio_path)

        assert isinstance(result, AudioAnalysisResult)
        assert result.analysis_id.startswith("AUD-")
        assert result.file_hash is not None
        assert result.properties.duration > 2.9
        assert result.overall_prediction in [
            AudioStatus.NORMAL,
            AudioStatus.LOW_SUSPICION,
            AudioStatus.SUSPICIOUS,
            AudioStatus.POTENTIAL_MANIPULATION,
            AudioStatus.UNKNOWN,
        ]
        assert 0 <= result.overall_probability <= 1
        assert result.processing_time > 0

        print(f"  [PASS] Pipeline analysis complete")
        print(f"    Prediction: {result.overall_prediction.value}")
        print(f"    Probability: {result.overall_probability:.3f}")
        print(f"    Segments: {result.total_segments}")
        print(f"    Time: {result.processing_time:.2f}s")

    finally:
        os.unlink(audio_path)


def test_pipeline_analyze_array():
    """Test pipeline analysis with audio array."""
    print("Testing pipeline array analysis...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    # Create audio array
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    result = pipeline.analyze_audio_array(audio, sample_rate)

    assert isinstance(result, AudioAnalysisResult)
    assert result.properties.duration > 2.9
    assert result.properties.sample_rate == sample_rate

    print(f"  [PASS] Array analysis complete")
    print(f"    Prediction: {result.overall_prediction.value}")


def test_pipeline_segments():
    """Test pipeline segment results."""
    print("Testing pipeline segments...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    audio_path = create_speech_like_audio(duration=5.0)

    try:
        result = pipeline.analyze(audio_path, return_segments=True)

        assert result.total_segments > 0
        assert len(result.segments) > 0

        # Check segment properties
        for segment in result.segments:
            assert segment.start_time >= 0
            assert segment.end_time > segment.start_time
            assert segment.duration > 0
            assert 0 <= segment.probability <= 1

        print(f"  [PASS] Found {result.total_segments} segments")

    finally:
        os.unlink(audio_path)


def test_pipeline_timeline():
    """Test pipeline timeline generation."""
    print("Testing pipeline timeline...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    audio_path = create_speech_like_audio(duration=5.0)

    try:
        result = pipeline.analyze(audio_path)

        assert len(result.timeline) > 0

        for entry in result.timeline:
            assert "start_time" in entry
            assert "end_time" in entry
            assert "prediction" in entry
            assert "probability" in entry

        print(f"  [PASS] Timeline has {len(result.timeline)} entries")

    finally:
        os.unlink(audio_path)


def test_pipeline_evidence():
    """Test pipeline evidence generation."""
    print("Testing pipeline evidence...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    audio_path = create_speech_like_audio(duration=3.0)

    try:
        result = pipeline.analyze(audio_path)

        # Evidence should be a list
        assert isinstance(result.evidence, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.recommendations, list)

        print(f"  [PASS] Evidence: {len(result.evidence)} items")
        print(f"    Warnings: {len(result.warnings)}")
        print(f"    Recommendations: {len(result.recommendations)}")

    finally:
        os.unlink(audio_path)


def test_pipeline_result_to_dict():
    """Test result serialization."""
    print("Testing result to_dict...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    audio_path = create_test_audio(duration=2.0)

    try:
        result = pipeline.analyze(audio_path)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "analysis_id" in result_dict
        assert "overall_prediction" in result_dict
        assert "segments" in result_dict

        print(f"  [PASS] Result serialized to dict")

    finally:
        os.unlink(audio_path)


def test_pipeline_result_summary():
    """Test result summary generation."""
    print("Testing result summary...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    audio_path = create_test_audio(duration=2.0)

    try:
        result = pipeline.analyze(audio_path)

        summary = result.get_summary()

        assert isinstance(summary, str)
        assert "Audio Analysis Summary" in summary
        assert result.file_path in summary

        print(f"  [PASS] Summary generated")

    finally:
        os.unlink(audio_path)


def test_pipeline_single_segment():
    """Test pipeline with single segment audio."""
    print("Testing single segment audio...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    audio_path = create_test_audio(duration=1.0)

    try:
        result = pipeline.analyze(audio_path)

        assert isinstance(result, AudioAnalysisResult)
        # May have 0 or 1 segments depending on VAD

        print(f"  [PASS] Single segment handled")

    finally:
        os.unlink(audio_path)


def test_pipeline_no_segments():
    """Test pipeline with silent audio."""
    print("Testing silent audio...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    # Create silent audio
    sample_rate = 16000
    duration = 2.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    try:
        result = pipeline.analyze(temp_path)

        # Should handle gracefully
        assert isinstance(result, AudioAnalysisResult)

        print(f"  [PASS] Silent audio handled gracefully")

    finally:
        os.unlink(temp_path)


def test_pipeline_error_handling():
    """Test pipeline error handling."""
    print("Testing error handling...")

    config = AudioForensicsConfig()
    pipeline = AudioInferencePipeline(config)

    # Try to analyze non-existent file
    result = pipeline.analyze("nonexistent.wav")

    # Should return error result, not crash
    assert isinstance(result, AudioAnalysisResult)
    assert result.overall_prediction == AudioStatus.REQUIRES_REVIEW

    print("  [PASS] Error handling works")


def test_detector_model_info():
    """Test detector model info."""
    print("Testing detector model info...")

    config = AudioForensicsConfig()
    detector = AudioDetector(config)

    info = detector.get_model_info()

    assert isinstance(info, dict)
    assert "architecture" in info
    assert "is_loaded" in info
    assert "device" in info
    assert "thresholds" in info

    print(f"  [PASS] Model info: {info['architecture']}")


def run_all_tests():
    """Run all inference tests."""
    print("=" * 60)
    print("AUDIO INFERENCE PIPELINE TESTS")
    print("=" * 60)

    tests = [
        test_detector_initialization,
        test_detector_rule_based,
        test_detector_empty_features,
        test_neural_models_are_nn_modules,
        test_neural_models_forward_pass,
        test_detector_load_model_roundtrip,
        test_detector_load_model_missing_checkpoint,
        test_pipeline_initialization,
        test_pipeline_analyze_file,
        test_pipeline_analyze_array,
        test_pipeline_segments,
        test_pipeline_timeline,
        test_pipeline_evidence,
        test_pipeline_result_to_dict,
        test_pipeline_result_summary,
        test_pipeline_single_segment,
        test_pipeline_no_segments,
        test_pipeline_error_handling,
        test_detector_model_info,
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
