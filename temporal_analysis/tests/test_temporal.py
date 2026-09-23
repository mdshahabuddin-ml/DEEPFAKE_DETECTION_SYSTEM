"""Tests for temporal analysis module."""

import os
import sys

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from temporal_analysis.config import TemporalConfig
from temporal_analysis.frame_sampler import FrameSampler
from temporal_analysis.feature_extractor import TemporalFeatureExtractor
from temporal_analysis.sequence_analyzer import SequenceAnalyzer
from temporal_analysis.anomaly_detector import AnomalyDetector, AnomalyResult
from temporal_analysis.temporal_analyzer import (
    TemporalAnalyzer,
    TemporalResult,
    get_temporal_analyzer,
)


def create_test_frames(
    n_frames: int = 10,
    height: int = 224,
    width: int = 224,
    consistent: bool = True,
) -> list:
    """Create test video frames."""
    frames = []
    for i in range(n_frames):
        if consistent:
            # Consistent frames
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        else:
            # Frames with sudden changes
            if i % 5 == 0:
                frame = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
            else:
                frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def create_consistent_features(n_frames: int = 20, n_features: int = 10) -> np.ndarray:
    """Create consistent feature sequence."""
    base = np.random.randn(1, n_features)
    noise = np.random.randn(n_frames, n_features) * 0.01
    return base + noise


def create_anomalous_features(n_frames: int = 20, n_features: int = 10) -> np.ndarray:
    """Create feature sequence with anomalies."""
    features = np.random.randn(n_frames, n_features) * 0.1
    # Add anomaly at frames 10-12
    features[10:13] += 5.0
    return features


def test_config_initialization():
    """Test config initialization."""
    print("Testing config initialization...")

    config = TemporalConfig()
    assert config.target_fps == 10.0
    assert config.max_frames == 100
    assert config.anomaly_threshold == 0.5

    print("  [PASS] Config initialized")


def test_frame_sampler_uniform():
    """Test uniform frame sampling."""
    print("Testing uniform frame sampling...")

    config = TemporalConfig(max_frames=10)
    sampler = FrameSampler(config)

    frames = create_test_frames(n_frames=50)
    sampled = sampler.sample_from_array(frames, method="uniform")

    assert len(sampled) <= config.max_frames
    assert len(sampled) > 0

    print(f"  [PASS] Sampled {len(sampled)} frames from 50")


def test_feature_extractor_simple():
    """Test simple feature extraction."""
    print("Testing simple feature extraction...")

    config = TemporalConfig()
    extractor = TemporalFeatureExtractor(config)

    frames = create_test_frames(n_frames=5)
    features = extractor.extract_features(frames)

    assert features.shape[0] == 5
    assert features.shape[1] > 0

    print(f"  [PASS] Features shape: {features.shape}")


def test_feature_differences():
    """Test feature difference extraction."""
    print("Testing feature differences...")

    config = TemporalConfig()
    extractor = TemporalFeatureExtractor(config)

    frames = create_test_frames(n_frames=5)
    features = extractor.extract_features(frames)
    diffs = extractor.extract_differences(features)

    assert diffs.shape[0] == 4  # n_frames - 1

    print(f"  [PASS] Differences shape: {diffs.shape}")


def test_sequence_analyzer_consistent():
    """Test sequence analyzer with consistent data."""
    print("Testing sequence analyzer consistent...")

    config = TemporalConfig()
    analyzer = SequenceAnalyzer(config)

    features = create_consistent_features(n_frames=20)
    result = analyzer.analyze_sequence(features)

    assert result["status"] == "analyzed"
    # Consistent data should have relatively high consistency
    assert result["consistency_score"] > 0.3

    print(f"  [PASS] Consistency: {result['consistency_score']:.3f}")


def test_sequence_analyzer_anomalous():
    """Test sequence analyzer with anomalous data."""
    print("Testing sequence analyzer anomalous...")

    config = TemporalConfig()
    analyzer = SequenceAnalyzer(config)

    features = create_anomalous_features(n_frames=20)
    result = analyzer.analyze_sequence(features)

    assert result["status"] == "analyzed"
    assert result["anomaly_score"] > 0

    print(f"  [PASS] Anomaly score: {result['anomaly_score']:.3f}")


def test_sequence_analyzer_insufficient():
    """Test sequence analyzer with insufficient frames."""
    print("Testing sequence analyzer insufficient...")

    config = TemporalConfig(min_frames=5)
    analyzer = SequenceAnalyzer(config)

    features = create_consistent_features(n_frames=2)
    result = analyzer.analyze_sequence(features)

    assert result["status"] == "insufficient_frames"

    print("  [PASS] Insufficient frames handled")


def test_anomaly_detector_statistical():
    """Test statistical anomaly detection."""
    print("Testing statistical anomaly detection...")

    config = TemporalConfig()
    detector = AnomalyDetector(config)

    features = create_anomalous_features(n_frames=30)
    result = detector.detect(features, method="statistical")

    assert isinstance(result, AnomalyResult)
    assert 0 <= result.anomaly_score <= 1

    print(f"  [PASS] Anomaly score: {result.anomaly_score:.3f}")


def test_anomaly_detector_threshold():
    """Test threshold anomaly detection."""
    print("Testing threshold anomaly detection...")

    config = TemporalConfig()
    detector = AnomalyDetector(config)

    features = create_anomalous_features(n_frames=30)
    result = detector.detect(features, method="threshold")

    assert isinstance(result, AnomalyResult)

    print(f"  [PASS] Threshold detection complete")


def test_anomaly_detector_segments():
    """Test anomaly segment detection."""
    print("Testing anomaly segment detection...")

    config = TemporalConfig()
    detector = AnomalyDetector(config)

    features = create_anomalous_features(n_frames=30)
    segments = detector.detect_segments(features, min_segment_length=2)

    assert isinstance(segments, list)

    print(f"  [PASS] Found {len(segments)} anomaly segments")


def test_anomaly_detector_consistent():
    """Test anomaly detector with consistent data."""
    print("Testing anomaly detector consistent...")

    config = TemporalConfig(anomaly_threshold=0.5)
    detector = AnomalyDetector(config)

    features = create_consistent_features(n_frames=30)
    result = detector.detect(features)

    assert not result.is_anomalous

    print(f"  [PASS] Consistent data not flagged as anomalous")


def test_temporal_analyzer_initialization():
    """Test temporal analyzer initialization."""
    print("Testing temporal analyzer initialization...")

    config = TemporalConfig()
    analyzer = TemporalAnalyzer(config)

    assert analyzer is not None

    print("  [PASS] Temporal analyzer initialized")


def test_temporal_analyzer_frames():
    """Test temporal analyzer with frames."""
    print("Testing temporal analyzer frames...")

    config = TemporalConfig()
    analyzer = TemporalAnalyzer(config)

    frames = create_test_frames(n_frames=15, consistent=True)
    result = analyzer.analyze_frames(frames, "test_video.mp4")

    assert isinstance(result, TemporalResult)
    assert result.n_frames == 15
    assert result.analysis_id.startswith("TEMP-")

    print(f"  [PASS] Analysis complete: suspicious={result.is_suspicious}")


def test_temporal_analyzer_anomalous_frames():
    """Test temporal analyzer with anomalous frames."""
    print("Testing temporal analyzer anomalous frames...")

    config = TemporalConfig()
    analyzer = TemporalAnalyzer(config)

    frames = create_test_frames(n_frames=20, consistent=False)
    result = analyzer.analyze_frames(frames, "anomalous_video.mp4")

    assert isinstance(result, TemporalResult)

    print(f"  [PASS] Anomalous analysis: score={result.anomaly_score:.3f}")


def test_temporal_result_summary():
    """Test temporal result summary."""
    print("Testing temporal result summary...")

    config = TemporalConfig()
    analyzer = TemporalAnalyzer(config)

    frames = create_test_frames(n_frames=10)
    result = analyzer.analyze_frames(frames, "test.mp4")

    summary = result.get_summary()

    assert "TEMPORAL ANALYSIS REPORT" in summary
    assert result.analysis_id in summary

    print("  [PASS] Summary generated")


def test_temporal_result_to_dict():
    """Test temporal result to_dict."""
    print("Testing temporal result to_dict...")

    config = TemporalConfig()
    analyzer = TemporalAnalyzer(config)

    frames = create_test_frames(n_frames=10)
    result = analyzer.analyze_frames(frames, "test.mp4")

    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert "analysis_id" in result_dict
    assert "consistency_score" in result_dict

    print("  [PASS] Result converted to dict")


def test_temporal_analyzer_factory():
    """Test temporal analyzer factory."""
    print("Testing temporal analyzer factory...")

    analyzer = get_temporal_analyzer()

    assert isinstance(analyzer, TemporalAnalyzer)

    print("  [PASS] Factory works")


def test_temporal_analyzer_array():
    """Test temporal analyzer with numpy array."""
    print("Testing temporal analyzer array...")

    config = TemporalConfig()
    analyzer = TemporalAnalyzer(config)

    video_frames = np.random.randint(0, 255, (15, 224, 224, 3), dtype=np.uint8)
    result = analyzer.analyze_array(video_frames, "array_video.mp4")

    assert isinstance(result, TemporalResult)
    assert result.n_frames > 0

    print(f"  [PASS] Array analysis complete")


def test_sudden_change_detection():
    """Test sudden change detection."""
    print("Testing sudden change detection...")

    config = TemporalConfig()
    analyzer = SequenceAnalyzer(config)

    features = create_anomalous_features(n_frames=30)
    sudden = analyzer.detect_sudden_changes(features, threshold=2.0)

    assert isinstance(sudden, list)

    print(f"  [PASS] Found {len(sudden)} sudden changes")


def run_all_tests():
    """Run all temporal analysis tests."""
    print("=" * 60)
    print("TEMPORAL ANALYSIS TESTS")
    print("=" * 60)

    tests = [
        test_config_initialization,
        test_frame_sampler_uniform,
        test_feature_extractor_simple,
        test_feature_differences,
        test_sequence_analyzer_consistent,
        test_sequence_analyzer_anomalous,
        test_sequence_analyzer_insufficient,
        test_anomaly_detector_statistical,
        test_anomaly_detector_threshold,
        test_anomaly_detector_segments,
        test_anomaly_detector_consistent,
        test_temporal_analyzer_initialization,
        test_temporal_analyzer_frames,
        test_temporal_analyzer_anomalous_frames,
        test_temporal_result_summary,
        test_temporal_result_to_dict,
        test_temporal_analyzer_factory,
        test_temporal_analyzer_array,
        test_sudden_change_detection,
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
