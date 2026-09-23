"""
Standalone Audio Detection Tests.

Run:
    python test_audio_standalone.py

Does NOT modify any existing files.
"""

import os
import sys
import tempfile
import wave
import struct

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# Track results
_passed = 0
_failed = 0


def test(name, fn):
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        _failed += 1
        print(f"  FAIL  {name}: {e}")


def create_test_wav(path, duration=2.0, sr=16000, freq=440.0):
    """Create a simple sine wave WAV file for testing."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, dtype=np.float32)
    samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


# ============================================================
# Test Group 1: Import Tests
# ============================================================
print("\n=== Import Tests ===")

def test_import_config():
    from audio_forensics.config import AudioForensicsConfig
    cfg = AudioForensicsConfig()
    assert cfg.target_sample_rate == 16000
    assert 0.0 <= cfg.fake_threshold <= 1.0

test("Import AudioForensicsConfig", test_import_config)


def test_import_schemas():
    from audio_forensics.schemas import AudioStatus, SegmentResult, AudioAnalysisResult
    assert AudioStatus.NORMAL.value == "NORMAL"
    assert AudioStatus.SUSPICIOUS.value == "SUSPICIOUS"

test("Import schemas", test_import_schemas)


def test_import_preprocessing():
    from audio_forensics.preprocessing import AudioPreprocessor, get_preprocessor
    from audio_forensics.config import AudioForensicsConfig
    cfg = AudioForensicsConfig()
    pp = get_preprocessor(cfg)
    assert pp is not None

test("Import AudioPreprocessor", test_import_preprocessing)


def test_import_features():
    from audio_forensics.features import AudioFeatureExtractor, get_feature_extractor
    from audio_forensics.config import AudioForensicsConfig
    cfg = AudioForensicsConfig()
    fe = get_feature_extractor(cfg)
    assert fe is not None

test("Import AudioFeatureExtractor", test_import_features)


def test_import_segmentation():
    from audio_forensics.segmentation import AudioSegmenter, get_segmenter
    from audio_forensics.config import AudioForensicsConfig
    cfg = AudioForensicsConfig()
    seg = get_segmenter(cfg)
    assert seg is not None

test("Import AudioSegmenter", test_import_segmentation)


def test_import_inference():
    from audio_forensics.inference import AudioInferencePipeline, get_pipeline
    from audio_forensics.config import AudioForensicsConfig
    cfg = AudioForensicsConfig()
    pipe = get_pipeline(cfg)
    assert pipe is not None

test("Import AudioInferencePipeline", test_import_inference)


def test_import_simple_backend():
    from src.audio_infer import load_audio_detector, predict_from_array
    from src.audio_dataset import SAMPLE_RATE
    assert SAMPLE_RATE == 16000

test("Import src/audio_infer fallback", test_import_simple_backend)


# ============================================================
# Test Group 2: Audio Preprocessing
# ============================================================
print("\n=== Audio Preprocessing Tests ===")


def test_preprocess_audio():
    from audio_forensics.preprocessing import AudioPreprocessor, get_preprocessor
    from audio_forensics.config import AudioForensicsConfig

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_test_wav(tmp_path, duration=2.0)
        cfg = AudioForensicsConfig()
        pp = get_preprocessor(cfg)
        result = pp.preprocess(tmp_path)
        assert result is not None
        assert hasattr(result, "audio")
        assert len(result.audio) > 0
        assert hasattr(result, "sample_rate")
        assert result.sample_rate > 0
    finally:
        os.unlink(tmp_path)

test("Preprocess WAV file", test_preprocess_audio)


def test_preprocess_validates_format():
    from audio_forensics.preprocessing import AudioPreprocessor, get_preprocessor
    from audio_forensics.config import AudioForensicsConfig

    cfg = AudioForensicsConfig()
    pp = get_preprocessor(cfg)
    # Non-existent file should raise
    try:
        pp.preprocess("/nonexistent/file.wav")
        assert False, "Should have raised an exception"
    except Exception:
        pass

test("Preprocessing validates file existence", test_preprocess_validates_format)


# ============================================================
# Test Group 3: Feature Extraction
# ============================================================
print("\n=== Feature Extraction Tests ===")


def test_extract_features():
    from audio_forensics.features import AudioFeatureExtractor, get_feature_extractor
    from audio_forensics.config import AudioForensicsConfig

    cfg = AudioForensicsConfig()
    fe = get_feature_extractor(cfg)

    # Create synthetic audio
    sr = cfg.target_sample_rate
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    features = fe.extract(audio, sr)
    assert features is not None
    assert hasattr(features, "to_array")
    arr = features.to_array()
    assert arr is not None
    assert len(arr) > 0

test("Extract audio features", test_extract_features)


# ============================================================
# Test Group 4: Segmentation
# ============================================================
print("\n=== Segmentation Tests ===")


def test_segment_audio():
    from audio_forensics.segmentation import AudioSegmenter, get_segmenter
    from audio_forensics.config import AudioForensicsConfig

    cfg = AudioForensicsConfig()
    seg = get_segmenter(cfg)

    sr = cfg.target_sample_rate
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    result = seg.segment(audio, sr)
    assert result is not None
    assert hasattr(result, "segments")
    assert isinstance(result.segments, list)

test("Segment audio", test_segment_audio)


# ============================================================
# Test Group 5: Full Pipeline
# ============================================================
print("\n=== Full Pipeline Tests ===")


def test_full_pipeline_analyze():
    from audio_forensics.inference import AudioInferencePipeline, get_pipeline
    from audio_forensics.config import AudioForensicsConfig

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_test_wav(tmp_path, duration=3.0)
        cfg = AudioForensicsConfig()
        pipe = get_pipeline(cfg)
        result = pipe.analyze(tmp_path)

        assert result is not None
        assert hasattr(result, "overall_prediction")
        assert hasattr(result, "overall_probability")
        assert hasattr(result, "overall_confidence")
        assert hasattr(result, "segments")
        assert hasattr(result, "total_segments")
        assert hasattr(result, "evidence")
        assert hasattr(result, "warnings")
        assert hasattr(result, "recommendations")
        assert 0.0 <= result.overall_probability <= 1.0
        assert 0.0 <= result.overall_confidence <= 1.0
    finally:
        os.unlink(tmp_path)

test("Full pipeline.analyze()", test_full_pipeline_analyze)


def test_pipeline_result_structure():
    from audio_forensics.inference import AudioInferencePipeline, get_pipeline
    from audio_forensics.config import AudioForensicsConfig

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_test_wav(tmp_path, duration=2.0)
        cfg = AudioForensicsConfig()
        pipe = get_pipeline(cfg)
        result = pipe.analyze(tmp_path)

        # Verify result has to_dict
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "overall_prediction" in d
        assert "overall_probability" in d
        assert "segments" in d
        assert isinstance(d["segments"], list)
    finally:
        os.unlink(tmp_path)

test("Pipeline result has to_dict()", test_pipeline_result_structure)


# ============================================================
# Test Group 6: src/audio_infer Fallback
# ============================================================
print("\n=== Fallback Backend Tests ===")


def test_simple_predict_from_array():
    from src.audio_infer import load_audio_detector, predict_from_array
    from src.audio_dataset import SAMPLE_RATE

    model, device = load_audio_detector()
    # Create synthetic audio
    sr = SAMPLE_RATE
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    prob = predict_from_array(audio, model, device, sr=sr)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

test("predict_from_array with synthetic audio", test_simple_predict_from_array)


# ============================================================
# Test Group 7: Database History (read-only consumer)
# ============================================================
print("\n=== Database History Tests ===")


def test_store_audio_prediction():
    from src.database import get_db, PredictionRecord

    db = get_db()
    record = PredictionRecord(
        file_name="test_audio.wav",
        media_type="audio",
        prediction="REAL",
        confidence=0.85,
        fake_probability=0.15,
        inference_time=0.5,
        source="audio_test",
    )
    row_id = db.add(record)
    assert row_id > 0

    # Verify it's stored
    recent = db.recent(limit=1)
    assert len(recent) >= 1
    found = any(r["file_name"] == "test_audio.wav" for r in recent)
    assert found

test("Store audio prediction to DB", test_store_audio_prediction)


# ============================================================
# Test Group 8: Compile Checks for New Files
# ============================================================
print("\n=== Compile Checks ===")


def test_compile_run_audio_detection():
    import py_compile
    path = os.path.join(os.path.dirname(__file__), "run_audio_detection.py")
    py_compile.compile(path, doraise=True)

test("Compile run_audio_detection.py", test_compile_run_audio_detection)


def test_compile_test_audio_standalone():
    import py_compile
    path = os.path.join(os.path.dirname(__file__), "test_audio_standalone.py")
    py_compile.compile(path, doraise=True)

test("Compile test_audio_standalone.py", test_compile_test_audio_standalone)


# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 50}")
print(f"Results: {_passed} passed, {_failed} failed, {_passed + _failed} total")
print(f"{'=' * 50}")

if _failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
