"""
Audio Detection Pipeline Profiler
===================================
Profiles each stage of the audio detection pipeline on a synthetic 1-minute MP3.
No code changes -- read-only diagnostics.
"""

import os
import sys
import io
import time
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Fix Windows console encoding for Unicode box-drawing chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# STAGE 0: Generate synthetic 1-minute audio file
# ============================================================

def generate_test_audio(duration_sec=60, sr=16000, out_path=None):
    """Generate a 1-minute synthetic speech-like WAV for testing."""
    if out_path is None:
        out_path = os.path.join(tempfile.gettempdir(), "test_audio_60s.wav")

    try:
        import scipy.io.wavfile as wavfile
        # Generate speech-like signal: mixed sine waves with noise
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        # Simulate speech formants
        signal = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 400 * t) +
            0.1 * np.sin(2 * np.pi * 800 * t) +
            0.05 * np.sin(2 * np.pi * 1200 * t) +
            0.02 * np.random.randn(len(t))
        )
        # Add amplitude modulation (simulates speech rhythm)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        signal *= envelope
        signal = signal.astype(np.float32)
        # Normalize
        signal = signal / (np.abs(signal).max() + 1e-8) * 0.9
        # Convert to int16 for WAV
        signal_int16 = (signal * 32767).astype(np.int16)
        wavfile.write(out_path, sr, signal_int16)
        return out_path
    except Exception as e:
        print(f"  [WARN] Could not generate audio: {e}")
        return None


# ============================================================
# STAGE 1: Profile Backend A (audio_forensics)
# ============================================================

def profile_forensics_backend(audio_path):
    """Profile the audio_forensics pipeline stage by stage."""
    print("\n" + "=" * 70)
    print("  BACKEND A: audio_forensics/ (Segment-Level Forensics)")
    print("=" * 70)

    timings = {}

    # --- 0. Pipeline initialization (model loading) ---
    t_start = time.time()
    try:
        from audio_forensics.config import AudioForensicsConfig
        from audio_forensics.inference import AudioInferencePipeline
        config = AudioForensicsConfig(fake_threshold=0.47)
        pipeline = AudioInferencePipeline(config)
        model_loaded = pipeline.model_loaded
    except Exception as e:
        print(f"  [ERROR] Pipeline init failed: {e}")
        return None
    timings["0_pipeline_init"] = time.time() - t_start

    print(f"\n  Pipeline initialized: {timings['0_pipeline_init']:.3f}s")
    print(f"  Model loaded: {model_loaded}")
    print(f"  Device: {pipeline._detector._device}")

    # --- Profile individual stages manually ---
    # Re-run with timing instrumentation

    # Stage 1: Preprocessing
    t0 = time.time()
    preprocess_result = pipeline._preprocessor.preprocess(audio_path, None)
    timings["1_preprocessing"] = time.time() - t0

    audio_data = preprocess_result.audio
    sr = preprocess_result.sample_rate
    duration = len(audio_data) / sr

    print(f"\n  Audio loaded: {duration:.1f}s, {sr}Hz, {len(audio_data)} samples")
    print(f"  Stage 1 - Preprocessing:     {timings['1_preprocessing']:.3f}s")

    # Stage 2: Segmentation
    t0 = time.time()
    seg_result = pipeline._segmenter.segment(audio_data, sr)
    timings["2_segmentation"] = time.time() - t0

    n_segments = len(seg_result.segments)
    print(f"  Stage 2 - Segmentation:      {timings['2_segmentation']:.3f}s  ({n_segments} segments)")

    # Stage 3+4: Feature extraction + Detection (per segment)
    t_feat_total = 0.0
    t_det_total = 0.0
    segment_details = []

    for i, segment in enumerate(seg_result.segments):
        start_sample = segment.start_sample
        end_sample = segment.end_sample
        segment_audio = audio_data[start_sample:end_sample]

        min_samples = pipeline.config.min_speech_duration * sr
        if len(segment_audio) < min_samples:
            continue

        # Feature extraction
        t0 = time.time()
        features = pipeline._feature_extractor.extract(segment_audio, sr)
        t_feat = time.time() - t0
        t_feat_total += t_feat

        # Detection
        t0 = time.time()
        feature_array = features.to_array()
        detection = pipeline._detector.detect(feature_array, features.feature_names)
        t_det = time.time() - t0
        t_det_total += t_det

        segment_details.append({
            "id": i,
            "start": segment.start_time,
            "end": segment.end_time,
            "duration": segment.duration,
            "feat_time": t_feat,
            "det_time": t_det,
            "prediction": detection.prediction.value if hasattr(detection.prediction, 'value') else str(detection.prediction),
            "prob": detection.probability,
        })

    timings["3_feature_extraction"] = t_feat_total
    timings["4_model_inference"] = t_det_total

    print(f"  Stage 3 - Feature extraction:{timings['3_feature_extraction']:.3f}s  ({len(segment_details)} segments)")
    print(f"  Stage 4 - Model inference:   {timings['4_model_inference']:.3f}s  ({len(segment_details)} segments)")

    # Stage 5: Aggregation
    t0 = time.time()
    overall = pipeline._aggregate_results(
        [pipeline._segmenter.segment(audio_data, sr)] and [],
        preprocess_result.properties,
    )
    # Actually re-run aggregation properly
    from audio_forensics.schemas import SegmentResult as SR
    seg_results_for_agg = []
    for sd in segment_details:
        seg_results_for_agg.append(SR(
            segment_id=sd["id"], start_time=sd["start"], end_time=sd["end"],
            duration=sd["duration"], speech_type="speech",
            prediction=sd["prediction"], probability=sd["prob"],
            confidence=abs(sd["prob"] - 0.5) * 2, features_used=[], metadata={},
        ))
    overall = pipeline._aggregate_results(seg_results_for_agg, preprocess_result.properties)
    timings["5_aggregation"] = time.time() - t0

    print(f"  Stage 5 - Aggregation:       {timings['5_aggregation']:.3f}s")

    # Total
    total = sum(timings.values())
    timings["total"] = total
    print(f"\n  {'─' * 50}")
    print(f"  TOTAL (excl. DB write):      {total:.3f}s")
    print(f"  Duration analyzed:           {duration:.1f}s")
    print(f"  Processing ratio:            {total/duration:.2f}x realtime")

    # Segment detail table
    if segment_details:
        print(f"\n  Segment Breakdown:")
        print(f"  {'Seg':>4} {'Time':>12} {'Duration':>10} {'Feat':>8} {'Det':>8} {'Prob':>7} {'Pred':>18}")
        print(f"  {'─' * 75}")
        for sd in segment_details[:20]:  # Show first 20
            print(f"  {sd['id']:>4} {sd['start']:>5.1f}-{sd['end']:.1f}s {sd['duration']:>8.2f}s "
                  f"{sd['feat_time']:>7.3f}s {sd['det_time']:>7.3f}s {sd['prob']:>6.3f} {sd['prediction']:>18}")
        if len(segment_details) > 20:
            print(f"  ... and {len(segment_details) - 20} more segments")

    return timings, segment_details


# ============================================================
# STAGE 2: Profile Backend B (simple classifier)
# ============================================================

def profile_simple_backend(audio_path):
    """Profile the simple EfficientNet backend."""
    print("\n" + "=" * 70)
    print("  BACKEND B: src/audio_infer.py (Simple EfficientNet)")
    print("=" * 70)

    timings = {}

    # Model loading
    t0 = time.time()
    from src.audio_infer import load_audio_detector, predict_from_array
    from src.audio_dataset import SAMPLE_RATE, waveform_to_input
    audio_model, audio_device = load_audio_detector()
    timings["0_model_load"] = time.time() - t0

    print(f"\n  Device: {audio_device}")
    print(f"  Stage 0 - Model loading:     {timings['0_model_load']:.3f}s")

    # Audio loading + resampling
    t0 = time.time()
    import librosa
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    timings["1_audio_load"] = time.time() - t0

    duration = len(y) / SAMPLE_RATE
    print(f"  Audio: {duration:.1f}s, {SAMPLE_RATE}Hz, {len(y)} samples")
    print(f"  Stage 1 - Audio load/resamp: {timings['1_audio_load']:.3f}s")

    # Feature extraction
    t0 = time.time()
    arr = waveform_to_input(y.astype(np.float32), sr=SAMPLE_RATE)
    timings["2_feature_extraction"] = time.time() - t0

    print(f"  Stage 2 - Feature extraction:{timings['2_feature_extraction']:.3f}s")
    print(f"    Output shape: {arr.shape}")

    # Model inference
    t0 = time.time()
    import torch
    tensor = torch.from_numpy(arr).unsqueeze(0).to(audio_device)
    with torch.no_grad():
        prob = torch.sigmoid(audio_model(tensor)).item()
    timings["3_model_inference"] = time.time() - t0

    print(f"  Stage 3 - Model inference:   {timings['3_model_inference']:.3f}s")
    print(f"    Fake probability: {prob:.4f}")

    total = sum(timings.values())
    timings["total"] = total
    print(f"\n  {'─' * 50}")
    print(f"  TOTAL:                       {total:.3f}s")
    print(f"  Processing ratio:            {total/duration:.2f}x realtime")

    return timings


# ============================================================
# STAGE 3: Check device & system info
# ============================================================

def print_system_info():
    import torch
    print("\n" + "=" * 70)
    print("  SYSTEM INFO")
    print("=" * 70)
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory     : {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        print(f"  Device         : CPU only")
    print(f"  PyTorch        : {torch.__version__}")

    try:
        import librosa
        print(f"  Librosa        : {librosa.__version__}")
    except ImportError:
        print(f"  Librosa        : NOT INSTALLED")

    try:
        import soundfile
        print(f"  SoundFile      : {soundfile.__version__}")
    except ImportError:
        print(f"  SoundFile      : NOT INSTALLED")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("#" * 70)
    print("#  AUDIO DETECTION PIPELINE PROFILER")
    print("#  Synthetic 1-minute audio, CPU/GPU timing per stage")
    print("#" * 70)

    print_system_info()

    # Generate test audio
    print("\n" + "=" * 70)
    print("  GENERATING TEST AUDIO")
    print("=" * 70)
    audio_path = generate_test_audio(duration_sec=60)
    if audio_path is None:
        print("  [FATAL] Could not generate test audio.")
        return

    file_size = os.path.getsize(audio_path) / 1024
    print(f"  File: {audio_path}")
    print(f"  Size: {file_size:.0f} KB")
    print(f"  Duration: 60s @ 16kHz")

    # Profile Backend A
    try:
        result_a = profile_forensics_backend(audio_path)
    except Exception as e:
        print(f"\n  [ERROR] Backend A failed: {e}")
        import traceback
        traceback.print_exc()
        result_a = None

    # Profile Backend B
    try:
        result_b = profile_simple_backend(audio_path)
    except Exception as e:
        print(f"\n  [ERROR] Backend B failed: {e}")
        import traceback
        traceback.print_exc()
        result_b = None

    # ============================================================
    # COMPARISON SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 70)

    if result_a:
        timings_a, seg_details = result_a
        total_a = timings_a["total"]
        print(f"\n  Backend A (audio_forensics) total: {total_a:.3f}s")
        print(f"  {'Stage':<30} {'Time':>8} {'% of Total':>10}")
        print(f"  {'─' * 50}")
        for key in sorted(timings_a.keys()):
            if key.startswith("_") or key == "total":
                continue
            val = timings_a[key]
            pct = val / total_a * 100
            label = key.replace("_", " ").lstrip("0123456789 ").strip()
            print(f"  {label:<30} {val:>7.3f}s {pct:>9.1f}%")
        print(f"  {'─' * 50}")
        print(f"  {'TOTAL':<30} {total_a:>7.3f}s {'100.0%':>10}")

        # Identify bottleneck
        stage_keys = [k for k in timings_a if k.startswith(("1_", "2_", "3_", "4_", "5_"))]
        if stage_keys:
            bottleneck = max(stage_keys, key=lambda k: timings_a[k])
            bn_label = bottleneck.replace("_", " ").lstrip("0123456789 ").strip()
            print(f"\n  >>> BOTTLENECK: {bn_label} ({timings_a[bottleneck]:.3f}s, "
                  f"{timings_a[bottleneck]/total_a*100:.1f}% of total)")

    if result_b:
        total_b = result_b["total"]
        print(f"\n  Backend B (simple EfficientNet) total: {total_b:.3f}s")
        print(f"  {'Stage':<30} {'Time':>8} {'% of Total':>10}")
        print(f"  {'─' * 50}")
        for key in sorted(result_b.keys()):
            if key.startswith("_") or key == "total":
                continue
            val = result_b[key]
            pct = val / total_b * 100
            label = key.replace("_", " ").lstrip("0123456789 ").strip()
            print(f"  {label:<30} {val:>7.3f}s {pct:>9.1f}%")
        print(f"  {'─' * 50}")
        print(f"  {'TOTAL':<30} {total_b:>7.3f}s {'100.0%':>10}")

    # Comparison
    if result_a and result_b:
        print(f"\n  SPEED COMPARISON:")
        print(f"  Backend A (forensics) : {timings_a['total']:.3f}s")
        print(f"  Backend B (simple)    : {result_b['total']:.3f}s")
        ratio = timings_a['total'] / result_b['total'] if result_b['total'] > 0 else float('inf')
        print(f"  Ratio A/B             : {ratio:.1f}x slower")
        if timings_a.get("0_pipeline_init", 0) > 1.0:
            print(f"\n  NOTE: Pipeline initialization alone takes {timings_a['0_pipeline_init']:.1f}s")
            print(f"  This is NOT cached via @st.cache_resource and runs EVERY upload.")

    # Recommendations
    print(f"\n{'=' * 70}")
    print("  RECOMMENDATIONS")
    print("=" * 70)
    print("""
  1. CACHE the AudioInferencePipeline (currently re-created every upload)
  2. The forensics backend processes each segment individually --
     batch inference across segments would reduce overhead
  3. Feature extraction (librosa) runs per-segment on CPU --
     this is likely the dominant cost for long audio
  4. Consider pre-loading models at app startup, not on first upload
  5. For short audio (<30s), Backend B (simple) is much faster
  6. For production, consider ONNX or TorchScript for inference speedup
""")

    # Cleanup
    try:
        os.remove(audio_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
