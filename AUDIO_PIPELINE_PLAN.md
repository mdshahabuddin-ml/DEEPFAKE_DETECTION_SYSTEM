# Audio Analysis Pipeline Implementation Plan

## Overview
Implement a complete audio deepfake detection pipeline following the specified flowchart:

```
Audio Input → Audio Extraction → Preprocessing → Voice Activity Detection → 
Speech Segmentation → Feature Extraction → Audio Deepfake Model → 
Segment-Level Scores → Temporal Aggregation → Audio Manipulation Assessment
```

## Current State Analysis

### Existing Components:
- `src/audio_analysis.py` - Basic spectral analysis, VAD, quality assessment
- `src/audio_model.py` - CNN and EfficientNet models for spectrograms
- `src/audio_dataset.py` - Dataset loading and feature extraction
- `src/audio_train.py` - Training loop
- `src/audio_infer.py` - Basic inference helpers

### Gaps to Fill:
1. **Audio Extraction** - Extract audio from video files (ffmpeg integration)
2. **Speech Segmentation** - Split audio into analyzable segments
3. **Segment-Level Scoring** - Per-segment fake probability scores
4. **Temporal Aggregation** - Combine segment scores with weighted averaging
5. **Manipulation Assessment** - Detailed manipulation type classification
6. **Pipeline Orchestration** - Unified pipeline class

## Implementation Plan

### Step 1: Audio Extraction Module (`src/audio_extraction.py`)
**Purpose**: Extract audio from various input sources

```python
class AudioExtractor:
    - extract_from_video(video_path) -> audio_path
    - extract_from_file(file_path) -> np.ndarray
    - normalize_audio(audio, target_sr=16000) -> np.ndarray
    - handle_multiple_formats() ->支持wav, mp3, mp4,avi, mov, etc.
```

**Dependencies**: ffmpeg, librosa, pydub

### Step 2: Preprocessing Module (`src/audio_preprocessing.py`)
**Purpose**: Clean and prepare audio for analysis

```python
class AudioPreprocessor:
    - remove_noise(audio) -> audio
    - normalize_volume(audio) -> audio
    - resample(audio, target_sr) -> audio
    - convert_mono(audio) -> audio
    - trim_silence(audio) -> audio
    - apply_preemphasis(audio) -> audio
```

### Step 3: Enhanced Voice Activity Detection (`src/audio_vad.py`)
**Purpose**: Detect speech segments with improved accuracy

```python
class AdvancedVoiceActivityDetector:
    - detect_vad(audio) -> List[VoiceSegment]
    - compute_energy_contour(audio) -> np.ndarray
    - detect_speech_onset(audio) -> List[float]
    - merge_short_segments(segments, min_duration) -> List[VoiceSegment]
    - filter_by_quality(segments, min_snr) -> List[VoiceSegment]
```

### Step 4: Speech Segmentation Module (`src/audio_segmentation.py`)
**Purpose**: Split audio into analysis-ready segments

```python
class SpeechSegmenter:
    - segment_by_vad(audio, vad_segments) -> List[AudioSegment]
    - segment_by_fixed_length(audio, segment_duration) -> List[AudioSegment]
    - segment_by_silence(audio, min_silence_duration) -> List[AudioSegment]
    - overlap_segments(segments, overlap_ratio) -> List[AudioSegment]
```

### Step 5: Enhanced Feature Extraction (`src/audio_features.py`)
**Purpose**: Extract comprehensive features per segment

```python
class AudioFeatureExtractor:
    - extract_mel_spectrogram(audio) -> np.ndarray
    - extract_mfcc(audio, n_mfcc=13) -> np.ndarray
    - extract_spectral_features(audio) -> Dict
    - extract_prosodic_features(audio) -> Dict
    - extract_phoneme_features(audio) -> np.ndarray
    - extract_all(audio) -> AudioFeatures
```

### Step 6: Segment-Level Scoring (`src/audio_segment_scorer.py`)
**Purpose**: Compute fake probability for each segment

```python
class SegmentScorer:
    - score_segment(audio_segment, model) -> SegmentScore
    - score_batch(segments, model) -> List[SegmentScore]
    - compute_segment_confidence(score) -> float
    - detect_anomalies(segment_scores) -> List[Anomaly]
```

### Step 7: Temporal Aggregation (`src/audio_temporal.py`)
**Purpose**: Combine segment scores into final prediction

```python
class TemporalAggregator:
    - aggregate_scores(segment_scores, method) -> AggregatedResult
    - weighted_average(scores, weights) -> float
    - max_pooling(scores) -> float
    - attention_aggregation(scores, features) -> float
    - compute_temporal_consistency(scores) -> float
```

### Step 8: Manipulation Assessment (`src/audio_manipulation.py`)
**Purpose**: Classify type of manipulation detected

```python
class ManipulationAssessor:
    - assess_manipulation(audio, segment_scores) -> ManipulationReport
    - detect_splicing(audio) -> SplicingResult
    - detect_tts(audio) -> TTSResult
    - detect_voice_conversion(audio) -> VCResult
    - detect_replay(audio) -> ReplayResult
    - generate_manipulation_map(audio) -> np.ndarray
```

### Step 9: Pipeline Orchestrator (`src/audio_pipeline.py`)
**Purpose**: Unified pipeline combining all components

```python
class AudioDeepfakePipeline:
    def __init__(self, config):
        self.extractor = AudioExtractor()
        self.preprocessor = AudioPreprocessor()
        self.vad = AdvancedVoiceActivityDetector()
        self.segmenter = SpeechSegmenter()
        self.feature_extractor = AudioFeatureExtractor()
        self.model = load_audio_model()
        self.scorer = SegmentScorer()
        self.aggregator = TemporalAggregator()
        self.assessor = ManipulationAssessor()
    
    def analyze(self, input_path) -> AudioAnalysisResult:
        # Step 1: Extract audio
        audio = self.extractor.extract(input_path)
        
        # Step 2: Preprocess
        audio = self.preprocessor.process(audio)
        
        # Step 3: Voice Activity Detection
        vad_segments = self.vad.detect(audio)
        
        # Step 4: Speech Segmentation
        segments = self.segmenter.segment(audio, vad_segments)
        
        # Step 5: Feature Extraction
        features = [self.feature_extractor.extract(seg) for seg in segments]
        
        # Step 6: Segment-Level Scoring
        segment_scores = self.scorer.score_batch(features, self.model)
        
        # Step 7: Temporal Aggregation
        final_score = self.aggregator.aggregate(segment_scores)
        
        # Step 8: Manipulation Assessment
        manipulation_report = self.assessor.assess(audio, segment_scores)
        
        return AudioAnalysisResult(
            fake_probability=final_score,
            segment_scores=segment_scores,
            manipulation_report=manipulation_report,
            ...
        )
```

### Step 10: Integration & Testing
- Integrate with existing `app.py` and `api/main.py`
- Add comprehensive unit tests
- Create example notebooks/scripts
- Update documentation

## File Structure
```
src/
├── audio_extraction.py      # NEW - Audio extraction from video/files
├── audio_preprocessing.py   # NEW - Audio cleaning and normalization
├── audio_vad.py             # NEW - Advanced voice activity detection
├── audio_segmentation.py    # NEW - Speech segmentation
├── audio_features.py        # NEW - Enhanced feature extraction
├── audio_segment_scorer.py  # NEW - Segment-level scoring
├── audio_temporal.py        # NEW - Temporal aggregation
├── audio_manipulation.py    # NEW - Manipulation type assessment
├── audio_pipeline.py        # NEW - Unified pipeline orchestrator
├── audio_analysis.py        # EXISTING - Will be refactored
├── audio_model.py           # EXISTING - Model architectures
├── audio_dataset.py         # EXISTING - Dataset handling
├── audio_train.py           # EXISTING - Training loop
└── audio_infer.py           # EXISTING - Basic inference
```

## Implementation Order
1. `audio_extraction.py` - Foundation for all audio input
2. `audio_preprocessing.py` - Clean audio for analysis
3. `audio_vad.py` - Detect speech regions
4. `audio_segmentation.py` - Split into analyzable chunks
5. `audio_features.py` - Extract comprehensive features
6. `audio_segment_scorer.py` - Per-segment scoring
7. `audio_temporal.py` - Combine segment results
8. `audio_manipulation.py` - Detailed manipulation analysis
9. `audio_pipeline.py` - Orchestrate everything
10. Integration & testing

## Key Design Decisions
1. **Modularity**: Each component is independent and testable
2. **Configurability**: Pipeline parameters configurable via config
3. **Backward Compatibility**: Existing code continues to work
4. **Performance**: Batch processing for efficiency
5. **Extensibility**: Easy to add new manipulation detectors

## Dependencies Required
```python
# Already present:
librosa
numpy
torch
soundfile

# May need to add:
pydub          # Audio format conversion
ffmpeg-python  # Video audio extraction
webrtcvad      # WebRTC-based VAD (optional)
scipy          # Signal processing
```

## Success Metrics
- Pipeline processes audio files end-to-end
- Segment-level scores are meaningful and interpretable
- Temporal aggregation improves accuracy over single-score
- Manipulation assessment provides actionable insights
- Performance is acceptable (< 5s for 10s audio on CPU)
- All existing tests continue to pass
