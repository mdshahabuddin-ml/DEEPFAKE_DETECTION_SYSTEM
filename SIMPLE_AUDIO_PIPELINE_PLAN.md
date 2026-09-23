# Simple Audio Deepfake Detection Pipeline

## Pipeline Flow
```
Audio Upload → Audio Preprocessing → Spectrogram Generation → Basic Audio Classifier → REAL / SYNTHETIC
```

## Current State Analysis

### Existing Components (Already Implemented):
✅ **Audio Preprocessing** - `src/audio_analysis.py` (AudioQualityAnalyzer)
✅ **Spectrogram Generation** - `src/audio_dataset.py` (waveform_to_logmel, waveform_to_input)
✅ **Audio Classifier** - `src/audio_model.py` (SpectrogramCNN, TimmAudioDetector)
✅ **Basic Inference** - `src/audio_infer.py` (predict_from_file, predict_from_array)

### Missing Component:
❌ **Unified Pipeline** - Single class to orchestrate the flow

## Implementation Plan

### Step 1: Create Simple Audio Pipeline (`src/simple_audio_pipeline.py`)

```python
class SimpleAudioPipeline:
    """
    Simple 5-step audio deepfake detection pipeline.
    
    Flow: Upload → Preprocess → Spectrogram → Classify → REAL/SYNTHETIC
    """
    
    def __init__(self, model_name="efficientnet", checkpoint_path=None):
        self.model, self.device = load_audio_detector(model_name, checkpoint_path)
        self.sr = 16000
        self.duration = 4.0  # seconds
    
    def analyze(self, audio_input) -> dict:
        """
        Process audio through the complete pipeline.
        
        Args:
            audio_input: File path (str/Path) or numpy array
            
        Returns:
            dict with keys: prediction, confidence, fake_probability, 
                           processing_time, spectrogram
        """
        start_time = time.time()
        
        # Step 1: Handle audio input (upload)
        if isinstance(audio_input, (str, Path)):
            audio_array = self._load_audio(audio_input)
        else:
            audio_array = audio_input
        
        # Step 2: Audio preprocessing
        preprocessed = self._preprocess(audio_array)
        
        # Step 3: Spectrogram generation
        spectrogram = self._generate_spectrogram(preprocessed)
        
        # Step 4: Basic audio classifier
        fake_prob = self._classify(spectrogram)
        
        # Step 5: REAL / SYNTHETIC decision
        prediction = "SYNTHETIC" if fake_prob >= 0.5 else "REAL"
        confidence = abs(fake_prob - 0.5) * 2
        
        processing_time = time.time() - start_time
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_probability": float(fake_prob),
            "processing_time": float(processing_time),
            "spectrogram": spectrogram,
        }
    
    def _load_audio(self, file_path):
        """Load audio file."""
        y, _ = librosa.load(str(file_path), sr=self.sr, mono=True)
        return y.astype(np.float32)
    
    def _preprocess(self, audio):
        """Step 2: Audio preprocessing."""
        # Normalize
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Fix length
        target_length = int(self.sr * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        
        return audio
    
    def _generate_spectrogram(self, audio):
        """Step 3: Generate log-mel spectrogram."""
        return waveform_to_input(audio, sr=self.sr)
    
    def _classify(self, spectrogram):
        """Step 4: Run through audio classifier."""
        tensor = torch.from_numpy(spectrogram).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor)
            fake_prob = torch.sigmoid(logit).item()
        return fake_prob
```

### Step 2: Add Configuration Options

```python
# In src/config.py - add these settings:

# Audio Pipeline Settings
AUDIO_PIPELINE_MODEL = os.getenv("AUDIO_PIPELINE_MODEL", "efficientnet")
AUDIO_PIPELINE_CHECKPOINT = os.getenv(
    "AUDIO_PIPELINE_CHECKPOINT", 
    str(MODELS_DIR / "best_audio_efficientnet.pth")
)
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 4.0  # seconds
AUDIO_FAKE_THRESHOLD = 0.5
```

### Step 3: Create Usage Examples

```python
# Example 1: Basic usage
from src.simple_audio_pipeline import SimpleAudioPipeline

pipeline = SimpleAudioPipeline()
result = pipeline.analyze("path/to/audio.wav")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")

# Example 2: With numpy array
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
result = pipeline.analyze(audio)

# Example 3: Batch processing
files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = [pipeline.analyze(f) for f in files]
```

### Step 4: Integration with Existing App

```python
# In app.py - Audio tab integration:

from src.simple_audio_pipeline import SimpleAudioPipeline

# Initialize pipeline (cached)
@st.cache_resource
def load_audio_pipeline():
    return SimpleAudioPipeline()

# In the Audio tab:
if uploaded_audio:
    pipeline = load_audio_pipeline()
    result = pipeline.analyze(uploaded_audio)
    
    # Display results
    st.metric("Prediction", result['prediction'])
    st.metric("Confidence", f"{result['confidence']:.2%}")
    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
```

## File Structure
```
src/
├── simple_audio_pipeline.py  # NEW - Main pipeline class
├── audio_analysis.py         # EXISTING - Preprocessing helpers
├── audio_dataset.py          # EXISTING - Spectrogram generation
├── audio_model.py            # EXISTING - Classifier models
├── audio_infer.py            # EXISTING - Inference helpers
└── config.py                 # EXISTING - Add audio pipeline settings
```

## Implementation Order
1. Create `src/simple_audio_pipeline.py` with SimpleAudioPipeline class
2. Add configuration constants to `src/config.py`
3. Create example usage script (`examples/audio_pipeline_example.py`)
4. Integrate with existing `app.py` Audio tab
5. Add unit tests (`tests/test_simple_audio_pipeline.py`)

## Key Features
- **Simple API**: Single `analyze()` method
- **Flexible Input**: Accepts file paths or numpy arrays
- **Complete Output**: Returns prediction, confidence, probability, timing
- **Reusable**: Can be instantiated once and used multiple times
- **Configurable**: Model choice, thresholds, duration all configurable

## Dependencies
Already in requirements.txt:
- librosa (audio loading)
- torch (model inference)
- numpy (array operations)

No new dependencies needed!

## Success Criteria
- [ ] Pipeline processes audio files end-to-end
- [ ] Returns REAL/SYNTHETIC prediction with confidence
- [ ] Processing time < 2s for 4s audio on CPU
- [ ] Works with existing audio models
- [ ] Integrates with Streamlit app
- [ ] All existing tests pass
