# Audio Forensics Tests

This directory contains tests for the audio forensics module.

## Running Tests

```bash
# Run all tests
python -m pytest audio_forensics/tests/

# Run specific test
python -m audio_forensics.tests.test_preprocessing
```

## Test Coverage

- Preprocessing pipeline
- Feature extraction
- Segmentation
- Inference
- Speaker verification

## Adding Tests

1. Create test file: `test_<module>.py`
2. Use pytest conventions
3. Mock external dependencies
4. Test edge cases and error handling
