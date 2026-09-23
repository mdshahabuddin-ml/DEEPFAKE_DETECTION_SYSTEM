# Audio Forensics Datasets

This directory stores datasets for audio forensics research and evaluation.

## Recommended Datasets

- **ASVspoof**: Anti-spoofing benchmark
- **In-the-Wild**: Real-world deepfake audio
- **Celeb-DF**: Audio from celebrity deepfake videos
- **FakeAVCeleb**: Audio-visual deepfake dataset
- **WaveFake**: Generated audio detection

## Data Organization

```
datasets/
├── train/
│   ├── bonafide/
│   └── fake/
├── val/
│   ├── bonafide/
│   └── fake/
└── test/
    ├── bonafide/
    └── fake/
```

## Privacy Notice

- Never store user-uploaded audio as training data
- Use only publicly available datasets
- Follow dataset licensing terms
- Separate production evidence from training data
