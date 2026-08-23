# 🕵️ AI-Powered Explainable Deepfake Detection System

Detect whether an **image**, **video**, or **live webcam** stream is **REAL** or a **DEEPFAKE** — with confidence scores and visual **Grad-CAM** explanations of *why*.

A modular, production-style system built with **PyTorch, OpenCV, Streamlit, FastAPI, SQLite, and Docker**, suitable for a final-year B.Tech AI/ML project and placement portfolio.

---

## ✨ Features

| Capability | Description |
|---|---|
| 📷 **Image detection** | Upload JPG/PNG → face detection → crop → resize → inference → REAL/FAKE + confidence |
| 🎬 **Video detection** | Frame sampling → per-frame inference → **majority voting** + frame statistics |
| 📹 **Real-time webcam** | Live bounding box, label, confidence, and FPS (desktop + in-browser) |
| 🔥 **Explainable AI** | **Grad-CAM** heatmap overlay comparing original vs. model attention |
| 🗂️ **Prediction history** | Every result stored in **SQLite** (file, prediction, confidence, time, date) |
| 📊 **Dashboard** | Aggregate analytics (real/fake counts, avg confidence, avg latency) |
| 🌐 **REST API** | **FastAPI** with auto-generated Swagger docs |
| 🐳 **Docker** | One-command deployment of API + UI via Docker Compose |
| 🔐 **User login** | Auth-gated UI (hashed passwords in SQLite) on Streamlit & Gradio |
| 🔊 **Audio (bonus)** | Voice-clone detection + audio/video score **fusion** (see `src/audio_*`) |
| 🛡️ **AI Trust Score** | 0–100 authenticity score fusing face, temporal, lip-sync & texture signals |
| 📄 **PDF report** | Exportable authenticity report (verdict, sub-scores, hash, recommendation) |
| ⚙️ **ONNX export** | Edge/cross-runtime model export (`src/export_onnx.py`) |
| 🛡️ **Robustness benchmark** | Eval under compression/blur/noise/crop/resize (`src/robustness.py`) |
| ✅ **CI** | GitHub Actions runs the test suite on every push |

---

## 🏗️ Architecture

```
                ┌──────────────────────────────────────────────┐
                │                Streamlit UI                  │
                │  Home · Image · Video · Webcam · Grad-CAM ·  │
                │        History · Dashboard · About           │
                └───────────────┬──────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                        │
┌───────▼────────┐   ┌──────────▼─────────┐   ┌──────────▼─────────┐
│  FastAPI (API) │   │  Inference Engine  │   │   Grad-CAM (XAI)   │
│  /predict/*    │──▶│  DeepfakeDetector  │◀──│ GradCAMExplainer   │
│  /health /info │   │  + FaceDetector    │   └────────────────────┘
└───────┬────────┘   └──────────┬─────────┘
        │                       │
        │              ┌────────▼─────────┐      ┌──────────────────┐
        └─────────────▶│  SQLite history  │      │  XceptionNet     │
                       │  predictions.db  │      │  (PyTorch/timm)  │
                       └──────────────────┘      └──────────────────┘

Pipeline:  Media → Face detection (MTCNN/Haar) → Crop → Resize(299) →
           Xception → Sigmoid → Fake probability → Verdict + Confidence
                            ┌─────────────┐
                 │    USER     │
                 └──────┬──────┘
                        │
                        ▼
              ┌─────────────────┐
              │   STREAMLIT UI  │
              └────────┬────────┘
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
       ┌───────┐   ┌───────┐   ┌────────┐
       │ Image │   │ Video │   │ Webcam │
       └───┬───┘   └───┬───┘   └───┬────┘
           └───────────┼────────────┘
                       ▼
              ┌─────────────────┐
              │ Face Detection  │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │    Face Crop    │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Preprocessing   │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │    Xception     │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Fake Probability│
              └────────┬────────┘
                       │
                ┌──────┴──────┐
                ▼             ▼
           ┌────────┐    ┌──────────┐
           │ Verdict│    │ Grad-CAM │
           └────┬───┘    └────┬─────┘
                └──────┬──────┘
                       ▼
              ┌─────────────────┐
              │ Prediction      │
              │ History /SQLite │
              └─────────────────┘
```

### Project structure

```
deepfake_detection_system/
├── app/
│   ├── streamlit_app.py     # Multi-page Streamlit UI
│   ├── predict.py           # UI ↔ inference helpers (+ DB logging)
│   └── webcam.py            # Webcam page (WebRTC + snapshot fallback)
├── api/
│   └── main.py              # FastAPI service (Swagger at /docs)
├── src/
│   ├── config.py            # Central config (paths, model, thresholds)
│   ├── logger.py            # Rotating file + console logging
│   ├── database.py          # SQLite prediction history
│   ├── inference.py         # FaceDetector + DeepfakeDetector
│   ├── gradcam.py           # Grad-CAM explainer
│   ├── webcam_detector.py   # Real-time webcam class + CLI
│   ├── model.py             # Xception / EfficientNet / ViT architectures
│   ├── dataset.py           # Datasets + transforms
│   ├── train.py             # Training loop
│   ├── evaluate.py          # Evaluation
│   ├── face_extractor.py    # MTCNN batch face extraction (dataset prep)
│   └── audio_*.py, fusion.py, realtime_*.py   # Audio + fusion (bonus)
├── models/                  # best_xception_celebdf.pth  (place here)
├── uploads/ outputs/ logs/ results/            # runtime artifacts
├── notebooks/ tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**Celeb-DF v2** — real celebrity videos + high-quality synthesized deepfakes.
Faces are extracted per frame (`src/face_extractor.py`) and the model is trained
to output a single *fake* probability. The pipeline also supports FaceForensics++
and DFDC (see `src/dataset.py`).

---

## ⚙️ Installation

Requires **Python 3.11+**.

```bash
git clone <your-repo-url>
cd deepfake_detection_system

python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

Place your trained checkpoint at:

```
models/best_xception_celebdf.pth
```

---

## 🎓 Training

### Organize the dataset into train/val/test splits
```bash
# Scans <root>/real/ and <root>/fake/ and writes train.csv, val.csv, test.csv
python -m src.make_dataset_splits --data_root data/processed/celeb-df-v2 \
                                  --output_dir data/splits
```
Each CSV has columns `path,label`; the split is **stratified** to preserve class balance.

### Train
```bash
# From the separate split CSVs above
python src/train.py --split_dir data/splits --model xception --epochs 10

# Celeb-DF v2 directly (faces already extracted)
python src/train.py --dataset celebdf --data_root data/processed/celeb-df-v2 --epochs 10

# Or from a single combined split CSV (columns: path,label,split)
python src/train.py --split_csv data/celebdf_split.csv --model xception --epochs 10
```

### Train and test on local DFDC data

Prepare source-disjoint crops and splits (an original video and every fake
derived from it are always assigned to the same split), then train:

```bash
python -m src.prepare_dfdc --input_dir "E:\DFDC_dataset\DFDC Dtata_set_kaggle" --output_dir data/processed/dfdc --frames_per_video 4
python src/train.py --dataset dfdc --split_csv data/processed/dfdc/dfdc_split_seed42.csv --model xception --epochs 10 --batch_size 16 --no_wandb
python src/evaluate.py --dataset dfdc --checkpoint models/best_xception_dfdc_split_seed42.pth --data_root data/processed/dfdc --split_csv data/processed/dfdc/dfdc_split_seed42.csv --no_plots
```

The best checkpoint (by validation AUC) is saved to `models/best_xception_celebdf.pth`.

---

## 🚀 Inference

### Gradio UI (KYC screening — recommended for demos & cloud deploy)
```bash
python app/gradio_app.py
# open http://localhost:7860
```
Image / video / **batch** screening with KYC verdicts, confidence %, Grad-CAM
overlays, and timestamped audit artifacts saved to `results/`.

### Streamlit UI
```bash
streamlit run app/streamlit_app.py
# open http://localhost:8501
```

### Real-time webcam (desktop)
```bash
python -m src.webcam_detector --camera 0 --threshold 0.5
```

### Python API
```python
from src.inference import get_detector
det = get_detector()
print(det.predict_image("path/to/face.jpg"))
print(det.predict_video("path/to/clip.mp4"))
```

---

## 🛡️ Advanced features

### AI Trust Score + PDF report
Multi-signal 0–100 authenticity assessment (face consistency, temporal
consistency, lip-sync/voice agreement, spectral texture) with an exportable
PDF report. Available in the **Gradio "Trust Score" tab** and the API:

```bash
curl -X POST "http://localhost:8000/analyze/image" -F "file=@face.jpg"      # JSON report
curl -X POST "http://localhost:8000/report/image"  -F "file=@face.jpg" -o report.pdf
```

### Face localization (multi-face)
Detects and scores **every** face in an image (not just the largest), drawing
per-face REAL/FAKE boxes. Available in the Gradio **Face Localization** tab and
the `POST /localize/image` API.

### Batch video processing
Screen **many videos** in one pass → results table + CSV, via the Gradio
**Batch Videos** tab or `DeepfakeDetector.predict_videos([...])`.

### Annotated output video
Generate a **downloadable MP4** with per-frame detection boxes and REAL/FAKE
labels drawn on it — Gradio **Annotated Video** tab or
`DeepfakeDetector.annotate_video(path)`.

### User login (auth-gated UI)
Both the Streamlit and Gradio apps show a **login page** before granting access.
Users are stored in SQLite with salted PBKDF2 password hashes (`src/auth.py`).
A default admin is created on first run:

```
username: admin
password: admin123          # change via DEEPFAKE_ADMIN_PASSWORD
```

```bash
export DEEPFAKE_ADMIN_USER="admin"
export DEEPFAKE_ADMIN_PASSWORD="a-strong-password"
# Disable the Gradio login (e.g. for a public demo):
export DEEPFAKE_DISABLE_LOGIN=1
```

Signed in as an **admin**, a **👥 User Management** page appears in the Streamlit
sidebar to create users, reset passwords, and delete accounts (the last admin
is protected). Users can also be added programmatically:
```python
from src.auth import get_auth_db
get_auth_db().add_user("analyst", "their-password", role="user")
```

### API security (auth + rate limiting)
Optional, production-grade guards on the inference endpoints:
```bash
export DEEPFAKE_API_KEY="your-secret-key"   # require header  X-API-Key: your-secret-key
export DEEPFAKE_RATE_LIMIT=60               # max 60 requests/min per client (0 = unlimited)
```
`GET /health` reports `auth_required` and `rate_limit_per_min`. System/docs
routes stay open; only `/predict`, `/analyze`, `/report`, `/localize` are guarded.

### ONNX export + lightweight edge model
```bash
python -m src.export_onnx --output models/xception_deepfake.onnx            # FP32 ONNX
python -m src.export_onnx --output models/xception_deepfake.onnx --quantize # + INT8 edge model (~4x smaller)
# -> ONNX Runtime / TensorRT / OpenVINO / onnxruntime-web ready
# quantization needs: pip install onnxruntime
```

### Robustness benchmark
```bash
# data/robustness_eval/real/*.jpg and .../fake/*.jpg
python -m src.robustness --data_root data/robustness_eval --max_images 200
# prints a before/after table and saves a CSV to results/
```

### Continuous integration
`.github/workflows/ci.yml` installs dependencies and runs `pytest` on every
push / PR to `main`.

## 🐳 Docker

```bash
# Build + run both API and UI
docker compose up --build

# API  -> http://localhost:8000/docs
# UI   -> http://localhost:8501
```

Mount your checkpoint via the `./models` volume (already configured in
`docker-compose.yml`).

Single-container usage:

```bash
docker build -t deepfake-detection .
docker run -p 8000:8000 -v $(pwd)/models:/app/models deepfake-detection
```

### Cloud deployment

Ready-made configs for **Hugging Face Spaces**, **Render**, and **Railway** are
included (`render.yaml`, `railway.json`, `Procfile`, `runtime.txt`). See
**[DEPLOYMENT.md](DEPLOYMENT.md)** for step-by-step instructions per platform.

---

## 🌐 REST API

Interactive Swagger UI at **`/docs`**, ReDoc at **`/redoc`**.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict/image` | Classify an uploaded image |
| `POST` | `/predict/video` | Classify an uploaded video (majority voting) |
| `POST` | `/localize/image` | Detect & score **every** face (multi-face localization) |
| `POST` | `/analyze/image` | Multi-signal **Trust Score** report (JSON) |
| `POST` | `/analyze/video` | Multi-signal **Trust Score** report (JSON) |
| `POST` | `/report/image` | Downloadable **PDF** authenticity report |
| `GET`  | `/health` | Service + model health |
| `GET`  | `/model/info` | Model metadata |
| `GET`  | `/history` | Recent predictions |
| `GET`  | `/stats` | Aggregate statistics |

Example:

```bash
curl -X POST "http://localhost:8000/predict/image" \
     -F "file=@face.jpg"
```

```json
{
  "file_name": "face.jpg",
  "media_type": "image",
  "prediction": "FAKE",
  "confidence": 0.94,
  "fake_probability": 0.94,
  "inference_time": 0.08,
  "details": {"num_faces": 1}
}
```

---

## 🧪 Tests

```bash
pytest -q
```

Model-dependent tests auto-skip when no checkpoint is present, so the suite
passes on a fresh clone.

---

## 📝 Logging

Structured logs stream to the console and to `logs/application.log`
(rotating, 5 MB × 3 backups) via `src/logger.py`.

---

## 🛡️ Error handling

Handled gracefully across the stack:
- **Missing model** → API returns `503`; UI shows a guided message
- **Invalid / empty upload** → `400`
- **Unsupported file type** → `400`; **oversized** → `413`
- **No face detected** → explicit `NO_FACE` result (no crash)
- **Camera / video read failure** → clear error, clean resource release

---

## 📊 Results

Report your metrics here after training, e.g.:

| Metric | Value |
|---|---|
| Test AUC | _e.g. 0.98_ |
| Accuracy | _e.g. 0.95_ |
| Avg inference (CPU) | _e.g. 80 ms/image_ |

---

## 📸 Screenshots

_Add screenshots of the Home, Image, Grad-CAM, and Dashboard pages here._

---

## 🔮 Future work

- RetinaFace / InsightFace detector for harder poses
- Temporal (video) models (e.g. CNN+LSTM, timesformer)
- Multi-model ensembling (Xception + EfficientNet + ViT) 
- Confidence calibration and adversarial robustness
- Cloud deployment (AWS/GCP) with GPU inference

---

## ⚠️ Disclaimer

Predictions are probabilistic and can err. Use as a decision-support aid, not
as sole ground truth.
###
Local host:http://localhost:7860 
 