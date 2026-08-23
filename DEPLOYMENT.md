# 🚀 Deployment Guide

The **Gradio KYC app** (`app/gradio_app.py`) is the primary web entry point for
deployment. It reads the `PORT` environment variable and binds to `0.0.0.0`, so
it works out-of-the-box on every platform below. Provide your trained checkpoint
at `models/best_xception_celebdf.pth` (gitignored) or set `DEEPFAKE_CHECKPOINT`.

Entry points:
- **Gradio UI** → `python app/gradio_app.py` (port 7860 locally, `$PORT` in cloud)
- **Streamlit UI** → `streamlit run app/streamlit_app.py` (8501)
- **REST API** → `uvicorn api.main:app` (8000, Swagger at `/docs`)

---

## 1) 🤗 Hugging Face Spaces (Gradio SDK)

Spaces reads front matter from the repo's `README.md`. Because our default
`app.py` is the Streamlit app, point the Space at the Gradio file with
`app_file`. Create the Space (SDK = Gradio) and make its `README.md` start with:

```yaml
---
title: Deepfake Detection KYC
emoji: 🕵️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.20.0
app_file: app/gradio_app.py
pinned: false
---
```

Steps:
1. Create a new **Gradio** Space on huggingface.co.
2. Push this repository to it (`git remote add space https://huggingface.co/spaces/<user>/<name>` → `git push space main`).
3. Upload the checkpoint to the Space (via the UI or `huggingface_hub`), or load
   it from the Hub at startup.
4. The Space installs `requirements.txt` and launches `app/gradio_app.py`.

> Tip: for GPU, select a GPU hardware tier in the Space settings; the app uses
> CUDA automatically when available.

---

## 2) 🎨 Render

A blueprint is provided in **`render.yaml`**.

1. Push the repo to GitHub.
2. Render dashboard → **New + → Blueprint** → select this repo.
3. Render builds (installs CPU PyTorch + `requirements.txt`) and runs
   `python app/gradio_app.py`, injecting `$PORT`.

Notes:
- Use at least the **Starter** plan — PyTorch needs more memory/build time than
  the free tier allows.
- Add the checkpoint via a **persistent disk** mounted at `models/`, or bake it
  into a custom Docker image (Render also supports `Dockerfile` deploys).

---

## 3) 🚄 Railway

Config is provided in **`railway.json`** (Nixpacks) and **`Procfile`**.

1. `railway init` (or connect the GitHub repo in the Railway dashboard).
2. Railway builds with Nixpacks and runs `python app/gradio_app.py`.
3. It injects `$PORT`, which the app honours.

To run the **API** instead of the UI, change the start command to
`uvicorn api.main:app --host 0.0.0.0 --port $PORT` (see the `api` line in `Procfile`).

---

## 4) 🐳 Docker / Docker Compose

Single container (Gradio):

```bash
docker build -t deepfake-detection .
docker run -p 7860:7860 -e PORT=7860 \
    -v $(pwd)/models:/app/models \
    deepfake-detection python app/gradio_app.py
```

All services at once (API + Streamlit + Gradio):

```bash
docker compose up --build
# API     -> http://localhost:8000/docs
# Streamlit -> http://localhost:8501
# Gradio  -> http://localhost:7860
```

`.dockerignore` keeps datasets and runtime artifacts out of the build context.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `7860` | Web port (set automatically by Render/Railway/HF) |
| `DEEPFAKE_CHECKPOINT` | `models/best_xception_celebdf.pth` | Model weights path |
| `DEEPFAKE_CHECKPOINT_URL` | _(unset)_ | Direct HTTPS URL to auto-download weights on startup |
| `DEEPFAKE_HF_REPO` | _(unset)_ | Hugging Face repo id to auto-download weights from |
| `DEEPFAKE_HF_FILENAME` | checkpoint basename | File to fetch from the HF repo |
| `DEEPFAKE_HF_REVISION` | _(unset)_ | Optional HF branch/tag/commit |
| `DEEPFAKE_MODEL_NAME` | `xception` | Backbone architecture |
| `DEEPFAKE_FACE_DETECTOR` | `mtcnn` | `mtcnn`, `haar`, or `auto` |
| `DEEPFAKE_DB` | `results/predictions.db` | SQLite history location |
| `DEEPFAKE_API_KEY` | _(unset)_ | If set, inference endpoints require header `X-API-Key` |
| `DEEPFAKE_RATE_LIMIT` | `0` | Max requests/min per client (0 = unlimited) |
| `DEEPFAKE_ADMIN_USER` | `admin` | Default UI login username (bootstrapped on first run) |
| `DEEPFAKE_ADMIN_PASSWORD` | `admin123` | Default UI login password — **change in production** |
| `DEEPFAKE_DISABLE_LOGIN` | _(unset)_ | Set to `1` to disable the Gradio login page (public demo) |

---

## Checkpoint auto-download (recommended for cloud)

`models/*.pth` is gitignored (weights are large). Instead of baking the file in,
let the app fetch it on first startup by setting **one** of these env vars:

```bash
# Option A — direct URL (no extra dependency)
export DEEPFAKE_CHECKPOINT_URL="https://example.com/best_xception_celebdf.pth"

# Option B — Hugging Face Hub (needs huggingface_hub, already in requirements.txt)
export DEEPFAKE_HF_REPO="your-username/deepfake-xception"
# export DEEPFAKE_HF_FILENAME="best_xception_celebdf.pth"   # defaults to the checkpoint basename
# export DEEPFAKE_HF_REVISION="main"                        # optional
```

The first prediction (or `python -m src.checkpoint`) downloads the file to
`models/` and every entry point — Gradio, Streamlit, FastAPI, CLI — uses it.
Trigger it manually with:

```bash
python -m src.checkpoint
```

Without a checkpoint (and no download source), the apps still start and render
but show a clear "model not found" message instead of predictions.
