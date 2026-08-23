# AI-Powered Explainable Deepfake Detection System
# Single image that can run either the FastAPI service or the Streamlit UI
# (selected by the command in docker-compose.yml).

FROM python:3.11-slim

# System libraries required by OpenCV, video/audio decoding, and soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU PyTorch wheels + the rest of the requirements.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Copy the application source.
COPY . .

# Runtime directories (also created by src/config.py at import time).
RUN mkdir -p models uploads outputs logs results

# FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default command runs the API; docker-compose overrides for the UI service.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
