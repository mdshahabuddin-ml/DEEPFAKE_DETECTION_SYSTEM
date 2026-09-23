"""
Standalone Audio Voice Detection Application.

Run independently:
    streamlit run run_audio_detection.py --server.port 8502

This does NOT modify app.py or any existing file.
"""

from __future__ import annotations

import sys
import os
import time
import tempfile

import streamlit as st
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Audio Voice Detection",
    page_icon="\U0001f3a4",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] label { color: #e0e0e0; }
    .sidebar-header { padding: 0 0 10px 0; border-bottom: 1px solid #333; margin-bottom: 15px; }
    .sidebar-header h2 { color: #fff; margin: 0; font-size: 22px; font-weight: 700; }
    .info-card {
        background: #1e1e2e; padding: 20px; border-radius: 12px;
        border-top: 3px solid #667eea; text-align: center; min-height: 100px;
    }
    .info-card h3 { color: #888; margin: 0; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
    .info-card p { color: #667eea; margin: 8px 0 0 0; font-size: 22px; font-weight: 700; }
    .cap-card {
        background: #1e1e2e; padding: 25px; border-radius: 12px;
        border: 1px solid #2a2a3e; min-height: 160px;
    }
    .cap-card h4 { color: #fff; margin: 0 0 8px 0; font-size: 16px; }
    .cap-card p { color: #888; margin: 0; font-size: 13px; line-height: 1.5; }
    .hero-section {
        background: linear-gradient(135deg, #1a1a3e 0%, #2d2b55 50%, #1a1a3e 100%);
        padding: 40px; border-radius: 16px; margin-bottom: 30px; border: 1px solid #333;
    }
    .hero-badge {
        display: inline-block; background: rgba(102,126,234,0.2); color: #667eea;
        padding: 4px 14px; border-radius: 20px; font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;
    }
    .hero-section h1 { color: #4da3ff; margin: 0 0 15px 0; font-size: 32px; font-weight: 700; }
    .hero-section p { color: #b0b0b0; margin: 0; font-size: 15px; line-height: 1.6; }
    .section-label { color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 15px; font-weight: 600; }
    .verdict-card { padding: 20px; border-radius: 12px; margin-bottom: 20px; border: 2px solid; }
    .segment-card { background: #1e1e2e; padding: 12px 15px; border-radius: 8px; border-left: 3px solid; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Import audio modules (isolated from existing app)
# ---------------------------------------------------------------------------
_FORENSICS_AVAILABLE = False
_SIMPLE_AVAILABLE = False

try:
    from audio_forensics.config import AudioForensicsConfig
    from audio_forensics.inference import AudioInferencePipeline
    _FORENSICS_AVAILABLE = True
except Exception:
    pass

try:
    from src.audio_infer import load_audio_detector, predict_from_array
    from src.audio_dataset import SAMPLE_RATE
    _SIMPLE_AVAILABLE = True
except Exception:
    pass

if not _FORENSICS_AVAILABLE and not _SIMPLE_AVAILABLE:
    st.error("No audio detection backend available.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>Audio Voice Detection</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p style="color:#888;font-size:13px;">'
        "Standalone audio deepfake detection system.<br>"
        "Upload audio files to detect synthetic speech, voice cloning, and manipulation.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<p class="section-label">Configuration</p>', unsafe_allow_html=True)

    threshold = st.slider(
        "Decision Threshold", 0.0, 1.0, 0.5, 0.01,
        help="Lower values flag more samples as manipulated",
    )
    st.markdown(
        f'<p style="color:#888;font-size:12px;margin:0;">'
        f'Verdict fires at <span style="color:#667eea;font-weight:700;">{threshold:.2f}</span></p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    backend = "audio_forensics" if _FORENSICS_AVAILABLE else "src/audio_infer"
    st.markdown(
        f'<p style="color:#888;font-size:12px;">'
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:#28a745;margin-right:6px;"></span>'
        f'Backend: <code>{backend}</code></p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#555;font-size:11px;">Audio Voice Detection v1.0<br>Standalone Module</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <div class="hero-badge">AUDIO SECURITY</div>
    <h1>Audio Voice Deepfake Detection</h1>
    <p>
        Upload audio files to detect synthetic speech, voice cloning, and audio
        manipulation using AI analysis. Supports WAV, MP3, FLAC, OGG, and M4A formats.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="info-card"><h3>Analysis Type</h3><p>Audio Forensics</p></div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="info-card"><h3>Detection Target</h3><p>Voice Cloning & TTS</p></div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="info-card"><h3>Features</h3><p>Segmentation & Timeline</p></div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="section-label">Capabilities</p>', unsafe_allow_html=True)
row = st.columns(3)
with row[0]:
    st.markdown("""<div class="cap-card"><h4>Voice Activity Detection</h4><p>Automatically segments speech from silence for targeted analysis.</p></div>""", unsafe_allow_html=True)
with row[1]:
    st.markdown("""<div class="cap-card"><h4>Feature Extraction</h4><p>Log-Mel spectrograms, MFCC, and spectral features for robust detection.</p></div>""", unsafe_allow_html=True)
with row[2]:
    st.markdown("""<div class="cap-card"><h4>Temporal Timeline</h4><p>Visual timeline showing suspicious segments with probability scores.</p></div>""", unsafe_allow_html=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Upload and Analyze
# ---------------------------------------------------------------------------
st.markdown("## Upload Audio")
st.markdown('<p class="section-label">Select an audio file for deepfake analysis</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Supported: WAV, MP3, FLAC, OGG, M4A",
    key="audio_upload",
)

if uploaded_file is not None:
    t0 = time.time()
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        st.audio(uploaded_file)

        if _FORENSICS_AVAILABLE:
            with st.spinner("Running full audio forensics analysis..."):
                config = AudioForensicsConfig(fake_threshold=threshold)
                pipeline = AudioInferencePipeline(config)
                result = pipeline.analyze(tmp_path)

            elapsed = time.time() - t0
            pred_val = result.overall_prediction.value
            prob = result.overall_probability

            if pred_val in ("POTENTIAL_MANIPULATION", "SUSPICIOUS"):
                bg, icon, text = "#dc3545", "\u26a0\ufe0f", "POTENTIALLY MANIPULATED"
            elif pred_val == "LOW_SUSPICION":
                bg, icon, text = "#ffc107", "\u26a1", "LOW SUSPICION"
            else:
                bg, icon, text = "#28a745", "\u2705", "AUTHENTIC"

            st.markdown(f"""
            <div class="verdict-card" style="background: {bg}20; border-color: {bg};">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 40px;">{icon}</span>
                    <div>
                        <h2 style="color: {bg}; margin: 0;">{text}</h2>
                        <p style="color: #888; margin: 5px 0 0 0;">
                            Confidence: {result.overall_confidence:.1%} |
                            Probability: {prob:.3f} |
                            Time: {elapsed:.2f}s
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Duration", f"{result.properties.duration:.1f}s")
            m2.metric("Sample Rate", f"{result.properties.sample_rate} Hz")
            m3.metric("Segments", result.total_segments)
            m4.metric("Suspicious", result.suspicious_segments)

            if result.segments:
                st.markdown("### Segment Analysis")
                for seg in result.segments:
                    if seg.prediction.value in ("SUSPICIOUS", "POTENTIAL_MANIPULATION"):
                        sc = "#dc3545"
                    elif seg.prediction.value == "LOW_SUSPICION":
                        sc = "#ffc107"
                    else:
                        sc = "#28a745"
                    st.markdown(f"""
                    <div class="segment-card" style="border-left-color: {sc};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #fff;">{seg.start_time:.1f}s - {seg.end_time:.1f}s</span>
                            <span style="color: {sc}; font-weight: bold;">{seg.prediction.value} ({seg.probability:.1%})</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            if result.evidence:
                st.markdown("### Evidence")
                for ev in result.evidence:
                    st.markdown(f"- {ev}")

            if result.warnings:
                st.markdown("### Warnings")
                for w in result.warnings:
                    st.warning(w)

        elif _SIMPLE_AVAILABLE:
            with st.spinner("Analyzing audio (simple backend)..."):
                import librosa
                audio_model, audio_device = load_audio_detector()
                y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
                prob = predict_from_array(y, audio_model, audio_device)

            elapsed = time.time() - t0
            label = "FAKE" if prob >= threshold else "REAL"
            bg = "#dc3545" if label == "FAKE" else "#28a745"
            icon = "\u26a0\ufe0f" if label == "FAKE" else "\u2705"

            st.markdown(f"""
            <div class="verdict-card" style="background: {bg}20; border-color: {bg};">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <span style="font-size: 40px;">{icon}</span>
                    <div>
                        <h2 style="color: {bg}; margin: 0;">{label}</h2>
                        <p style="color: #888; margin: 5px 0 0 0;">
                            Fake probability: {prob:.4f} |
                            Threshold: {threshold:.2f} |
                            Time: {elapsed:.2f}s
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(prob, 0.0), 1.0))

            try:
                import matplotlib.pyplot as plt
                from src.audio_dataset import waveform_to_logmel, fix_waveform_length

                y_fixed = fix_waveform_length(y, sr=SAMPLE_RATE)
                logmel = waveform_to_logmel(y_fixed, sr=SAMPLE_RATE)

                fig, axes = plt.subplots(2, 1, figsize=(8, 4), constrained_layout=True,
                                         facecolor="#0e1117")
                axes[0].plot(np.linspace(0, len(y_fixed) / SAMPLE_RATE, len(y_fixed)),
                             y_fixed, linewidth=0.5, color="#667eea")
                axes[0].set_title("Waveform", color="#e0e0e0")
                axes[0].set_xlabel("Time (s)", color="#888")
                axes[0].set_facecolor("#1e1e2e")
                axes[0].tick_params(colors="#888")

                im = axes[1].imshow(logmel, aspect="auto", origin="lower", cmap="magma")
                axes[1].set_title("Log-mel spectrogram", color="#e0e0e0")
                axes[1].set_ylabel("Mel bin", color="#888")
                axes[1].set_facecolor("#1e1e2e")
                axes[1].tick_params(colors="#888")
                fig.colorbar(im, ax=axes[1], format="%+2.0f dB")

                st.pyplot(fig)
            except Exception:
                pass

    except Exception as e:
        st.error(f"Analysis failed: {e}")
    finally:
        os.unlink(tmp_path)


st.markdown("---")
st.markdown(
    '<p style="color:#555;font-size:12px;text-align:center;">'
    "Audio Voice Detection - Standalone Module - Independent of main application</p>",
    unsafe_allow_html=True,
)
