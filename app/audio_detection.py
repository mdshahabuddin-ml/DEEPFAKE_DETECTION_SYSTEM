"""Audio Detection page for Deepfake Detection System.

This is a NEW standalone page that can be integrated into the existing
Streamlit app without modifying any existing files.

To integrate, add this to the navigation in app.py:
    if st.sidebar.button("🎤 Audio Detection"):
        st.session_state.page = "audio"
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import numpy as np

from audio_forensics.config import AudioForensicsConfig
from audio_forensics.inference import AudioInferencePipeline
from audio_forensics.segmentation import AudioSegmenter


def render_audio_detection():
    """Render the Audio Detection page."""

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🎤 Audio Deepfake Detection</h1>
        <p style="color: #e0e0e0; margin: 10px 0 0 0;">
            Upload audio files to detect synthetic speech, voice cloning,
            and audio manipulation using AI analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Info cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 3px solid #667eea;">
            <p style="color: #888; margin: 0; font-size: 12px;">ANALYSIS TYPE</p>
            <p style="color: #fff; margin: 5px 0 0 0; font-weight: bold;">Audio Forensics</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 3px solid #764ba2;">
            <p style="color: #888; margin: 0; font-size: 12px;">DETECTION TARGET</p>
            <p style="color: #fff; margin: 5px 0 0 0; font-weight: bold;">Voice Cloning & TTS</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: #1e1e2e; padding: 15px; border-radius: 10px; border-left: 3px solid #00d4aa;">
            <p style="color: #888; margin: 0; font-size: 12px;">FEATURES</p>
            <p style="color: #fff; margin: 5px 0 0 0; font-weight: bold;">Segmentation & Timeline</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Threshold slider
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Lower values flag more samples as manipulated"
    )

    st.markdown(f"""
    <div style="background: #1e1e2e; padding: 10px 15px; border-radius: 8px; margin-bottom: 20px;">
        <span style="color: #888;">Verdict fires at </span>
        <span style="color: #667eea; font-weight: bold;">{threshold:.2f}</span>
        <span style="color: #888;"> — lower values flag more samples as manipulated</span>
    </div>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        help="Supported formats: WAV, MP3, FLAC, OGG, M4A"
    )

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Initialize pipeline
            config = AudioForensicsConfig(fake_threshold=threshold)
            pipeline = AudioInferencePipeline(config)

            # Run analysis
            with st.spinner("Analyzing audio..."):
                result = pipeline.analyze(tmp_path)

            # Display results
            display_audio_results(result, threshold)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

        finally:
            # Clean up
            os.unlink(tmp_path)

    # Features section
    st.markdown("---")
    st.markdown("### Capabilities")

    feat_col1, feat_col2, feat_col3 = st.columns(3)

    with feat_col1:
        st.markdown("""
        <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; height: 180px;">
            <div style="font-size: 30px; margin-bottom: 10px;">🔍</div>
            <h4 style="color: #fff; margin: 0 0 10px 0;">Voice Activity Detection</h4>
            <p style="color: #888; margin: 0; font-size: 14px;">
                Automatically segments speech from silence for targeted analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with feat_col2:
        st.markdown("""
        <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; height: 180px;">
            <div style="font-size: 30px; margin-bottom: 10px;">🎵</div>
            <h4 style="color: #fff; margin: 0 0 10px 0;">Feature Extraction</h4>
            <p style="color: #888; margin: 0; font-size: 14px;">
                Log-Mel spectrograms, MFCC, and spectral features for detection.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with feat_col3:
        st.markdown("""
        <div style="background: #1e1e2e; padding: 20px; border-radius: 10px; height: 180px;">
            <div style="font-size: 30px; margin-bottom: 10px;">📊</div>
            <h4 style="color: #fff; margin: 0 0 10px 0;">Temporal Timeline</h4>
            <p style="color: #888; margin: 0; font-size: 14px;">
                Visual timeline showing suspicious segments with probability scores.
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_audio_results(result, threshold):
    """Display audio analysis results."""

    # Overall verdict
    prediction = result.overall_prediction.value
    probability = result.overall_probability

    if prediction in ["POTENTIAL_MANIPULATION", "SUSPICIOUS"]:
        bg_color = "#dc3545"
        icon = "⚠️"
        verdict_text = "POTENTIALLY MANIPULATED"
    elif prediction == "LOW_SUSPICION":
        bg_color = "#ffc107"
        icon = "⚡"
        verdict_text = "LOW SUSPICION"
    else:
        bg_color = "#28a745"
        icon = "✅"
        verdict_text = "AUTHENTIC"

    st.markdown(f"""
    <div style="background: {bg_color}20; border: 2px solid {bg_color};
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <span style="font-size: 40px;">{icon}</span>
            <div>
                <h2 style="color: {bg_color}; margin: 0;">{verdict_text}</h2>
                <p style="color: #888; margin: 5px 0 0 0;">
                    Confidence: {probability:.1%}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)

    with met_col1:
        st.metric("Duration", f"{result.properties.duration:.1f}s")
    with met_col2:
        st.metric("Sample Rate", f"{result.properties.sample_rate} Hz")
    with met_col3:
        st.metric("Segments", result.total_segments)
    with met_col4:
        st.metric("Suspicious", result.suspicious_segments)

    # Segment details
    if result.segments:
        st.markdown("### Segment Analysis")

        for seg in result.segments:
            if seg.prediction.value in ["SUSPICIOUS", "POTENTIAL_MANIPULATION"]:
                seg_color = "#dc3545"
            elif seg.prediction.value == "LOW_SUSPICION":
                seg_color = "#ffc107"
            else:
                seg_color = "#28a745"

            st.markdown(f"""
            <div style="background: #1e1e2e; padding: 12px 15px; border-radius: 8px;
                        border-left: 3px solid {seg_color}; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #fff;">
                        {seg.start_time:.1f}s — {seg.end_time:.1f}s
                    </span>
                    <span style="color: {seg_color}; font-weight: bold;">
                        {seg.prediction.value} ({seg.probability:.1%})
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Evidence
    if result.evidence:
        st.markdown("### Evidence")
        for ev in result.evidence:
            st.markdown(f"- {ev}")

    # Warnings
    if result.warnings:
        st.markdown("### Warnings")
        for w in result.warnings:
            st.warning(w)


if __name__ == "__main__":
    render_audio_detection()
