"""
app/dashboard.py
Enhanced dashboard page for Deepfake Detection System.

Displays advanced analysis results from all detection modules
in a comprehensive, visually rich layout.

To integrate, add to app.py navigation:
    if st.sidebar.button("Dashboard"):
        st.session_state.page = "dashboard"
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st


def render_dashboard():
    """Render the enhanced dashboard page."""

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                padding: 30px; border-radius: 15px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">Detection Dashboard</h1>
        <p style="color: #b0b0b0; margin: 10px 0 0 0;">
            Comprehensive analysis results from all detection modules.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Module status cards
    _render_module_status()

    st.markdown("---")

    # Analysis history summary
    _render_analysis_summary()

    st.markdown("---")

    # Recent analyses
    _render_recent_analyses()


def _render_module_status():
    """Show status of all detection modules."""
    st.markdown("### Module Status")

    try:
        from api.advanced import list_modules
        status = list_modules()
        modules = status.get("modules", [])
    except Exception:
        modules = [
            {"name": "audio_forensics", "available": False, "description": "Audio deepfake detection"},
            {"name": "temporal_analysis", "available": False, "description": "Temporal consistency"},
            {"name": "multimodal_fusion", "available": False, "description": "Multi-modal fusion"},
            {"name": "media_forensics", "available": False, "description": "File integrity"},
            {"name": "forensic_report", "available": False, "description": "PDF reports"},
        ]

    cols = st.columns(min(len(modules), 5))
    for i, mod in enumerate(modules):
        with cols[i % len(cols)]:
            color = "#00d4aa" if mod["available"] else "#dc3545"
            status_text = "ACTIVE" if mod["available"] else "INACTIVE"
            st.markdown(f"""
            <div style="background: #1e1e2e; padding: 15px; border-radius: 10px;
                        border-top: 3px solid {color}; min-height: 120px;">
                <p style="color: {color}; margin: 0; font-size: 11px;
                          font-weight: bold; text-transform: uppercase;">{status_text}</p>
                <h4 style="color: #fff; margin: 8px 0 5px 0;">
                    {mod['name'].replace('_', ' ').title()}</h4>
                <p style="color: #888; margin: 0; font-size: 12px;">
                    {mod['description']}</p>
            </div>
            """, unsafe_allow_html=True)


def _render_analysis_summary():
    """Show summary of analysis statistics."""
    st.markdown("### Analysis Statistics")

    try:
        from src.database import get_db
        db = get_db()
        stats = db.stats()
        total = stats.get("total", 0)
        fake = stats.get("fake", 0)
        real = stats.get("real", 0)
        avg_time = stats.get("avg_inference_time", 0)
    except Exception:
        total, fake, real, avg_time = 0, 0, 0, 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        _stat_card("Total Analyses", str(total), "#667eea")
    with col2:
        _stat_card("Fake Detected", str(fake), "#dc3545")
    with col3:
        _stat_card("Authentic", str(real), "#28a745")
    with col4:
        _stat_card("Avg Inference Time", f"{avg_time:.2f}s", "#ffc107")


def _stat_card(label: str, value: str, color: str):
    """Render a single stat card."""
    st.markdown(f"""
    <div style="background: #1e1e2e; padding: 20px; border-radius: 10px;
                border-left: 4px solid {color}; text-align: center;">
        <p style="color: {color}; font-size: 28px; font-weight: bold;
                  margin: 0;">{value}</p>
        <p style="color: #888; margin: 5px 0 0 0; font-size: 13px;">{label}</p>
    </div>
    """, unsafe_allow_html=True)


def _render_recent_analyses():
    """Show recent analysis results."""
    st.markdown("### Recent Analyses")

    try:
        from src.database import get_db
        db = get_db()
        records = db.recent(limit=20)
    except Exception:
        records = []

    if records:
        for rec in records:
            pred = rec.get("prediction", "UNKNOWN")
            color = "#dc3545" if pred == "FAKE" else "#28a745" if pred == "REAL" else "#ffc107"
            st.markdown(
                f'<div style="background: #1e1e2e; padding: 12px 16px; border-radius: 8px; '
                f'border-left: 3px solid {color}; margin-bottom: 8px; '
                f'display: flex; justify-content: space-between; align-items: center;">'
                f'<div><span style="color: #fff; font-weight: 600;">{rec.get("file_name", "Unknown")}</span>'
                f'<span style="color: #888; margin-left: 10px; font-size: 12px;">{rec.get("media_type", "N/A")}</span></div>'
                f'<div><span style="color: {color}; font-weight: 600;">{pred}</span>'
                f'<span style="color: #888; margin-left: 10px; font-size: 12px;">{rec.get("confidence", 0):.1%}</span></div>'
                f'</div>',
                unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #1e1e2e; padding: 40px; border-radius: 10px;
                    text-align: center;">
            <p style="color: #666; font-size: 16px; margin: 0;">
                No analyses yet. Upload a file to get started.
            </p>
            <p style="color: #444; font-size: 13px; margin: 10px 0 0 0;">
                Use the Image, Video, or Audio tabs to run detections.
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_module_detail(module_name: str, module_data: Dict[str, Any]):
    """Render detailed results for a specific module."""
    st.markdown(f"#### {module_name.replace('_', ' ').title()}")

    if not module_data.get("success", False):
        st.error(f"Module failed: {module_data.get('error', 'Unknown error')}")
        return

    data = module_data.get("data", {})
    pt = module_data.get("processing_time", 0)

    st.caption(f"Processing time: {pt:.2f}s")

    if module_name == "audio_forensics":
        _render_audio_detail(data)
    elif module_name == "temporal_analysis":
        _render_temporal_detail(data)
    elif module_name == "multimodal_fusion":
        _render_fusion_detail(data)
    elif module_name == "media_forensics":
        _render_media_detail(data)


def _render_audio_detail(data: Dict[str, Any]):
    """Render audio forensics detail."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", data.get("overall_prediction", "N/A"))
    col2.metric("Probability", f"{data.get('overall_probability', 0):.3f}")
    col3.metric("Suspicious Segments",
                f"{data.get('suspicious_segments', 0)}/{data.get('total_segments', 0)}")

    if data.get("evidence"):
        st.markdown("**Evidence:**")
        for ev in data["evidence"]:
            st.markdown(f"- {ev}")

    if data.get("timeline"):
        st.markdown("**Timeline:**")
        for seg in data["timeline"][:10]:
            prob = seg.get("probability", 0)
            color = "#dc3545" if prob > 0.5 else "#28a745"
            st.markdown(
                f"  {seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s: "
                f"`{seg.get('prediction', 'N/A')}` ({prob:.3f})"
            )


def _render_temporal_detail(data: Dict[str, Any]):
    """Render temporal analysis detail."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Consistency", f"{data.get('consistency_score', 0):.3f}")
    col2.metric("Anomaly Score", f"{data.get('anomaly_score', 0):.3f}")
    suspicious = data.get("is_suspicious", False)
    col3.metric("Suspicious", "Yes" if suspicious else "No")

    if data.get("evidence"):
        st.markdown("**Evidence:**")
        for ev in data["evidence"]:
            st.markdown(f"- {ev}")


def _render_fusion_detail(data: Dict[str, Any]):
    """Render multimodal fusion detail."""
    col1, col2 = st.columns(2)
    col1.metric("Verdict", data.get("verdict", "N/A"))
    col2.metric("Confidence", f"{data.get('confidence', 0):.3f}")

    scores = data.get("scores", {})
    if scores:
        st.markdown("**Component Scores:**")
        for key, val in scores.items():
            bar_val = min(max(val, 0.0), 1.0)
            st.progress(bar_val, text=f"{key}: {val:.3f}")


def _render_media_detail(data: Dict[str, Any]):
    """Render media forensics detail."""
    for key, val in data.items():
        if key not in ("evidence", "recommendations"):
            st.markdown(f"**{key}:** {val}")
