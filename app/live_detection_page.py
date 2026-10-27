"""
app/live_detection_page.py

Professional Live Webcam + Audio deepfake detection page.

Renders the "Live Detection" view for app.py. Reuses existing inference:
  - Video model bundle passed in from app.py (Xception + FaceDetector)
  - Audio model loaded lazily via src.audio_infer (same checkpoint as Audio page)
  - Fusion via src.fusion.fuse_scores through src.live_detection.LiveSession

Server-side capture (OpenCV + sounddevice) is used so live detection works
without streamlit-webrtc. All device access is optional and degrades
gracefully to UNAVAILABLE status cards + snapshot fallback.
"""

from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

from src.live_detection import (
    LiveSession,
    LiveSessionConfig,
    audio_backend_available,
    classify_live,
    list_cameras,
    list_microphones,
)


# ---------------------------------------------------------------------------
# Cached helpers (models / device probes)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading live audio detector...")
def _get_live_audio_bundle():
    """Load the audio model once per server process. Returns bundle or None."""
    try:
        from src.audio_infer import (
            load_audio_detector,
            DEFAULT_AUDIO_MODEL,
            DEFAULT_AUDIO_CHECKPOINT,
        )
        from src.audio_dataset import SAMPLE_RATE, DURATION
        import os

        if not os.path.exists(DEFAULT_AUDIO_CHECKPOINT):
            return None
        model, device = load_audio_detector(DEFAULT_AUDIO_MODEL, DEFAULT_AUDIO_CHECKPOINT)
        return {
            "model": model,
            "device": device,
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
            "name": DEFAULT_AUDIO_MODEL,
        }
    except Exception:
        return None


@st.cache_data(ttl="60s", show_spinner=False)
def _cached_cameras():
    try:
        return list_cameras()
    except Exception:
        return [{"index": 0, "label": "Camera 0 (default)"}]


@st.cache_data(ttl="60s", show_spinner=False)
def _cached_mics():
    try:
        return list_microphones()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Small presentational helpers
# ---------------------------------------------------------------------------

_LABEL_COLOR = {
    "FAKE": "#dc3545",
    "REAL": "#28a745",
    "UNCERTAIN": "#ffc107",
    "NO_SIGNAL": "#5a6b7e",
    "UNAVAILABLE": "#5a6b7e",
}


def _pill(label: str) -> str:
    color = _LABEL_COLOR.get(label, "#5a6b7e")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'background:rgba(20,28,38,.9);color:{color};border:1px solid {color}55;'
        f'padding:6px 14px;border-radius:8px;font-size:13px;font-weight:700;'
        f'letter-spacing:.5px;">{label}</span>'
    )


def _prob_bar(prob: float | None, color: str) -> str:
    pct = 0.0 if prob is None else float(min(max(prob, 0.0), 1.0)) * 100.0
    txt = "--" if prob is None else f"{prob:.1%}"
    return (
        f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
        f'<div style="flex:1;height:6px;background:#1a2332;border-radius:3px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct:.1f}%;background:{color};border-radius:3px;"></div>'
        f"</div>"
        f'<span style="color:#c0c8d4;font-size:12px;font-weight:600;min-width:52px;text-align:right;">{txt}</span>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render_live_detection_page(
    model,
    device: str,
    transform,
    face_cascade,
    threshold: float,
    store_prediction,
    model_name: str = "xception",
) -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        .ld-hdr { background: linear-gradient(135deg, #0f1923 0%, #1a2332 50%, #0f1923 100%);
            border: 1px solid #2a3a4e; border-radius: 12px; padding: 18px 24px;
            margin-bottom: 16px; position: relative; overflow: hidden; }
        .ld-hdr::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent); }
        .ld-badge { display: inline-flex; align-items: center; gap: 5px;
            background: rgba(102,126,234,.12); color: #667eea; padding: 3px 10px;
            border-radius: 12px; font-size: 9px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px;
            border: 1px solid rgba(102,126,234,.2); }
        .ld-title { color: #e8edf3; margin: 0 0 4px 0; font-size: 20px; font-weight: 700; }
        .ld-sub { color: #7a8ba0; margin: 0; font-size: 12px; line-height: 1.4; }
        .ld-live { display: inline-flex; align-items: center; gap: 6px;
            background: rgba(220,53,69,.12); color: #dc3545; padding: 4px 12px;
            border-radius: 8px; font-size: 11px; font-weight: 700; letter-spacing: 1px;
            border: 1px solid rgba(220,53,69,.25); }
        .ld-live .ld-dot { width: 7px; height: 7px; border-radius: 50%; background: #dc3545;
            animation: ld-pulse 1.4s ease-in-out infinite; }
        @keyframes ld-pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
        .ld-idle { display: inline-flex; align-items: center; gap: 6px;
            background: rgba(90,107,126,.12); color: #5a6b7e; padding: 4px 12px;
            border-radius: 8px; font-size: 11px; font-weight: 700; letter-spacing: 1px;
            border: 1px solid rgba(90,107,126,.2); }
        .ld-idle .ld-dot { width: 7px; height: 7px; border-radius: 50%; background: #5a6b7e; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ld-hdr">
            <div class="ld-badge">Multimodal forensics</div>
            <h1 class="ld-title">Live Webcam + Audio Detection</h1>
            <p class="ld-sub">Simultaneous face and voice analysis with fused AI verdict in real time.
            Video: Xception face detector &bull; Audio: spectrogram detector &bull; Fusion: weighted average.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -- session state -----------------------------------------------------
    st.session_state.setdefault("live_running", False)
    st.session_state.setdefault("live_session", None)
    st.session_state.setdefault("live_last_snap", None)
    st.session_state.setdefault("live_history", [])
    st.session_state.setdefault("live_cam_idx", 0)
    st.session_state.setdefault("live_mic_idx", None)

    running: bool = bool(st.session_state.live_running)
    session: LiveSession | None = st.session_state.live_session

    # -- device selects ----------------------------------------------------
    cams = _cached_cameras()
    cam_labels = [c["label"] for c in cams]
    cam_indices = [c["index"] for c in cams]
    default_cam_pos = cam_indices.index(st.session_state.live_cam_idx) \
        if st.session_state.live_cam_idx in cam_indices else 0

    mics = _cached_mics()
    mic_ok, mic_reason = audio_backend_available()
    audio_bundle = _get_live_audio_bundle()

    ctrl_top = st.columns([1.2, 1.2, 1], gap="medium")
    with ctrl_top[0]:
        cam_pos = st.selectbox(
            "Camera", options=list(range(len(cam_labels))),
            format_func=lambda i: cam_labels[i], index=default_cam_pos,
            disabled=running,
            help="Server-side camera device. Change while stopped.",
        )
        st.session_state.live_cam_idx = cam_indices[cam_pos]
    with ctrl_top[1]:
        if not mic_ok or not mics:
            st.selectbox("Microphone", options=["Default input"],
                         disabled=True,
                         help=f"Mic selection unavailable: {mic_reason}"
                         if not mic_ok else "No input devices reported.")
        else:
            mic_labels = ["Default input"] + [m["label"] for m in mics]
            mic_pos = st.selectbox("Microphone", options=list(range(len(mic_labels))),
                                   format_func=lambda i: mic_labels[i],
                                   disabled=running,
                                   help="Microphone input device. Change while stopped.")
            st.session_state.live_mic_idx = None if mic_pos == 0 else mics[mic_pos - 1]["index"]
    with ctrl_top[2]:
        fusion_w = st.slider("Video weight", 0.0, 1.0, 0.5, 0.05, disabled=running,
                             help="Fusion = w·video + (1−w)·audio. Missing modality is dropped.")

    with st.expander("Advanced live settings"):
        adv = st.columns(3)
        with adv[0]:
            infer_every_n = st.slider("Inference every N frames", 1, 6, 2, disabled=running)
        with adv[1]:
            smooth_win = st.slider("Video smoothing window", 3, 16, 8, disabled=running)
        with adv[2]:
            uncertain_margin = st.slider("Uncertain margin", 0.0, 0.25, 0.10, 0.01, disabled=running,
                                         help="Predictions within ±margin of the threshold report UNCERTAIN.")

    # -- availability banners (graceful degradation) ------------------------
    if not mic_ok:
        st.warning(f"Microphone capture unavailable — {mic_reason}. "
                   "Video analysis still works; audio card will show UNAVAILABLE.")
    elif audio_bundle is None:
        st.warning("Audio model checkpoint not found — audio card will show UNAVAILABLE. "
                   "Video analysis still works.")

    # -- Start / Stop --------------------------------------------------------
    btn_col = st.columns([1, 1, 2], gap="medium")
    with btn_col[0]:
        start_clicked = st.button(":material/play_arrow: Start live analysis",
                                  type="primary", width="stretch", disabled=running)
    with btn_col[1]:
        stop_clicked = st.button(":material/stop: Stop & save", width="stretch", disabled=not running)
    with btn_col[2]:
        status_html = ('<span class="ld-live"><span class="ld-dot"></span>LIVE</span>'
                       if running else '<span class="ld-idle"><span class="ld-dot"></span>IDLE</span>')
        st.markdown(f"<div style='padding-top:6px;'>{status_html}</div>", unsafe_allow_html=True)

    if start_clicked and not running:
        cfg = LiveSessionConfig(
            camera_index=int(st.session_state.live_cam_idx),
            mic_index=st.session_state.live_mic_idx,
            threshold=float(threshold),
            uncertain_margin=float(uncertain_margin),
            infer_every_n=int(infer_every_n),
            smooth_window=int(smooth_win),
            fusion_w_video=float(fusion_w),
        )
        sess = LiveSession(cfg)
        sess.start(
            video_bundle={"model": model, "device": device,
                          "transform": transform, "face_detector": face_cascade},
            audio_bundle=audio_bundle,
        )
        st.session_state.live_session = sess
        st.session_state.live_running = True
        st.rerun()

    if stop_clicked and running and session is not None:
        snap = session.stop()
        st.session_state.live_last_snap = snap
        st.session_state.live_running = False
        # Persist the completed analysis to prediction history.
        try:
            f = snap["fusion"]
            label = f["label"] if f["label"] in ("REAL", "FAKE", "UNCERTAIN") else "UNCERTAIN"
            # DB schema predates UNCERTAIN; store it — history page colors REAL green,
            # everything else is styled by exact match (FAKE red). UNCERTAIN stays neutral.
            store_prediction(
                file_name=f"live_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                media_type="live",
                prediction=label,
                confidence=float(f["confidence"]),
                fake_prob=float(f["prob"]) if f["prob"] is not None else 0.5,
                inference_time=float(snap.get("uptime_sec", 0.0)),
            )
            st.session_state.live_history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "verdict": label,
                "fused": f["prob"],
                "video": snap["video"]["prob"],
                "audio": snap["audio"]["prob"],
                "uptime": snap.get("uptime_sec", 0.0),
            })
        except Exception:
            pass
        st.session_state.live_session = None
        st.rerun()

    # Refresh local handles after potential rerun-triggering clicks above.
    running = bool(st.session_state.live_running)
    session = st.session_state.live_session

    # -- live dashboard (auto-refreshing fragment) ---------------------------
    @st.fragment(run_every="1s")
    def _live_dashboard() -> None:
        sess: LiveSession | None = st.session_state.get("live_session")
        is_run: bool = bool(st.session_state.get("live_running")) and sess is not None
        if not is_run or sess is None:
            last = st.session_state.get("live_last_snap")
            if last is not None:
                st.caption("Last session verdict: "
                           f"{last['fusion']['label']} "
                           f"(fused {last['fusion']['prob']:.1%}) — "
                           "press Start for a new analysis.")
            else:
                with st.container(border=True):
                    st.subheader("Camera preview")
                    st.info("Press **Start live analysis** to open the camera and microphone. "
                            "Grant OS permissions if prompted. "
                            "If no camera is attached, use the snapshot fallback below.")
            return

        snap = sess.snapshot()
        v, a, f = snap["video"], snap["audio"], snap["fusion"]

        if v.get("error"):
            st.error(f"Camera issue: {v['error']}")

        prev_col, side_col = st.columns([3, 2], gap="medium")
        with prev_col:
            with st.container(border=True):
                st.subheader("Camera preview")
                jpeg = sess.latest_frame_jpeg
                if jpeg:
                    st.image(jpeg, channels="BGR", width="stretch")
                else:
                    st.info("Warming up camera…")
                st.caption(f"Faces detected: {v['faces']} · Frames: {v['frames']} · "
                           f"Smoothing over last {len(v['history'])} predictions")
                rej = int(v.get('rejected_small', 0)) + int(v.get('rejected_blur', 0))
                if rej:
                    st.caption(f"Skipped low-quality frames: {rej} "
                               f"(small: {v.get('rejected_small', 0)}, "
                               f"blurred: {v.get('rejected_blur', 0)}) · "
                               f"Last: {v.get('last_reject') or '—'}")

        with side_col:
            with st.container(border=True):
                st.subheader("Detection status")
                st.markdown(_pill(f["label"]), unsafe_allow_html=True)
                st.caption(f"Combined verdict · confidence {f['confidence']:.0%} · "
                           f"threshold {snap['threshold']:.2f}")
                if f.get("uncertain"):
                    st.warning("Uncertain — score is near the decision boundary. "
                               "Hold still, improve lighting, speak clearly.")
                st.markdown(_prob_bar(f["prob"], "#667eea"), unsafe_allow_html=True)

            with st.container(border=True):
                st.subheader("Performance")
                pcol = st.columns(2)
                pcol[0].metric("FPS", f"{v['fps']:.1f}")
                pcol[1].metric("Latency", f"{v['latency_ms']:.0f} ms")
                st.caption(f"Device {str(device).upper()} · {model_name.upper()} · "
                           f"uptime {snap.get('uptime_sec', 0):.0f}s")

        va_col, au_col = st.columns(2, gap="medium")
        with va_col:
            with st.container(border=True):
                st.subheader("Visual analysis")
                st.markdown(_pill(v["label"]), unsafe_allow_html=True)
                st.markdown(_prob_bar(v["prob"], "#dc3545"), unsafe_allow_html=True)
                st.caption(f"Fake {v['prob']:.1%}" if v["prob"] is not None else "Fake --")
                st.caption(f"Real {(1 - v['prob']):.1%}" if v["prob"] is not None else "Real --")
                st.caption(f"Confidence {v['confidence']:.0%} · "
                           f"{'uncertain' if v['uncertain'] else 'decisive'}")
                if v["history"]:
                    st.line_chart({"fake_prob": v["history"]}, height=120)

        with au_col:
            with st.container(border=True):
                st.subheader("Audio analysis")
                st.markdown(_pill(a["label"]), unsafe_allow_html=True)
                if not a.get("available"):
                    st.caption(a.get("error") or "Audio model/input unavailable.")
                else:
                    st.markdown(_prob_bar(a["prob"], "#764ba2"), unsafe_allow_html=True)
                    st.progress(min(max(float(a.get("level", 0.0)), 0.0), 1.0),
                                text=f"Mic level {float(a.get('level', 0.0)) * 100:.0f}%"
                                + (" · silence" if a.get("silence") else ""))
                    st.caption(f"Segments analysed: {a['analyses']}")
                    if a["waveform"]:
                        st.line_chart({"waveform": a["waveform"][-160:]}, height=120)
                    segs = a.get("segments", [])[-5:]
                    if segs:
                        st.caption("Recent segments (newest last)")
                        for s in segs:
                            c = "#dc3545" if s["label"] == "FAKE" else (
                                "#28a745" if s["label"] == "REAL" else "#ffc107")
                            st.markdown(
                                f"<div style='font-size:12px;color:#7a8ba0;'>"
                                f"<span style='color:{c};font-weight:700;'>{s['label']}</span> "
                                f"{s['fake_prob']:.0%} · +{s['t_end'] - s['t_start']:.0f}s window</div>",
                                unsafe_allow_html=True,
                            )

        with st.container(border=True):
            st.subheader("Multimodal fusion")
            fcol = st.columns(3)
            fcol[0].metric("Video fake prob",
                           f"{v['prob']:.1%}" if v["prob"] is not None else "--")
            fcol[1].metric("Audio fake prob",
                           f"{a['prob']:.1%}" if a["prob"] is not None else "--")
            fcol[2].metric("Fused fake prob",
                           f"{f['prob']:.1%}" if f["prob"] is not None else "--",
                           delta=(f"{(f['prob'] - snap['threshold']):+.2f} vs threshold"
                                  if f["prob"] is not None else None))
            st.caption(f"Weighted average · video {snap['fusion_w_video']:.0%} / "
                       f"audio {1 - snap['fusion_w_video']:.0%} · "
                       f"{'audio dropped (unavailable)' if not a.get('available') else 'both modalities live'} · "
                       f"video {v['label']} + audio {a['label']} → {f['label']}")

        with st.container(border=True):
            st.subheader("Legend")
            lcol = st.columns(3)
            lcol[0].caption(":green[REAL] — likely authentic")
            lcol[1].caption(":red[FAKE] — likely manipulated")
            lcol[2].caption(":orange[UNCERTAIN] — near boundary, keep observing")

    _live_dashboard()

    # -- snapshot fallback (works with no camera / remote deploy) -----------
    st.markdown("---")
    with st.container(border=True):
        st.subheader("Snapshot fallback")
        st.caption("Single-frame check using the same video model. "
                   "Useful when no camera is attached or the live feed is blocked.")
        snap = st.camera_input("Take a snapshot to analyse", label_visibility="collapsed")
        if snap is not None:
            try:
                from PIL import Image as _PILImage
                pil_img = _PILImage.open(snap).convert("RGB")
                bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                from src.realtime_webcam import detect_largest_face as _dlf
                from src.realtime_webcam import predict_fake_prob as _pfp
                box = _dlf(bgr, face_cascade)
                if box is None:
                    st.warning("No face detected — face the camera and try again.")
                else:
                    x0, y0, x1, y1 = box
                    face = bgr[y0:y1, x0:x1]
                    t0 = time.time()
                    prob = float(_pfp(face, model, device, transform))
                    label, conf, unc = classify_live(prob, float(threshold), 0.10)
                    st.markdown(_pill(label), unsafe_allow_html=True)
                    st.markdown(_prob_bar(prob, "#dc3545"), unsafe_allow_html=True)
                    st.caption(f"Fake {prob:.1%} · confidence {conf:.0%} · "
                               f"{(time.time() - t0) * 1000:.0f} ms")
                    if st.button(":material/save: Save snapshot result to history"):
                        try:
                            store_prediction("live_snapshot", "live", label, conf, prob, 0.0)
                            st.success("Saved to Prediction History.")
                        except Exception as exc:
                            st.error(f"Could not save: {exc}")
            except Exception as exc:
                st.error(f"Snapshot analysis failed: {exc}")

    # -- completed-analyses history ------------------------------------------
    hist = st.session_state.get("live_history", [])
    if hist:
        st.markdown("---")
        with st.container(border=True):
            st.subheader("This session's completed analyses")
            st.dataframe(hist, width="stretch", hide_index=True)
            st.caption("All saved results are also listed under Prediction History / Dashboard.")
