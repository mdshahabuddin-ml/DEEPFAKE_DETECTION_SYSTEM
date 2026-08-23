"""
app/gradio_app.py
Gradio web interface — AI-Powered Deepfake Detection for KYC Verification.

A production, deployment-ready Gradio app that reuses the shared inference
engine (src/inference.py), Grad-CAM explainer (src/gradcam.py), SQLite history
(src/database.py), and timestamped result writer (src/results_writer.py).

Features
    • Image screening  : verdict, confidence %, detected face, Grad-CAM overlay
    • Video screening  : per-frame inference + majority voting + statistics
    • Batch screening  : many images at once → results table + downloadable CSV
    • KYC framing      : VERIFIED (likely real) vs FLAGGED (likely deepfake)
    • Timestamped audit artifacts saved to results/, history in SQLite
    • GPU acceleration when CUDA is available

Run locally:
    python app/gradio_app.py

Deployment (Render / Railway / HF Spaces): the module exposes a top-level
`demo` object and honours the PORT environment variable.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

# Make the project root importable when Gradio runs this as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import (
    CHECKPOINT_PATH, DEFAULT_THRESHOLD, DEVICE, MODEL_NAME, RESULTS_DIR,
    VIDEO_MAX_FRAMES, confidence_from_prob, label_from_prob,
)
from src.database import PredictionRecord, get_db
from src.inference import ModelNotFoundError, annotate_faces, get_detector
from src.logger import get_logger
from src.report import generate_pdf_report
from src.results_writer import save_batch_csv, save_result
from src.trust_score import analyze_image as trust_analyze_image_core
from src.trust_score import analyze_video as trust_analyze_video_core

logger = get_logger("gradio")

MODEL_MISSING_MSG = (
    f"### ⚠️ Model checkpoint not found\n"
    f"Place a trained checkpoint at `{CHECKPOINT_PATH}` "
    f"(or set `DEEPFAKE_CHECKPOINT`) and restart the app."
)


# ---------------------------------------------------------------------------
# Lazy singletons (so the UI builds even without a checkpoint)
# ---------------------------------------------------------------------------
_detector = None
_explainer = None


def _get_detector():
    """Return the shared detector, or None if the checkpoint is missing."""
    global _detector
    if _detector is None:
        try:
            _detector = get_detector()
        except ModelNotFoundError as exc:
            logger.error("Model unavailable: %s", exc)
            return None
    return _detector


def _get_explainer():
    """Return a Grad-CAM explainer, or None if unavailable."""
    global _explainer
    if _explainer is None:
        detector = _get_detector()
        if detector is None:
            return None
        try:
            from src.gradcam import GradCAMExplainer
            _explainer = GradCAMExplainer(detector)
        except Exception as exc:  # pragma: no cover
            logger.warning("Grad-CAM unavailable: %s", exc)
            return None
    return _explainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _kyc_verdict(prob: float, threshold: float) -> str:
    """KYC-oriented markdown verdict for a fake-probability."""
    label = label_from_prob(prob, threshold)
    conf = confidence_from_prob(prob, threshold)
    if label == "FAKE":
        return (f"## 🚨 FLAGGED — Likely DEEPFAKE\n"
                f"**Confidence:** {conf * 100:.1f}%  ·  fake probability "
                f"{prob * 100:.1f}%  ·  threshold {threshold:.2f}\n\n"
                f"> Recommend manual KYC review before approval.")
    return (f"## ✅ VERIFIED — Likely REAL\n"
            f"**Confidence:** {conf * 100:.1f}%  ·  fake probability "
            f"{prob * 100:.1f}%  ·  threshold {threshold:.2f}")


def _label_scores(prob: float) -> dict:
    """Dict for gr.Label: probability mass on REAL vs FAKE."""
    return {"REAL": float(1.0 - prob), "FAKE": float(prob)}


def _log_history(result: dict, media_type: str) -> None:
    if result.get("prediction") in (None, "NO_FACE", "ERROR"):
        return
    try:
        get_db().add(PredictionRecord(
            file_name=result.get("file_name", "gradio_upload"),
            media_type=media_type,
            prediction=result["prediction"],
            confidence=float(result.get("confidence") or 0.0),
            fake_probability=float(result.get("fake_probability") or 0.0),
            inference_time=float(result.get("inference_time") or 0.0),
            source="gradio",
        ))
    except Exception as exc:  # pragma: no cover
        logger.warning("History logging failed: %s", exc)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def analyze_image(image_rgb: Optional[np.ndarray], threshold: float):
    """Image tab handler. Returns (face, overlay, label_scores, verdict_md)."""
    if image_rgb is None:
        return None, None, None, "Please upload an image."

    detector = _get_detector()
    if detector is None:
        return None, None, None, MODEL_MISSING_MSG

    try:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        face_rgb: Optional[np.ndarray] = None
        overlay_rgb: Optional[np.ndarray] = None

        # Prefer Grad-CAM (gives face crop + overlay + prob in one pass).
        explainer = _get_explainer()
        if explainer is not None:
            out = explainer.explain_image_array(bgr)
            if out is None:
                return None, None, None, "❌ No face detected — upload a clear, front-facing photo."
            prob = out["fake_probability"]
            face_rgb, overlay_rgb = out["original"], out["overlay"]
        else:
            result = detector.predict_image_array(bgr)
            if result["prediction"] == "NO_FACE":
                return None, None, None, "❌ No face detected — upload a clear, front-facing photo."
            prob = result["fake_probability"]

        verdict = _kyc_verdict(prob, threshold)
        result = {
            "file_name": "gradio_image_upload",
            "prediction": label_from_prob(prob, threshold),
            "fake_probability": round(prob, 6),
            "confidence": round(confidence_from_prob(prob, threshold), 6),
            "threshold": threshold,
            "inference_time": 0.0,
            "media_type": "image",
        }
        _log_history(result, "image")
        save_result(result, overlay_rgb=overlay_rgb, prefix="image")
        return face_rgb, overlay_rgb, _label_scores(prob), verdict

    except Exception as exc:  # pragma: no cover - surface errors to the UI
        logger.error("Image analysis failed: %s\n%s", exc, traceback.format_exc())
        return None, None, None, f"⚠️ Error during analysis: {exc}"


def analyze_video(video_path: Optional[str], threshold: float, max_frames: int):
    """Video tab handler. Returns (label_scores, verdict_md, stats_dict)."""
    if not video_path:
        return None, "Please upload a video.", {}

    detector = _get_detector()
    if detector is None:
        return None, MODEL_MISSING_MSG, {}

    try:
        detector.threshold = threshold  # honour the UI threshold
        result = detector.predict_video(video_path, max_frames=int(max_frames))
        if result["prediction"] == "NO_FACE":
            return None, "❌ No face detected in any sampled frame.", result

        result["file_name"] = Path(video_path).name
        _log_history(result, "video")
        save_result(result, prefix="video")

        prob = result["fake_probability"]
        verdict = _kyc_verdict(prob, threshold)
        stats = {
            "frames_sampled": result["frames_sampled"],
            "frames_with_face": result["frames_with_face"],
            "voted_fake": result["fake_frames"],
            "voted_real": result["real_frames"],
            "no_face_frames": result["no_face_frames"],
            "mean_fake_probability": prob,
            "inference_time_s": result["inference_time"],
        }
        return _label_scores(prob), verdict, stats

    except Exception as exc:  # pragma: no cover
        logger.error("Video analysis failed: %s\n%s", exc, traceback.format_exc())
        return None, f"⚠️ Error during analysis: {exc}", {}


def analyze_batch(files: Optional[List], threshold: float):
    """Batch tab handler. Returns (table_rows, csv_path)."""
    if not files:
        return [], None

    detector = _get_detector()
    if detector is None:
        return [["—", MODEL_MISSING_MSG.replace("\n", " "), "", ""]], None

    # Gradio may pass file objects or path strings depending on version.
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
    results = detector.predict_images(paths)

    rows, csv_rows = [], []
    for res in results:
        prob = res.get("fake_probability")
        if prob is None:  # NO_FACE / ERROR
            verdict, conf = res.get("prediction", "N/A"), ""
        else:
            verdict = label_from_prob(prob, threshold)
            conf = f"{confidence_from_prob(prob, threshold) * 100:.1f}%"
            res["prediction"] = verdict
            res["confidence"] = confidence_from_prob(prob, threshold)
            _log_history(res, "image")
        rows.append([
            res.get("file_name", "?"),
            verdict,
            conf,
            f"{prob * 100:.1f}%" if prob is not None else "—",
        ])
        csv_rows.append({
            "file_name": res.get("file_name", "?"),
            "prediction": verdict,
            "confidence": conf,
            "fake_probability": prob if prob is not None else "",
            "inference_time": res.get("inference_time", ""),
        })

    csv_path = save_batch_csv(csv_rows, prefix="batch")
    return rows, str(csv_path)


# ---------------------------------------------------------------------------
# Face localization handler
# ---------------------------------------------------------------------------
def localize_faces_image(image_rgb, threshold):
    """Detect & score every face. Returns (annotated_image, table, summary)."""
    if image_rgb is None:
        return None, [], "Please upload an image."
    detector = _get_detector()
    if detector is None:
        return None, [], MODEL_MISSING_MSG
    try:
        detector.threshold = threshold
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = detector.localize_faces(bgr)
        if not results:
            return image_rgb, [], "❌ No faces detected."
        annotated = annotate_faces(bgr, results, threshold)
        rows = [
            [f"#{i}", r["prediction"],
             f"{r['confidence'] * 100:.1f}%", f"{r['fake_probability'] * 100:.1f}%",
             str(r["box"])]
            for i, r in enumerate(results, start=1)
        ]
        n_fake = sum(1 for r in results if r["prediction"] == "FAKE")
        summary = (f"**{len(results)} face(s) detected** — "
                   f"🔴 {n_fake} flagged FAKE, 🟢 {len(results) - n_fake} REAL.")
        return annotated, rows, summary
    except Exception as exc:  # pragma: no cover
        logger.error("Localization failed: %s\n%s", exc, traceback.format_exc())
        return None, [], f"⚠️ Error: {exc}"


# ---------------------------------------------------------------------------
# Batch video handler
# ---------------------------------------------------------------------------
def batch_videos(files, threshold, max_frames):
    """Screen multiple videos. Returns (table_rows, csv_path)."""
    if not files:
        return [], None
    detector = _get_detector()
    if detector is None:
        return [["—", MODEL_MISSING_MSG.replace("\n", " "), "", ""]], None

    detector.threshold = threshold
    paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
    results = detector.predict_videos(paths, max_frames=int(max_frames))

    rows, csv_rows = [], []
    for res in results:
        prob = res.get("fake_probability")
        verdict = res.get("prediction", "N/A")
        conf = (f"{res.get('confidence', 0) * 100:.1f}%"
                if res.get("confidence") is not None else "")
        if verdict not in ("NO_FACE", "ERROR"):
            _log_history(res, "video")
        rows.append([
            res.get("file_name", "?"), verdict, conf,
            f"{prob * 100:.1f}%" if prob is not None else "—",
            res.get("frames_with_face", "—"),
        ])
        csv_rows.append({
            "file_name": res.get("file_name", "?"),
            "prediction": verdict,
            "confidence": conf,
            "mean_fake_probability": prob if prob is not None else "",
            "frames_with_face": res.get("frames_with_face", ""),
            "inference_time": res.get("inference_time", ""),
        })
    csv_path = save_batch_csv(csv_rows, prefix="batch_videos")
    return rows, str(csv_path)


# ---------------------------------------------------------------------------
# Annotated video handler
# ---------------------------------------------------------------------------
def annotate_video_handler(video_path, threshold):
    """Produce an annotated MP4. Returns (video_path, summary_md)."""
    if not video_path:
        return None, "Please upload a video."
    detector = _get_detector()
    if detector is None:
        return None, MODEL_MISSING_MSG
    try:
        detector.threshold = threshold
        res = detector.annotate_video(video_path)
        if res["prediction"] == "NO_FACE":
            return res["output_path"], "❌ No face detected in the video."
        prob = res["fake_probability"]
        label = label_from_prob(prob, threshold)
        conf = confidence_from_prob(prob, threshold)
        emoji = "🔴" if label == "FAKE" else "🟢"
        md = (f"### {emoji} {label} — {conf * 100:.1f}% confidence\n"
              f"Annotated **{res['frames']}** frames "
              f"({res['frames_with_face']} with a face) in "
              f"{res['inference_time']:.1f}s. Download the boxed video below.")
        return res["output_path"], md
    except Exception as exc:  # pragma: no cover
        logger.error("Annotate video failed: %s\n%s", exc, traceback.format_exc())
        return None, f"⚠️ Error: {exc}"


# ---------------------------------------------------------------------------
# Trust Score handlers
# ---------------------------------------------------------------------------
def _trust_markdown(report) -> str:
    """Format a TrustReport as a rich markdown summary."""
    emoji = "🟢" if report.trust_score >= 75 else ("🟠" if report.trust_score >= 40 else "🔴")
    lines = [
        f"## {emoji} Trust Score: {report.trust_score:.0f} / 100",
        f"**Verdict:** {report.verdict}  \n"
        f"**Primary fake probability:** {report.fake_probability * 100:.1f}%  \n"
        f"**Faces detected:** {report.faces_detected}  ·  "
        f"**Frames analyzed:** {report.frames_analyzed}",
    ]
    if report.manipulated_frames:
        fr = report.manipulated_frames
        lines.append(f"**Manipulated frames:** {min(fr)}–{max(fr)} "
                     f"({len(fr)} of {report.frames_analyzed})")
    lines.append(f"\n> **Recommendation:** {report.recommendation}")
    return "\n\n".join(lines)


def _signal_scores(report) -> dict:
    """gr.Label-friendly dict of available sub-signals."""
    names = {
        "face_consistency": "Face Consistency",
        "temporal_consistency": "Temporal Consistency",
        "lip_sync": "Lip-Sync / Voice",
        "texture": "Texture / Spectral",
    }
    return {names[k]: float(v) for k, v in report.signals.items()
            if k in names and v is not None}


def trust_image(image_rgb, threshold):
    """Trust Score for an image. Returns (markdown, signal_scores, pdf_path)."""
    if image_rgb is None:
        return "Please upload an image.", None, None
    detector = _get_detector()
    if detector is None:
        return MODEL_MISSING_MSG, None, None
    try:
        detector.threshold = threshold
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        report = trust_analyze_image_core(detector, bgr, file_name="gradio_image")
        if report.verdict == "No Face":
            return "❌ No face detected — upload a clear, front-facing photo.", None, None

        # Add a Grad-CAM overlay to the PDF when available.
        overlay_path = None
        explainer = _get_explainer()
        if explainer is not None:
            out = explainer.explain_image_array(bgr)
            if out is not None:
                overlay_path = RESULTS_DIR / "_tmp_gradcam.png"
                cv2.imwrite(str(overlay_path),
                            cv2.cvtColor(out["overlay"], cv2.COLOR_RGB2BGR))

        pdf = generate_pdf_report(report, gradcam_image=overlay_path)
        return _trust_markdown(report), _signal_scores(report), str(pdf)
    except Exception as exc:  # pragma: no cover
        logger.error("Trust image analysis failed: %s\n%s", exc, traceback.format_exc())
        return f"⚠️ Error: {exc}", None, None


def trust_video(video_path, threshold, max_frames):
    """Trust Score for a video. Returns (markdown, signal_scores, pdf_path)."""
    if not video_path:
        return "Please upload a video.", None, None
    detector = _get_detector()
    if detector is None:
        return MODEL_MISSING_MSG, None, None
    try:
        detector.threshold = threshold
        report = trust_analyze_video_core(detector, video_path, max_frames=int(max_frames))
        if report.verdict == "No Face":
            return "❌ No face detected in any sampled frame.", None, None
        pdf = generate_pdf_report(report)
        return _trust_markdown(report), _signal_scores(report), str(pdf)
    except Exception as exc:  # pragma: no cover
        logger.error("Trust video analysis failed: %s\n%s", exc, traceback.format_exc())
        return f"⚠️ Error: {exc}", None, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Deepfake Detection — KYC Verification") as demo:
        gr.Markdown(
            f"""
            # 🕵️ AI-Powered Deepfake Detection — KYC Verification
            Screen face **images** and **videos** for deepfake manipulation with
            confidence scores and **Grad-CAM** explanations.
            &nbsp;·&nbsp; Model: `{MODEL_NAME}` &nbsp;·&nbsp; Device: `{DEVICE}`
            """
        )
        if not CHECKPOINT_PATH.exists():
            gr.Markdown(MODEL_MISSING_MSG)

        threshold = gr.Slider(0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01,
                              label="Decision threshold (fake if ≥)")

        with gr.Tab("📷 Image"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="Upload face image", type="numpy")
                    img_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    verdict_md = gr.Markdown()
                    scores = gr.Label(label="Real vs Fake", num_top_classes=2)
            with gr.Row():
                face_out = gr.Image(label="Detected face")
                overlay_out = gr.Image(label="Grad-CAM (manipulated regions)")
            img_btn.click(
                analyze_image, inputs=[img_in, threshold],
                outputs=[face_out, overlay_out, scores, verdict_md],
            )

        with gr.Tab("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="Upload video")
                    frames_slider = gr.Slider(4, 32, value=VIDEO_MAX_FRAMES, step=1,
                                              label="Frames to sample")
                    vid_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    vid_verdict = gr.Markdown()
                    vid_scores = gr.Label(label="Real vs Fake", num_top_classes=2)
                    vid_stats = gr.JSON(label="Frame statistics")
            vid_btn.click(
                analyze_video, inputs=[vid_in, threshold, frames_slider],
                outputs=[vid_scores, vid_verdict, vid_stats],
            )

        with gr.Tab("🗂️ Batch"):
            gr.Markdown("Upload multiple images to screen them in one pass.")
            batch_in = gr.File(label="Upload images", file_count="multiple",
                               file_types=["image"])
            batch_btn = gr.Button("Run batch", variant="primary")
            batch_table = gr.Dataframe(
                headers=["File", "Verdict", "Confidence", "Fake prob"],
                label="Results", wrap=True,
            )
            batch_csv = gr.File(label="Download results CSV")
            batch_btn.click(
                analyze_batch, inputs=[batch_in, threshold],
                outputs=[batch_table, batch_csv],
            )

        with gr.Tab("🧭 Face Localization"):
            gr.Markdown("Detect and score **every face** in an image (multi-face).")
            with gr.Row():
                with gr.Column():
                    loc_in = gr.Image(label="Upload image", type="numpy")
                    loc_btn = gr.Button("Localize faces", variant="primary")
                with gr.Column():
                    loc_out = gr.Image(label="Localized faces")
                    loc_summary = gr.Markdown()
            loc_table = gr.Dataframe(
                headers=["Face", "Verdict", "Confidence", "Fake prob", "Box"],
                label="Per-face results", wrap=True,
            )
            loc_btn.click(
                localize_faces_image, inputs=[loc_in, threshold],
                outputs=[loc_out, loc_table, loc_summary],
            )

        with gr.Tab("🎬 Batch Videos"):
            gr.Markdown("Screen **multiple videos** at once (enterprise workflow).")
            bv_in = gr.File(label="Upload videos", file_count="multiple",
                            file_types=["video"])
            bv_frames = gr.Slider(4, 32, value=VIDEO_MAX_FRAMES, step=1,
                                  label="Frames to sample per video")
            bv_btn = gr.Button("Run batch", variant="primary")
            bv_table = gr.Dataframe(
                headers=["File", "Verdict", "Confidence", "Fake prob", "Frames w/ face"],
                label="Results", wrap=True,
            )
            bv_csv = gr.File(label="Download results CSV")
            bv_btn.click(
                batch_videos, inputs=[bv_in, threshold, bv_frames],
                outputs=[bv_table, bv_csv],
            )

        with gr.Tab("🎥 Annotated Video"):
            gr.Markdown("Generate a **downloadable video** with detection boxes "
                        "and REAL/FAKE labels drawn on every frame.")
            with gr.Row():
                with gr.Column():
                    av_in = gr.Video(label="Upload video")
                    av_btn = gr.Button("Annotate", variant="primary")
                with gr.Column():
                    av_out = gr.Video(label="Annotated output")
                    av_md = gr.Markdown()
            av_btn.click(annotate_video_handler, inputs=[av_in, threshold],
                         outputs=[av_out, av_md])

        with gr.Tab("🛡️ Trust Score"):
            gr.Markdown(
                "A multi-signal **0–100 authenticity assessment** (face, temporal, "
                "lip-sync, texture) with an exportable **PDF report**."
            )
            with gr.Tabs():
                with gr.Tab("Image"):
                    with gr.Row():
                        with gr.Column():
                            ts_img_in = gr.Image(label="Upload face image", type="numpy")
                            ts_img_btn = gr.Button("Assess authenticity", variant="primary")
                        with gr.Column():
                            ts_img_md = gr.Markdown()
                            ts_img_scores = gr.Label(label="Signal breakdown")
                            ts_img_pdf = gr.File(label="Download PDF report")
                    ts_img_btn.click(
                        trust_image, inputs=[ts_img_in, threshold],
                        outputs=[ts_img_md, ts_img_scores, ts_img_pdf],
                    )
                with gr.Tab("Video"):
                    with gr.Row():
                        with gr.Column():
                            ts_vid_in = gr.Video(label="Upload video")
                            ts_vid_frames = gr.Slider(4, 32, value=VIDEO_MAX_FRAMES, step=1,
                                                      label="Frames to sample")
                            ts_vid_btn = gr.Button("Assess authenticity", variant="primary")
                        with gr.Column():
                            ts_vid_md = gr.Markdown()
                            ts_vid_scores = gr.Label(label="Signal breakdown")
                            ts_vid_pdf = gr.File(label="Download PDF report")
                    ts_vid_btn.click(
                        trust_video, inputs=[ts_vid_in, threshold, ts_vid_frames],
                        outputs=[ts_vid_md, ts_vid_scores, ts_vid_pdf],
                    )

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                f"""
                **AI-Powered Explainable Deepfake Detection System**

                - **Model:** {MODEL_NAME} (XceptionNet backbone), Celeb-DF v2
                - **Pipeline:** Face detection → crop → resize → normalize →
                  Xception → sigmoid → fake probability → KYC verdict
                - **Explainability:** Grad-CAM over the last convolutional layer
                - **Audit trail:** timestamped JSON/PNG in `results/`, history in SQLite
                - **Deployment:** Docker, Hugging Face Spaces, Render, Railway

                Predictions are probabilistic decision-support, not sole ground truth.
                """
            )
    return demo


demo = build_demo()


def _auth_fn(username: str, password: str) -> bool:
    """Credential check for Gradio's native login page."""
    from src.auth import authenticate
    return authenticate(username, password)


def launch(port: Optional[int] = None, enable_login: bool = True):
    """Launch the app, with a native login page unless disabled."""
    from src.auth import get_auth_db
    port = port or int(os.getenv("PORT", "7860"))
    auth = None
    if enable_login and os.getenv("DEEPFAKE_DISABLE_LOGIN", "").lower() not in ("1", "true", "yes"):
        get_auth_db()  # ensure the user store + default admin exist
        auth = _auth_fn
    # server_name 0.0.0.0 so it works inside Docker / cloud platforms.
    # theme is passed to launch() (Gradio 6+ moved it off the Blocks constructor).
    demo.launch(server_name="0.0.0.0", server_port=port, theme=gr.themes.Soft(),
                auth=auth,
                auth_message="Sign in — default admin/admin123 (change in production).")


if __name__ == "__main__":
    launch()
