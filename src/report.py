"""
src/report.py
AI-generated PDF authenticity report from a TrustReport.

Renders a professional, single/two-page PDF with the verdict, Trust Score,
per-signal breakdown (with bars), manipulated-frame range, file hash (content
passport), and a recommendation — suitable as KYC/forensics evidence.

Uses fpdf2 (pure-Python). Core PDF fonts are latin-1, so all text is sanitized
to ASCII before rendering.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from fpdf import FPDF

from src.config import API_VERSION, MODEL_NAME, RESULTS_DIR
from src.logger import get_logger
from src.trust_score import TrustReport

logger = get_logger(__name__)

# Signal display names + short explanations.
_SIGNAL_LABELS = {
    "face_consistency": "Face Consistency",
    "temporal_consistency": "Temporal Consistency",
    "lip_sync": "Lip-Sync / Voice Agreement",
    "texture": "Texture / Spectral Analysis",
}

# RGB colors for verdict bands.
_GREEN = (34, 160, 90)
_AMBER = (210, 150, 30)
_RED = (200, 60, 60)
_GREY = (110, 110, 110)
_LIGHT = (235, 235, 235)


def _s(text: object) -> str:
    """Sanitize any value to latin-1-safe ASCII for the core PDF fonts."""
    s = str(text)
    replacements = {
        "—": "-", "–": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "·": "-", "→": "->",
        "≥": ">=", "≤": "<=", "…": "...",
    }
    for uni, asc in replacements.items():
        s = s.replace(uni, asc)
    return s.encode("latin-1", "replace").decode("latin-1")


def _score_color(score: float):
    if score >= 75:
        return _GREEN
    if score >= 40:
        return _AMBER
    return _RED


class _ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, _s("Deepfake Authenticity Report"), new_x="LMARGIN",
                  new_y="NEXT", align="L")
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_GREY)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 8, _s(f"Generated {stamp}  |  Model {MODEL_NAME} v{API_VERSION}  "
                           f"|  Page {self.page_no()}  |  Decision-support only, not sole ground truth."),
                  align="C")


def _bar(pdf: FPDF, x: float, y: float, width: float, value: float, color) -> None:
    """Draw a horizontal progress bar for a [0,1] value."""
    height = 4
    pdf.set_fill_color(*_LIGHT)
    pdf.rect(x, y, width, height, style="F")
    pdf.set_fill_color(*color)
    pdf.rect(x, y, width * max(0.0, min(1.0, value)), height, style="F")


def generate_pdf_report(report: TrustReport,
                        output_path: Optional[Union[str, Path]] = None,
                        gradcam_image: Optional[Union[str, Path]] = None) -> Path:
    """
    Render `report` to a PDF and return the output path.

    `gradcam_image` (optional) is a path to a PNG/JPG (e.g. a Grad-CAM overlay)
    embedded in the report.
    """
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = RESULTS_DIR / f"{ts}_report.pdf"
    output_path = Path(output_path)

    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # --- Verdict + Trust Score band ---
    color = _score_color(report.trust_score)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, _s(f"Verdict: {report.verdict}"), new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(*color)
    pdf.cell(0, 16, _s(f"Trust Score: {report.trust_score:.0f} / 100"),
             new_x="LMARGIN", new_y="NEXT")
    _bar(pdf, pdf.l_margin, pdf.get_y() + 1, pdf.w - pdf.l_margin - pdf.r_margin,
         report.trust_score / 100.0, color)
    pdf.ln(8)

    # --- Metadata ---
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    meta = [
        ("File", report.file_name),
        ("Media type", report.media_type),
        ("Primary fake probability", f"{report.fake_probability * 100:.1f}%"),
        ("Faces detected", report.faces_detected),
        ("Frames analyzed", report.frames_analyzed),
        ("Analysis time", f"{report.inference_time:.2f} s"),
    ]
    if report.manipulated_frames:
        fr = report.manipulated_frames
        rng = f"{min(fr)}-{max(fr)} ({len(fr)} of {report.frames_analyzed})"
        meta.append(("Manipulated frames", rng))
    for k, v in meta:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 6, _s(f"{k}:"))
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _s(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Signal breakdown ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, _s("Signal Breakdown"), new_x="LMARGIN", new_y="NEXT")

    bar_x = pdf.l_margin + 70
    bar_w = pdf.w - pdf.r_margin - bar_x - 18
    for key, label in _SIGNAL_LABELS.items():
        val = report.signals.get(key)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        y = pdf.get_y()
        pdf.cell(70, 6, _s(label))
        if val is None:
            pdf.set_text_color(*_GREY)
            pdf.cell(0, 6, _s("n/a"), new_x="LMARGIN", new_y="NEXT")
        else:
            _bar(pdf, bar_x, y + 1, bar_w, val, _score_color(val * 100))
            pdf.set_xy(bar_x + bar_w + 2, y)
            pdf.cell(16, 6, _s(f"{val * 100:.0f}%"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Optional Grad-CAM image ---
    if gradcam_image is not None and Path(gradcam_image).exists():
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, _s("Grad-CAM (manipulated regions)"), new_x="LMARGIN", new_y="NEXT")
        try:
            pdf.image(str(gradcam_image), w=60)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not embed Grad-CAM image: %s", exc)
        pdf.ln(3)

    # --- Recommendation ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, _s("Recommendation"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, _s(report.recommendation))
    pdf.ln(2)

    # --- Content passport (hash) ---
    if report.file_hash:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_GREY)
        pdf.multi_cell(0, 5, _s(f"Content hash (SHA-256): {report.file_hash}"))

    pdf.output(str(output_path))
    logger.info("Generated PDF report: %s", output_path.name)
    return output_path
