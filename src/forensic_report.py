"""
src/forensic_report.py
Enhanced forensic PDF report with detailed evidence packages.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from fpdf import FPDF

from src.config import API_VERSION, MODEL_NAME, RESULTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def _sanitize(text: object) -> str:
    s = str(text)
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"', "\u00b7": "-", "\u2192": "->",
    }
    for uni, asc in replacements.items():
        s = s.replace(uni, asc)
    return s.encode("latin-1", "replace").decode("latin-1")


_GREEN = (34, 160, 90)
_AMBER = (210, 150, 30)
_RED = (200, 60, 60)
_GREY = (110, 110, 110)
_LIGHT = (235, 235, 235)
_DARK = (30, 30, 30)


def _score_color(score: float):
    if score >= 75:
        return _GREEN
    if score >= 40:
        return _AMBER
    return _RED


class ForensicReportPDF(FPDF):
    def __init__(self, evidence_id: Optional[str] = None):
        super().__init__()
        self.evidence_id = evidence_id

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*_DARK)
        self.cell(0, 10, _sanitize("Deepfake Forensic Analysis Report"),
                  new_x="LMARGIN", new_y="NEXT", align="L")
        if self.evidence_id:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_GREY)
            self.cell(0, 5, _sanitize(f"Evidence ID: {self.evidence_id}"),
                      new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_GREY)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 8, _sanitize(
            f"Generated {stamp} | Model {MODEL_NAME} v{API_VERSION} | "
            f"Page {self.page_no()} | CONFIDENTIAL - Forensic Evidence"
        ), align="C")


def _bar(pdf, x, y, width, value, color):
    height = 4
    pdf.set_fill_color(*_LIGHT)
    pdf.rect(x, y, width, height, style="F")
    pdf.set_fill_color(*color)
    pdf.rect(x, y, width * max(0.0, min(1.0, value)), height, style="F")


def _section_header(pdf, title):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*_DARK)
    pdf.cell(0, 8, _sanitize(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def _kv(pdf, key, value, kw=55):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(kw, 5, _sanitize(f"{key}:"))
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 5, _sanitize(str(value)), new_x="LMARGIN", new_y="NEXT")


class ForensicReportGenerator:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR / "forensic_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        analysis_result: Dict[str, Any],
        file_path: str,
        output_path: Optional[Union[str, Path]] = None,
        additional_images: Optional[List[Dict[str, str]]] = None,
    ) -> Path:
        evidence_id = analysis_result.get(
            "evidence_id",
            self._generate_evidence_id(file_path)
        )

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"forensic_{evidence_id}_{timestamp}.pdf"

        output_path = Path(output_path)
        pdf = ForensicReportPDF(evidence_id=evidence_id)
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()

        self._add_verdict(pdf, analysis_result)
        self._add_metadata(pdf, analysis_result, file_path)
        self._add_file_hash(pdf, analysis_result)
        self._add_prediction(pdf, analysis_result)
        self._add_uncertainty(pdf, analysis_result)
        self._add_signals(pdf, analysis_result)

        if "temporal_analysis" in analysis_result:
            self._add_temporal(pdf, analysis_result["temporal_analysis"])

        if "provenance" in analysis_result:
            self._add_provenance(pdf, analysis_result["provenance"])

        if additional_images:
            self._add_images(pdf, additional_images)

        self._add_recommendation(pdf, analysis_result)
        self._add_chain_of_custody(pdf, evidence_id)
        self._add_disclaimer(pdf)

        pdf.output(str(output_path))
        logger.info("Forensic report generated: %s", output_path)
        return output_path

    def _generate_evidence_id(self, file_path: str) -> str:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        fh = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"EV-{ts}-{fh}"

    def _add_verdict(self, pdf, result):
        verdict = result.get("prediction", "UNKNOWN")
        trust_score = result.get("trust_score", 0.0)
        confidence = result.get("confidence", 0.0)
        color = _score_color(trust_score)

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*_DARK)
        pdf.cell(0, 8, _sanitize(f"Verdict: {verdict}"),
                 new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(*color)
        pdf.cell(0, 14, _sanitize(f"Trust Score: {trust_score:.0f}/100"),
                 new_x="LMARGIN", new_y="NEXT")

        bar_x = pdf.l_margin
        bar_w = pdf.w - pdf.l_margin - pdf.r_margin
        _bar(pdf, bar_x, pdf.get_y() + 1, bar_w, trust_score / 100.0, color)
        pdf.ln(6)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(0, 6, _sanitize(f"Confidence: {confidence * 100:.1f}%"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    def _add_metadata(self, pdf, result, file_path):
        _section_header(pdf, "File Metadata")
        _kv(pdf, "File name", Path(file_path).name)
        _kv(pdf, "File path", file_path)
        _kv(pdf, "Media type", result.get("media_type", "unknown"))
        _kv(pdf, "Analysis time", result.get("analysis_time", "unknown"))
        pdf.ln(2)

    def _add_file_hash(self, pdf, result):
        if "file_hash" not in result:
            return
        _section_header(pdf, "File Integrity (Chain of Custody)")
        fh = result["file_hash"]
        _kv(pdf, "MD5", fh.get("md5", "N/A"))
        _kv(pdf, "SHA-256", fh.get("sha256", "N/A"))
        pdf.ln(2)

    def _add_prediction(self, pdf, result):
        _section_header(pdf, "Prediction Details")
        _kv(pdf, "Model", MODEL_NAME)
        _kv(pdf, "Fake probability",
            f"{result.get('fake_probability', 0) * 100:.2f}%")
        _kv(pdf, "Threshold", f"{result.get('threshold', 0.5):.2f}")
        if "fake_frames" in result:
            _kv(pdf, "Fake frames",
                f"{result['fake_frames']}/{result.get('frames_sampled', 0)}")
        pdf.ln(2)

    def _add_uncertainty(self, pdf, result):
        if "uncertainty" not in result:
            return
        _section_header(pdf, "Uncertainty Analysis")
        _kv(pdf, "Prediction uncertainty", f"{result['uncertainty']:.4f}")
        if "confidence_interval" in result:
            ci = result["confidence_interval"]
            _kv(pdf, "95% CI", f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        if "predictive_entropy" in result:
            _kv(pdf, "Predictive entropy", f"{result['predictive_entropy']:.4f}")
        pdf.ln(2)

    def _add_signals(self, pdf, result):
        signals = result.get("signals", {})
        if not signals:
            return
        _section_header(pdf, "Signal Breakdown")

        labels = {
            "face_consistency": "Face Consistency",
            "temporal_consistency": "Temporal Consistency",
            "lip_sync": "Lip-Sync Agreement",
            "texture": "Texture Analysis",
        }

        bar_x = pdf.l_margin + 70
        bar_w = pdf.w - pdf.r_margin - bar_x - 18

        for key, label in labels.items():
            val = signals.get(key)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(50, 50, 50)
            y = pdf.get_y()
            pdf.cell(70, 5, _sanitize(label))
            if val is None:
                pdf.set_text_color(*_GREY)
                pdf.cell(0, 5, _sanitize("n/a"), new_x="LMARGIN", new_y="NEXT")
            else:
                _bar(pdf, bar_x, y + 1, bar_w, val, _score_color(val * 100))
                pdf.set_xy(bar_x + bar_w + 2, y)
                pdf.cell(16, 5, _sanitize(f"{val * 100:.0f}%"),
                         new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    def _add_temporal(self, pdf, temporal):
        _section_header(pdf, "Temporal Analysis")
        _kv(pdf, "Anomaly score",
            f"{temporal.get('overall_anomaly_score', 0):.3f}")
        _kv(pdf, "Consistency",
            f"{temporal.get('temporal_consistency', 0):.3f}")
        _kv(pdf, "Motion smoothness",
            f"{temporal.get('motion_smoothness', 0):.3f}")
        pdf.ln(2)

    def _add_provenance(self, pdf, provenance):
        _section_header(pdf, "Provenance Verification")
        _kv(pdf, "Status", provenance.get("status", "unknown"))
        _kv(pdf, "Trust level", provenance.get("trust_level", "unknown"))
        _kv(pdf, "C2PA present", "Yes" if provenance.get("has_c2pa") else "No")
        pdf.ln(2)

    def _add_images(self, pdf, images):
        _section_header(pdf, "Visual Evidence")
        for img_info in images:
            img_path = img_info.get("path")
            caption = img_info.get("caption", "")
            if img_path and Path(img_path).exists():
                try:
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.cell(0, 5, _sanitize(caption),
                             new_x="LMARGIN", new_y="NEXT")
                    pdf.image(str(img_path), w=80)
                    pdf.ln(3)
                except Exception as e:
                    logger.warning("Could not embed image: %s", e)

    def _add_recommendation(self, pdf, result):
        _section_header(pdf, "Recommendation")
        rec = result.get("recommendation", "No recommendation available.")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, _sanitize(rec))
        pdf.ln(2)

    def _add_chain_of_custody(self, pdf, evidence_id):
        _section_header(pdf, "Chain of Custody")
        _kv(pdf, "Evidence ID", evidence_id)
        _kv(pdf, "Generated", datetime.now().isoformat())
        _kv(pdf, "Analyst", "Automated System")
        pdf.ln(2)

    def _add_disclaimer(self, pdf):
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*_GREY)
        pdf.multi_cell(0, 4, _sanitize(
            "DISCLAIMER: This report is generated by an AI system and is intended "
            "as decision-support only. Predictions are probabilistic and can err. "
            "This should not be used as the sole basis for legal or forensic conclusions. "
            "Human expert review is recommended."
        ))

    def _add_video_analysis(self, pdf, video):
        """Add video analysis section."""
        _section_header(pdf, "Video Analysis")
        _kv(pdf, "Faces Detected", video.faces_detected)
        _kv(pdf, "Face Consistency", f"{video.face_consistency_score:.3f}")
        _kv(pdf, "Temporal Consistency", f"{video.temporal_consistency_score:.3f}")
        _kv(pdf, "Visual Artifacts", f"{video.visual_artifact_score:.3f}")
        _kv(pdf, "Manipulated Frames", f"{len(video.manipulated_frames)}")
        
        if video.frame_analysis:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(20, 5, "Frame")
            pdf.cell(30, 5, "Fake Prob")
            pdf.cell(30, 5, "Consistency")
            pdf.ln()
            
            pdf.set_font("Helvetica", "", 8)
            for frame in video.frame_analysis[:10]:
                pdf.cell(20, 5, str(frame.get("frame_idx", "")))
                pdf.cell(30, 5, f"{frame.get('fake_prob', 0):.3f}")
                pdf.cell(30, 5, f"{frame.get('consistency', 0):.3f}")
                pdf.ln()
        pdf.ln(3)

    def _add_audio_analysis(self, pdf, audio):
        """Add audio analysis section."""
        _section_header(pdf, "Audio Analysis")
        _kv(pdf, "Audio Deepfake Score", f"{audio.audio_deepfake_score:.3f}")
        _kv(pdf, "Quality Score", f"{audio.quality_score:.3f}")
        _kv(pdf, "SNR Estimate", f"{audio.snr_estimate:.1f} dB")
        _kv(pdf, "Spectral Centroid", f"{audio.spectral_centroid:.1f} Hz")
        _kv(pdf, "Clipping Ratio", f"{audio.clipping_ratio:.4f}")
        _kv(pdf, "Duration", f"{audio.audio_duration:.2f} s")
        pdf.ln(3)

    def _add_speaker_similarity(self, pdf, speaker):
        """Add speaker similarity section."""
        _section_header(pdf, "Speaker Similarity")
        _kv(pdf, "Claimed Speaker", speaker.claimed_speaker)
        _kv(pdf, "Embedding Similarity", f"{speaker.embedding_similarity:.3f}")
        _kv(pdf, "Verification Status", "Verified" if speaker.is_verified else "Failed")
        _kv(pdf, "Confidence", f"{speaker.verification_confidence:.3f}")
        _kv(pdf, "Enrollment Samples", speaker.enrollment_samples)
        pdf.ln(3)

    def _add_lip_sync_analysis(self, pdf, lip_sync):
        """Add lip-sync analysis section."""
        _section_header(pdf, "Lip-Sync Analysis")
        _kv(pdf, "Sync Score", f"{lip_sync.sync_score:.3f}")
        _kv(pdf, "Anomalies Detected", lip_sync.anomalies_count)
        _kv(pdf, "Consistency", f"{lip_sync.consistency:.3f}")
        
        if lip_sync.anomalies:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(0, 5, "Anomalies:")
            pdf.ln()
            pdf.set_font("Helvetica", "", 8)
            for i, anomaly in enumerate(lip_sync.anomalies[:5], 1):
                pdf.cell(0, 5, f"  {i}. Frame {anomaly.get('frame_idx', '?')}: {anomaly.get('type', 'unknown')}")
                pdf.ln()
        pdf.ln(3)

    def _add_model_versions(self, pdf, tech):
        """Add model versions section."""
        _section_header(pdf, "Model Versions")
        _kv(pdf, "Visual Model", tech.visual_model)
        _kv(pdf, "Audio Model", tech.audio_model)
        _kv(pdf, "Sync Model", tech.sync_model)
        pdf.ln(3)

    def _add_evidence_timeline(self, pdf, timeline):
        """Add evidence timeline section."""
        _section_header(pdf, "Evidence Timeline")
        
        if timeline.events:
            for event in timeline.events:
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(30, 5, event.get("timestamp", ""))
                pdf.cell(0, 5, event.get("description", ""))
                pdf.ln()
        
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 5, timeline.temporal_summary)
        pdf.ln(3)

    def _add_limitations(self, pdf, limitations):
        """Add limitations section."""
        _section_header(pdf, "Limitations")
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Known Limitations:")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for limitation in limitations.known_limitations:
            pdf.cell(0, 5, f"  - {limitation}")
            pdf.ln()
        
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Confidence Factors:")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for factor in limitations.confidence_factors:
            pdf.cell(0, 5, f"  - {factor}")
            pdf.ln()
        
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, "Recommendations:")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for rec in limitations.recommendations:
            pdf.cell(0, 5, f"  - {rec}")
            pdf.ln()
        pdf.ln(3)
