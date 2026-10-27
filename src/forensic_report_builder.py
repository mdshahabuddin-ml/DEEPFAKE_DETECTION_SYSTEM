"""
src/forensic_report_builder.py
Build comprehensive forensic reports from multi-modal analysis results.

Orchestrates the collection of evidence from all detection modalities
and generates a complete forensic report with PDF output.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.config import RESULTS_DIR, MODEL_NAME, API_VERSION
from src.logger import get_logger
from src.forensic_report import ForensicReportGenerator, ForensicReportPDF
from src.forensic_report_enhanced import (
    CaseInformation,
    VideoAnalysis,
    AudioAnalysis,
    SpeakerSimilarity,
    LipSyncAnalysis,
    ProvenanceMetadata,
    TechnicalDetails,
    EvidenceTimeline,
    Limitations,
    EnhancedForensicReport,
)

logger = get_logger(__name__)


class ForensicReportBuilder:
    """
    Build comprehensive forensic reports from analysis results.
    
    Collects evidence from:
    - Visual deepfake detection
    - Audio deepfake detection
    - Speaker verification
    - Lip-sync analysis
    - Provenance verification
    - Forensic metadata analysis
    
    Generates a complete PDF report with all sections.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR / "forensic_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_generator = ForensicReportGenerator(output_dir=str(self.output_dir))
    
    def build(
        self,
        file_path: str,
        visual_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        speaker_result: Optional[Dict[str, Any]] = None,
        sync_result: Optional[Dict[str, Any]] = None,
        provenance_result: Optional[Dict[str, Any]] = None,
    ) -> EnhancedForensicReport:
        """
        Build complete forensic report from analysis results.
        
        Args:
            file_path: Path to the analyzed media file
            visual_result: Visual deepfake detection results
            audio_result: Audio deepfake detection results
            speaker_result: Speaker verification results
            sync_result: Lip-sync analysis results
            provenance_result: Provenance verification results
            
        Returns:
            EnhancedForensicReport with all sections populated
        """
        logger.info("Building forensic report for: %s", file_path)
        
        # Generate case ID
        case_id = self._generate_case_id(file_path)
        
        # Compute file hash
        file_hash = self._compute_file_hash(file_path)
        
        # Build case information
        case_info = CaseInformation(
            case_id=case_id,
            file_hash_sha256=file_hash["sha256"],
            file_hash_md5=file_hash["md5"],
            file_name=Path(file_path).name,
            file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            media_type=self._detect_media_type(file_path),
            analysis_timestamp=datetime.now().isoformat(),
        )
        
        # Build video analysis
        video_analysis = None
        if visual_result:
            video_analysis = VideoAnalysis(
                faces_detected=visual_result.get("faces_detected", 0),
                face_consistency_score=visual_result.get("face_consistency", 0),
                temporal_consistency_score=visual_result.get("temporal_consistency", 0),
                visual_artifact_score=visual_result.get("texture_score", 0),
                manipulated_frames=visual_result.get("manipulated_frames", []),
                frame_analysis=visual_result.get("frame_analysis", []),
            )
        
        # Build audio analysis
        audio_analysis = None
        if audio_result:
            audio_analysis = AudioAnalysis(
                audio_deepfake_score=audio_result.get("fake_probability", 0),
                quality_score=audio_result.get("quality_score", 0),
                snr_estimate=audio_result.get("snr_estimate", 0),
                spectral_centroid=audio_result.get("spectral_centroid", 0),
                clipping_ratio=audio_result.get("clipping_ratio", 0),
                audio_duration=audio_result.get("duration", 0),
            )
        
        # Build speaker similarity
        speaker_similarity = None
        if speaker_result:
            speaker_similarity = SpeakerSimilarity(
                claimed_speaker=speaker_result.get("claimed_speaker", "Unknown"),
                embedding_similarity=speaker_result.get("similarity", 0),
                is_verified=speaker_result.get("is_verified", False),
                verification_confidence=speaker_result.get("confidence", 0),
                enrollment_samples=speaker_result.get("enrollment_samples", 0),
            )
        
        # Build lip-sync analysis
        lip_sync_analysis = None
        if sync_result:
            lip_sync_analysis = LipSyncAnalysis(
                sync_score=sync_result.get("sync_score", 0),
                anomalies_count=sync_result.get("anomalies_count", 0),
                anomalies=sync_result.get("anomalies", []),
                consistency=sync_result.get("consistency", 0),
                timeline=sync_result.get("timeline", []),
            )
        
        # Build provenance metadata
        provenance_metadata = ProvenanceMetadata(
            c2pa_status=provenance_result.get("status", "unknown") if provenance_result else "unknown",
            trust_level=provenance_result.get("trust_level", "unknown") if provenance_result else "unknown",
            has_c2pa=provenance_result.get("has_c2pa", False) if provenance_result else False,
            metadata_consistent=provenance_result.get("metadata_consistent", True) if provenance_result else True,
            editing_indicators=provenance_result.get("editing_indicators", {}) if provenance_result else {},
        )
        
        # Build technical details
        technical_details = TechnicalDetails(
            visual_model=MODEL_NAME,
            audio_model="efficientnet",
            sync_model="mediapipe",
            confidence_interval=self._compute_confidence_interval(visual_result),
            uncertainty=self._compute_uncertainty(visual_result, audio_result),
            predictive_entropy=self._compute_entropy(visual_result, audio_result),
        )
        
        # Build evidence timeline
        evidence_timeline = self._build_timeline(
            visual_result, audio_result, speaker_result, sync_result
        )
        
        # Build limitations
        limitations = self._build_limitations(
            visual_result, audio_result, speaker_result, sync_result
        )
        
        # Compute overall scores
        overall_verdict, overall_confidence, trust_score = self._compute_overall(
            visual_result, audio_result, speaker_result, sync_result
        )
        
        return EnhancedForensicReport(
            case_info=case_info,
            video_analysis=video_analysis,
            audio_analysis=audio_analysis,
            speaker_similarity=speaker_similarity,
            lip_sync_analysis=lip_sync_analysis,
            provenance_metadata=provenance_metadata,
            technical_details=technical_details,
            evidence_timeline=evidence_timeline,
            limitations=limitations,
            overall_verdict=overall_verdict,
            overall_confidence=overall_confidence,
            trust_score=trust_score,
        )
    
    def generate_pdf(
        self,
        report: EnhancedForensicReport,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate PDF from forensic report.
        
        Args:
            report: EnhancedForensicReport to convert to PDF
            output_path: Optional output path for PDF
            
        Returns:
            Path to generated PDF file
        """
        pdf = ForensicReportPDF(evidence_id=report.case_info.case_id)
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.add_page()
        
        # Add verdict section manually
        verdict = report.overall_verdict
        trust_score = report.trust_score
        confidence = report.overall_confidence
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, f"Verdict: {verdict}".encode("latin-1", "replace").decode("latin-1"),
                 new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "B", 28)
        if trust_score >= 75:
            pdf.set_text_color(34, 160, 90)
        elif trust_score >= 40:
            pdf.set_text_color(210, 150, 30)
        else:
            pdf.set_text_color(200, 60, 60)
        
        pdf.cell(0, 14, f"Trust Score: {trust_score:.0f}/100".encode("latin-1", "replace").decode("latin-1"),
                 new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(70, 70, 70)
        pdf.cell(0, 6, f"Confidence: {confidence * 100:.1f}%".encode("latin-1", "replace").decode("latin-1"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        
        # Add case information
        _section_header(pdf, "Case Information")
        _kv(pdf, "Case ID", report.case_info.case_id)
        _kv(pdf, "SHA-256", report.case_info.file_hash_sha256)
        _kv(pdf, "MD5", report.case_info.file_hash_md5)
        _kv(pdf, "File Name", report.case_info.file_name)
        _kv(pdf, "File Size", f"{report.case_info.file_size / 1024:.2f} KB")
        _kv(pdf, "Media Type", report.case_info.media_type)
        _kv(pdf, "Analysis Time", report.case_info.analysis_timestamp)
        pdf.ln(3)
        
        # Add video analysis
        if report.video_analysis:
            video = report.video_analysis
            _section_header(pdf, "Video Analysis")
            _kv(pdf, "Faces Detected", video.faces_detected)
            _kv(pdf, "Face Consistency", f"{video.face_consistency_score:.3f}")
            _kv(pdf, "Temporal Consistency", f"{video.temporal_consistency_score:.3f}")
            _kv(pdf, "Visual Artifacts", f"{video.visual_artifact_score:.3f}")
            _kv(pdf, "Manipulated Frames", f"{len(video.manipulated_frames)}")
            pdf.ln(3)
        
        # Add audio analysis
        if report.audio_analysis:
            audio = report.audio_analysis
            _section_header(pdf, "Audio Analysis")
            _kv(pdf, "Audio Deepfake Score", f"{audio.audio_deepfake_score:.3f}")
            _kv(pdf, "Quality Score", f"{audio.quality_score:.3f}")
            _kv(pdf, "SNR Estimate", f"{audio.snr_estimate:.1f} dB")
            _kv(pdf, "Spectral Centroid", f"{audio.spectral_centroid:.1f} Hz")
            _kv(pdf, "Clipping Ratio", f"{audio.clipping_ratio:.4f}")
            _kv(pdf, "Duration", f"{audio.audio_duration:.2f} s")
            pdf.ln(3)
        
        # Add speaker similarity
        if report.speaker_similarity:
            speaker = report.speaker_similarity
            _section_header(pdf, "Speaker Similarity")
            _kv(pdf, "Claimed Speaker", speaker.claimed_speaker)
            _kv(pdf, "Embedding Similarity", f"{speaker.embedding_similarity:.3f}")
            _kv(pdf, "Verification Status", "Verified" if speaker.is_verified else "Failed")
            _kv(pdf, "Confidence", f"{speaker.verification_confidence:.3f}")
            _kv(pdf, "Enrollment Samples", speaker.enrollment_samples)
            pdf.ln(3)
        
        # Add lip-sync analysis
        if report.lip_sync_analysis:
            lip_sync = report.lip_sync_analysis
            _section_header(pdf, "Lip-Sync Analysis")
            _kv(pdf, "Sync Score", f"{lip_sync.sync_score:.3f}")
            _kv(pdf, "Anomalies Detected", lip_sync.anomalies_count)
            _kv(pdf, "Consistency", f"{lip_sync.consistency:.3f}")
            pdf.ln(3)
        
        # Add provenance metadata
        _section_header(pdf, "Provenance & Metadata")
        _kv(pdf, "C2PA Status", report.provenance_metadata.c2pa_status)
        _kv(pdf, "Trust Level", report.provenance_metadata.trust_level)
        _kv(pdf, "C2PA Present", "Yes" if report.provenance_metadata.has_c2pa else "No")
        _kv(pdf, "Metadata Consistent", "Yes" if report.provenance_metadata.metadata_consistent else "No")
        pdf.ln(3)
        
        # Add model versions
        tech = report.technical_details
        _section_header(pdf, "Model Versions")
        _kv(pdf, "Visual Model", tech.visual_model)
        _kv(pdf, "Audio Model", tech.audio_model)
        _kv(pdf, "Sync Model", tech.sync_model)
        pdf.ln(3)
        
        # Add evidence timeline
        timeline = report.evidence_timeline
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
        
        # Add limitations
        limitations = report.limitations
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
        
        # Add chain of custody
        evidence_id = report.case_info.case_id
        _section_header(pdf, "Chain of Custody")
        _kv(pdf, "Evidence ID", evidence_id)
        _kv(pdf, "Generated", datetime.now().isoformat())
        _kv(pdf, "Analyst", "Automated System")
        pdf.ln(2)
        
        # Add disclaimer
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 4, "DISCLAIMER: This report is generated by an AI system and is intended as decision-support only. Predictions are probabilistic and can err. This should not be used as the sole basis for legal or forensic conclusions. Human expert review is recommended.")
        
        # Save PDF
        if output_path is None:
            output_path = self.output_dir / f"forensic_{report.case_info.case_id}.pdf"
        
        pdf.output(str(output_path))
        logger.info("Forensic PDF report generated: %s", output_path)
        return str(output_path)
    
    def _generate_case_id(self, file_path: str) -> str:
        """Generate unique case ID."""
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        fh = hashlib.md5(file_path.encode()).hexdigest()[:8]
        return f"EV-{ts}-{fh}"
    
    def _compute_file_hash(self, file_path: str) -> Dict[str, str]:
        """Compute file hashes."""
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
        
        return {
            "md5": md5_hash.hexdigest(),
            "sha256": sha256_hash.hexdigest(),
        }
    
    def _detect_media_type(self, file_path: str) -> str:
        """Detect media type from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}:
            return "image"
        elif ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return "video"
        elif ext in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
            return "audio"
        return "unknown"
    
    def _compute_confidence_interval(
        self,
        visual_result: Optional[Dict[str, Any]],
    ) -> Optional[List[float]]:
        """Compute confidence interval from results."""
        if not visual_result:
            return None
        
        fake_prob = visual_result.get("fake_probability", 0.5)
        uncertainty = visual_result.get("uncertainty", 0.1)
        
        lower = max(0.0, fake_prob - 1.96 * uncertainty)
        upper = min(1.0, fake_prob + 1.96 * uncertainty)
        
        return [lower, upper]
    
    def _compute_uncertainty(
        self,
        visual_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
    ) -> float:
        """Compute overall uncertainty."""
        uncertainties = []
        
        if visual_result and "uncertainty" in visual_result:
            uncertainties.append(visual_result["uncertainty"])
        
        if audio_result and "uncertainty" in audio_result:
            uncertainties.append(audio_result["uncertainty"])
        
        if not uncertainties:
            return 0.1  # Default uncertainty
        
        return sum(uncertainties) / len(uncertainties)
    
    def _compute_entropy(
        self,
        visual_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
    ) -> float:
        """Compute predictive entropy."""
        import numpy as np
        
        probs = []
        
        if visual_result and "fake_probability" in visual_result:
            probs.append(visual_result["fake_probability"])
        
        if audio_result and "fake_probability" in audio_result:
            probs.append(audio_result["fake_probability"])
        
        if not probs:
            return 0.0
        
        avg_prob = sum(probs) / len(probs)
        
        # Binary entropy
        if avg_prob in [0.0, 1.0]:
            return 0.0
        
        entropy = -(avg_prob * np.log2(avg_prob) + (1 - avg_prob) * np.log2(1 - avg_prob))
        return float(entropy)
    
    def _build_timeline(
        self,
        visual_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
        speaker_result: Optional[Dict[str, Any]],
        sync_result: Optional[Dict[str, Any]],
    ) -> EvidenceTimeline:
        """Build evidence timeline from results."""
        events = []
        now = datetime.now()
        
        # Add analysis start event
        events.append({
            "timestamp": now.strftime("%H:%M:%S"),
            "description": "Analysis started",
        })
        
        # Add visual analysis events
        if visual_result:
            events.append({
                "timestamp": (now).strftime("%H:%M:%S"),
                "description": f"Visual analysis: {visual_result.get('faces_detected', 0)} faces detected",
            })
        
        # Add audio analysis events
        if audio_result:
            events.append({
                "timestamp": (now).strftime("%H:%M:%S"),
                "description": f"Audio analysis: score {audio_result.get('fake_probability', 0):.3f}",
            })
        
        # Add speaker verification events
        if speaker_result:
            status = "verified" if speaker_result.get("is_verified", False) else "failed"
            events.append({
                "timestamp": (now).strftime("%H:%M:%S"),
                "description": f"Speaker verification: {status}",
            })
        
        # Add lip-sync events
        if sync_result:
            events.append({
                "timestamp": (now).strftime("%H:%M:%S"),
                "description": f"Lip-sync analysis: score {sync_result.get('sync_score', 0):.3f}",
            })
        
        # Add completion event
        events.append({
            "timestamp": now.strftime("%H:%M:%S"),
            "description": "Analysis complete",
        })
        
        temporal_summary = f"Analysis completed with {len(events)} events recorded."
        
        return EvidenceTimeline(
            events=events,
            temporal_summary=temporal_summary,
        )
    
    def _build_limitations(
        self,
        visual_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
        speaker_result: Optional[Dict[str, Any]],
        sync_result: Optional[Dict[str, Any]],
    ) -> Limitations:
        """Build limitations section."""
        known_limitations = [
            "Single face detection only (multiple faces not supported)",
            "Frame sampling may miss manipulation in unsampled frames",
            "Audio quality affects detection accuracy",
            "Speaker verification requires enrollment samples",
            "Provenance detection depends on C2PA availability",
        ]
        
        confidence_factors = []
        
        if visual_result:
            if visual_result.get("faces_detected", 0) > 0:
                confidence_factors.append("Face detected in video (high confidence)")
            else:
                confidence_factors.append("No face detected (low confidence)")
        
        if audio_result:
            if audio_result.get("snr_estimate", 0) > 20:
                confidence_factors.append("Good audio quality (high confidence)")
            else:
                confidence_factors.append("Poor audio quality (low confidence)")
        
        if speaker_result:
            if speaker_result.get("is_verified", False):
                confidence_factors.append("Speaker verified (high confidence)")
            else:
                confidence_factors.append("Speaker verification failed (low confidence)")
        
        if sync_result:
            if sync_result.get("sync_score", 0) > 0.7:
                confidence_factors.append("Good lip-sync (high confidence)")
            else:
                confidence_factors.append("Poor lip-sync (low confidence)")
        
        recommendations = [
            "Manual expert review recommended for high-stakes decisions",
            "Consider additional analysis with different models",
            "Verify results against known deepfake datasets",
        ]
        
        return Limitations(
            known_limitations=known_limitations,
            confidence_factors=confidence_factors,
            recommendations=recommendations,
        )
    
    def _compute_overall(
        self,
        visual_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
        speaker_result: Optional[Dict[str, Any]],
        sync_result: Optional[Dict[str, Any]],
    ) -> tuple[str, float, float]:
        """Compute overall verdict, confidence, and trust score."""
        import numpy as np
        
        scores = []
        weights = []
        
        # Visual score (weight: 0.35)
        if visual_result and "fake_probability" in visual_result:
            scores.append(visual_result["fake_probability"])
            weights.append(0.35)
        
        # Audio score (weight: 0.25)
        if audio_result and "fake_probability" in audio_result:
            scores.append(audio_result["fake_probability"])
            weights.append(0.25)
        
        # Speaker score (weight: 0.20)
        if speaker_result and "similarity" in speaker_result:
            # Lower similarity = more likely fake
            speaker_fake = 1.0 - speaker_result["similarity"]
            scores.append(speaker_fake)
            weights.append(0.20)
        
        # Sync score (weight: 0.20)
        if sync_result and "sync_score" in sync_result:
            # Lower sync = more likely fake
            sync_fake = 1.0 - sync_result["sync_score"]
            scores.append(sync_fake)
            weights.append(0.20)
        
        if not scores:
            return "UNKNOWN", 0.0, 50.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # Compute confidence (based on agreement)
        if len(scores) > 1:
            agreement = 1.0 - np.std(scores)
        else:
            agreement = 0.5

        confidence = float(np.clip(agreement, 0.0, 1.0))

        # Compute trust score (0-100, higher = more trustworthy)
        trust_score = float(np.clip((1.0 - weighted_score) * 100, 0.0, 100.0))
        
        # Determine verdict
        if weighted_score >= 0.7:
            verdict = "FAKE"
        elif weighted_score >= 0.4:
            verdict = "UNCERTAIN"
        else:
            verdict = "REAL"
        
        return verdict, confidence, trust_score


def _section_header(pdf, title):
    """Section header helper."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 8, title.encode("latin-1", "replace").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def _kv(pdf, key, value, kw=55):
    """Key-value pair helper."""
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(kw, 5, f"{key}:".encode("latin-1", "replace").decode("latin-1"))
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 5, str(value).encode("latin-1", "replace").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
