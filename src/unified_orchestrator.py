"""
src/unified_orchestrator.py
Unified orchestrator that ties all advanced detection modules together.

Coordinates:
- Audio forensics (audio_forensics/)
- Temporal analysis (temporal_analysis/)
- Multimodal fusion (multimodal_fusion/)
- Media forensics (media_forensics/)
- Forensic report generation (src/forensic_report_builder.py)

Provides a single entry point for comprehensive multi-modal deepfake analysis.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the unified orchestrator."""
    run_audio_forensics: bool = True
    run_temporal_analysis: bool = True
    run_multimodal_fusion: bool = True
    run_media_forensics: bool = True
    run_forensic_report: bool = False
    fusion_method: str = "weighted"
    fake_threshold: float = 0.5


@dataclass
class ModuleResult:
    """Result from a single analysis module."""
    module_name: str = ""
    success: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class UnifiedResult:
    """Complete result from the unified orchestrator."""
    analysis_id: str = ""
    file_path: str = ""
    file_type: str = ""
    overall_verdict: str = "UNKNOWN"
    overall_confidence: float = 0.0
    overall_fake_probability: float = 0.0
    module_results: Dict[str, ModuleResult] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "overall_verdict": self.overall_verdict,
            "overall_confidence": self.overall_confidence,
            "overall_fake_probability": self.overall_fake_probability,
            "modules_run": list(self.module_results.keys()),
            "modules_succeeded": [
                k for k, v in self.module_results.items() if v.success
            ],
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "processing_time": self.processing_time,
        }

    def get_summary(self) -> str:
        lines = [
            "=" * 60,
            "UNIFIED ANALYSIS REPORT",
            "=" * 60,
            f"Analysis ID: {self.analysis_id}",
            f"File: {self.file_path}",
            f"Type: {self.file_type}",
            "",
            "OVERALL VERDICT",
            "-" * 40,
            f"  Verdict: {self.overall_verdict}",
            f"  Confidence: {self.overall_confidence:.3f}",
            f"  Fake Probability: {self.overall_fake_probability:.3f}",
            "",
            "MODULE RESULTS",
            "-" * 40,
        ]
        for name, result in self.module_results.items():
            status = "OK" if result.success else "FAILED"
            lines.append(
                f"  [{status}] {name}: {result.processing_time:.2f}s"
                + (f" - {result.error}" if result.error else "")
            )
        if self.evidence:
            lines.extend(["", "EVIDENCE", "-" * 40])
            for e in self.evidence[:10]:
                lines.append(f"  - {e}")
        if self.recommendations:
            lines.extend(["", "RECOMMENDATIONS", "-" * 40])
            for r in self.recommendations[:5]:
                lines.append(f"  - {r}")
        lines.extend(["", f"Total time: {self.processing_time:.2f}s", "=" * 60])
        return "\n".join(lines)


class UnifiedOrchestrator:
    """Unified orchestrator for comprehensive multi-modal deepfake analysis.

    Usage:
        orchestrator = UnifiedOrchestrator()
        result = orchestrator.analyze_video("video.mp4")
        result = orchestrator.analyze_audio("audio.wav")
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._audio_pipeline = None
        self._temporal_analyzer = None
        self._multimodal_analyzer = None
        self._forensic_analyzer = None
        self._report_builder = None

    def _get_audio_pipeline(self):
        if self._audio_pipeline is None:
            try:
                from audio_forensics.inference import AudioInferencePipeline
                from audio_forensics.config import AudioForensicsConfig
                self._audio_pipeline = AudioInferencePipeline(AudioForensicsConfig())
            except Exception as e:
                logger.warning("Could not load audio forensics: %s", e)
        return self._audio_pipeline

    def _get_temporal_analyzer(self):
        if self._temporal_analyzer is None:
            try:
                from temporal_analysis.temporal_analyzer import TemporalAnalyzer
                from temporal_analysis.config import TemporalConfig
                self._temporal_analyzer = TemporalAnalyzer(TemporalConfig())
            except Exception as e:
                logger.warning("Could not load temporal analysis: %s", e)
        return self._temporal_analyzer

    def _get_multimodal_analyzer(self):
        if self._multimodal_analyzer is None:
            try:
                from multimodal_fusion.multimodal_analyzer import MultimodalAnalyzer
                from multimodal_fusion.config import MultimodalConfig
                self._multimodal_analyzer = MultimodalAnalyzer(MultimodalConfig())
            except Exception as e:
                logger.warning("Could not load multimodal fusion: %s", e)
        return self._multimodal_analyzer

    def _get_forensic_analyzer(self):
        if self._forensic_analyzer is None:
            try:
                from media_forensics.forensic_analyzer import ForensicAnalyzer
                self._forensic_analyzer = ForensicAnalyzer()
            except Exception as e:
                logger.warning("Could not load media forensics: %s", e)
        return self._forensic_analyzer

    def _get_report_builder(self):
        if self._report_builder is None:
            try:
                from src.forensic_report_builder import ForensicReportBuilder
                self._report_builder = ForensicReportBuilder()
            except Exception as e:
                logger.warning("Could not load report builder: %s", e)
        return self._report_builder

    def analyze_video(
        self,
        video_path: str,
        visual_result: Optional[Dict[str, Any]] = None,
    ) -> UnifiedResult:
        """Run full multi-modal analysis on a video file.

        Args:
            video_path: Path to the video file.
            visual_result: Optional pre-computed visual detection results.

        Returns:
            UnifiedResult with all module results.
        """
        start_time = time.time()
        analysis_id = f"UNI-{uuid.uuid4().hex[:8].upper()}"

        result = UnifiedResult(
            analysis_id=analysis_id,
            file_path=video_path,
            file_type="video",
        )

        # 1. Media forensics
        if self.config.run_media_forensics:
            result.module_results["media_forensics"] = self._run_media_forensics(
                video_path
            )

        # 2. Audio forensics
        audio_result = None
        if self.config.run_audio_forensics:
            module_result = self._run_audio_forensics(video_path)
            result.module_results["audio_forensics"] = module_result
            if module_result.success:
                audio_result = module_result.data

        # 3. Temporal analysis
        temporal_result = None
        if self.config.run_temporal_analysis:
            module_result = self._run_temporal_analysis(video_path)
            result.module_results["temporal_analysis"] = module_result
            if module_result.success:
                temporal_result = module_result.data

        # 4. Multimodal fusion
        if self.config.run_multimodal_fusion:
            video_score = 0.0
            if visual_result:
                video_score = visual_result.get("fake_probability", 0.0)
            audio_score = 0.0
            if audio_result:
                audio_score = audio_result.get("overall_probability", 0.0)
            temporal_score = 0.0
            if temporal_result:
                temporal_score = temporal_result.get("anomaly_score", 0.0)

            module_result = self._run_multimodal_fusion(
                video_score=video_score,
                audio_score=audio_score,
                temporal_score=temporal_score,
            )
            result.module_results["multimodal_fusion"] = module_result
            if module_result.success:
                result.overall_verdict = module_result.data.get("verdict", "UNKNOWN")
                result.overall_confidence = module_result.data.get("confidence", 0.0)
                result.overall_fake_probability = module_result.data.get("fused_score", 0.0)

        # 5. Forensic report
        if self.config.run_forensic_report:
            result.module_results["forensic_report"] = self._run_forensic_report(
                video_path, visual_result, audio_result
            )

        # Aggregate evidence and recommendations
        for mod_result in result.module_results.values():
            if mod_result.success:
                result.evidence.extend(mod_result.data.get("evidence", []))
                result.recommendations.extend(mod_result.data.get("recommendations", []))

        # Deduplicate
        result.evidence = list(dict.fromkeys(result.evidence))[:20]
        result.recommendations = list(dict.fromkeys(result.recommendations))[:10]

        # Default verdict from probability if fusion didn't produce one
        if result.overall_verdict == "UNKNOWN" and result.overall_fake_probability > 0:
            if result.overall_fake_probability >= self.config.fake_threshold:
                result.overall_verdict = "FAKE"
            else:
                result.overall_verdict = "REAL"

        result.processing_time = time.time() - start_time
        logger.info("Unified analysis complete: %s (%.2fs)", analysis_id, result.processing_time)
        return result

    def analyze_audio(
        self,
        audio_path: str,
    ) -> UnifiedResult:
        """Run full analysis on an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            UnifiedResult with all module results.
        """
        start_time = time.time()
        analysis_id = f"UNI-{uuid.uuid4().hex[:8].upper()}"

        result = UnifiedResult(
            analysis_id=analysis_id,
            file_path=audio_path,
            file_type="audio",
        )

        # 1. Media forensics
        if self.config.run_media_forensics:
            result.module_results["media_forensics"] = self._run_media_forensics(
                audio_path
            )

        # 2. Audio forensics
        audio_result = None
        if self.config.run_audio_forensics:
            module_result = self._run_audio_forensics(audio_path)
            result.module_results["audio_forensics"] = module_result
            if module_result.success:
                audio_result = module_result.data
                result.overall_fake_probability = audio_result.get("overall_probability", 0.0)

        # 3. Determine verdict
        if result.overall_fake_probability >= self.config.fake_threshold:
            result.overall_verdict = "FAKE"
        elif result.overall_fake_probability > 0:
            result.overall_verdict = "REAL"

        # 4. Forensic report
        if self.config.run_forensic_report:
            result.module_results["forensic_report"] = self._run_forensic_report(
                audio_path, None, audio_result
            )

        # Aggregate
        for mod_result in result.module_results.values():
            if mod_result.success:
                result.evidence.extend(mod_result.data.get("evidence", []))
                result.recommendations.extend(mod_result.data.get("recommendations", []))
        result.evidence = list(dict.fromkeys(result.evidence))[:20]
        result.recommendations = list(dict.fromkeys(result.recommendations))[:10]

        result.processing_time = time.time() - start_time
        return result

    def _run_media_forensics(self, file_path: str) -> ModuleResult:
        """Run media forensics module."""
        t0 = time.time()
        try:
            analyzer = self._get_forensic_analyzer()
            if analyzer is None:
                return ModuleResult(
                    module_name="media_forensics",
                    success=False,
                    error="Module not available",
                    processing_time=time.time() - t0,
                )
            report = analyzer.analyze(file_path)
            return ModuleResult(
                module_name="media_forensics",
                success=True,
                data=report if isinstance(report, dict) else {"report": str(report)},
                processing_time=time.time() - t0,
            )
        except Exception as e:
            return ModuleResult(
                module_name="media_forensics",
                success=False,
                error=str(e),
                processing_time=time.time() - t0,
            )

    def _run_audio_forensics(self, file_path: str) -> ModuleResult:
        """Run audio forensics module."""
        t0 = time.time()
        try:
            pipeline = self._get_audio_pipeline()
            if pipeline is None:
                return ModuleResult(
                    module_name="audio_forensics",
                    success=False,
                    error="Module not available",
                    processing_time=time.time() - t0,
                )
            result = pipeline.analyze(file_path)
            return ModuleResult(
                module_name="audio_forensics",
                success=True,
                data={
                    "overall_prediction": result.overall_prediction.value,
                    "overall_probability": result.overall_probability,
                    "overall_confidence": result.overall_confidence,
                    "total_segments": result.total_segments,
                    "suspicious_segments": result.suspicious_segments,
                    "evidence": result.evidence,
                    "recommendations": result.recommendations,
                    "timeline": result.timeline,
                },
                processing_time=time.time() - t0,
            )
        except Exception as e:
            return ModuleResult(
                module_name="audio_forensics",
                success=False,
                error=str(e),
                processing_time=time.time() - t0,
            )

    def _run_temporal_analysis(self, video_path: str) -> ModuleResult:
        """Run temporal analysis module."""
        t0 = time.time()
        try:
            analyzer = self._get_temporal_analyzer()
            if analyzer is None:
                return ModuleResult(
                    module_name="temporal_analysis",
                    success=False,
                    error="Module not available",
                    processing_time=time.time() - t0,
                )
            result = analyzer.analyze(video_path)
            return ModuleResult(
                module_name="temporal_analysis",
                success=True,
                data={
                    "consistency_score": result.consistency_score,
                    "anomaly_score": result.anomaly_score,
                    "is_suspicious": result.is_suspicious,
                    "n_frames": result.n_frames,
                    "evidence": result.evidence,
                    "recommendations": result.recommendations,
                },
                processing_time=time.time() - t0,
            )
        except Exception as e:
            return ModuleResult(
                module_name="temporal_analysis",
                success=False,
                error=str(e),
                processing_time=time.time() - t0,
            )

    def _run_multimodal_fusion(
        self,
        video_score: float = 0.0,
        audio_score: float = 0.0,
        temporal_score: float = 0.0,
    ) -> ModuleResult:
        """Run multimodal fusion module."""
        t0 = time.time()
        try:
            analyzer = self._get_multimodal_analyzer()
            if analyzer is None:
                return ModuleResult(
                    module_name="multimodal_fusion",
                    success=False,
                    error="Module not available",
                    processing_time=time.time() - t0,
                )
            result = analyzer.analyze(
                video_score=video_score,
                audio_score=audio_score,
                temporal_score=temporal_score,
                method=self.config.fusion_method,
            )
            return ModuleResult(
                module_name="multimodal_fusion",
                success=True,
                data={
                    "verdict": result.verdict.value,
                    "confidence": result.confidence,
                    "scores": result.scores,
                    "fused_score": result.scores.get("fused", 0.0),
                    "evidence": [e.source + ": " + e.status for e in result.evidence],
                    "recommendations": result.recommendations,
                },
                processing_time=time.time() - t0,
            )
        except Exception as e:
            return ModuleResult(
                module_name="multimodal_fusion",
                success=False,
                error=str(e),
                processing_time=time.time() - t0,
            )

    def _run_forensic_report(
        self,
        file_path: str,
        visual_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
    ) -> ModuleResult:
        """Run forensic report builder."""
        t0 = time.time()
        try:
            builder = self._get_report_builder()
            if builder is None:
                return ModuleResult(
                    module_name="forensic_report",
                    success=False,
                    error="Module not available",
                    processing_time=time.time() - t0,
                )
            report = builder.build(
                file_path=file_path,
                visual_result=visual_result,
                audio_result=audio_result,
            )
            return ModuleResult(
                module_name="forensic_report",
                success=True,
                data={"report_id": getattr(report, "case_id", "unknown")},
                processing_time=time.time() - t0,
            )
        except Exception as e:
            return ModuleResult(
                module_name="forensic_report",
                success=False,
                error=str(e),
                processing_time=time.time() - t0,
            )
