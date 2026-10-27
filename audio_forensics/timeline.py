"""Audio forensic timeline visualization.

This module creates visual timeline representations of audio analysis
results, showing the temporal distribution of suspicious segments.

The timeline is a standalone component that can be integrated into
any UI without modifying existing systems.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import AudioForensicsConfig
from .schemas import AudioAnalysisResult, AudioStatus, SegmentResult

logger = logging.getLogger(__name__)


@dataclass
class TimelineSegment:
    """A segment in the forensic timeline."""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    status: str = "NORMAL"
    probability: float = 0.0
    confidence: float = 0.0
    color: str = "#4CAF50"  # Green for normal
    label: str = ""


@dataclass
class AudioForensicTimeline:
    """Timeline visualization data for audio analysis."""
    total_duration: float = 0.0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    colors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_duration": self.total_duration,
            "segments": self.segments,
            "summary": self.summary,
            "colors": self.colors,
        }


class AudioForensicTimelineGenerator:
    """Generate forensic timeline from analysis results.

    This class creates visual timeline representations that show
    the temporal distribution of analysis results.

    Usage:
        config = AudioForensicsConfig()
        generator = AudioForensicTimelineGenerator(config)
        timeline = generator.create_timeline(analysis_result)
    """

    # Status colors
    STATUS_COLORS = {
        "NORMAL": "#4CAF50",           # Green
        "LOW_SUSPICION": "#FFC107",    # Amber
        "SUSPICIOUS": "#FF9800",       # Orange
        "POTENTIAL_MANIPULATION": "#F44336",  # Red
        "REQUIRES_REVIEW": "#9C27B0",  # Purple
        "UNKNOWN": "#9E9E9E",          # Gray
        "SILENCE": "#E0E0E0",          # Light gray
        "NOISE": "#BDBDBD",            # Medium gray
    }

    # Status labels
    STATUS_LABELS = {
        "NORMAL": "Normal",
        "LOW_SUSPICION": "Low Suspicion",
        "SUSPICIOUS": "Suspicious",
        "POTENTIAL_MANIPULATION": "Potential Manipulation",
        "REQUIRES_REVIEW": "Requires Review",
        "UNKNOWN": "Unknown",
        "SILENCE": "Silence",
        "NOISE": "Noise",
    }

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        """Initialize timeline generator.

        Args:
            config: Audio forensics configuration.
        """
        self.config = config or AudioForensicsConfig()

    def create_timeline(
        self, analysis_result: AudioAnalysisResult
    ) -> AudioForensicTimeline:
        """Create timeline from analysis result.

        Args:
            analysis_result: Complete analysis result.

        Returns:
            AudioForensicTimeline with visualization data.
        """
        timeline = AudioForensicTimeline(
            total_duration=analysis_result.properties.duration,
            colors=self.STATUS_COLORS.copy(),
        )

        if not analysis_result.segments:
            # No segments - create single timeline entry
            timeline.segments.append({
                "start_time": 0.0,
                "end_time": analysis_result.properties.duration,
                "status": analysis_result.overall_prediction.value,
                "probability": analysis_result.overall_probability,
                "confidence": analysis_result.overall_confidence,
                "color": self.STATUS_COLORS.get(
                    analysis_result.overall_prediction.value, "#9E9E9E"
                ),
                "label": self.STATUS_LABELS.get(
                    analysis_result.overall_prediction.value, "Unknown"
                ),
            })
        else:
            # Create segments from analysis
            for segment in analysis_result.segments:
                status = segment.prediction.value
                timeline.segments.append({
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                    "duration": segment.duration,
                    "status": status,
                    "probability": segment.probability,
                    "confidence": segment.confidence,
                    "color": self.STATUS_COLORS.get(status, "#9E9E9E"),
                    "label": self.STATUS_LABELS.get(status, "Unknown"),
                })

        # Generate summary
        timeline.summary = self._generate_summary(analysis_result)

        return timeline

    def _generate_summary(
        self, result: AudioAnalysisResult
    ) -> Dict[str, Any]:
        """Generate timeline summary statistics."""
        segments = result.segments

        if not segments:
            return {
                "total_segments": 0,
                "suspicious_segments": 0,
                "normal_segments": 0,
                "mean_probability": 0.0,
                "max_probability": 0.0,
                "anomaly_score": 0.0,
            }

        # Count by status
        status_counts = {}
        probabilities = []

        for segment in segments:
            status = segment.prediction.value
            status_counts[status] = status_counts.get(status, 0) + 1
            probabilities.append(segment.probability)

        # Calculate statistics
        probs = np.array(probabilities)
        mean_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))

        # Anomaly score: combination of suspicious ratio and max probability
        n_suspicious = sum(
            1 for s in segments
            if s.prediction in [
                AudioStatus.SUSPICIOUS,
                AudioStatus.POTENTIAL_MANIPULATION,
            ]
        )
        suspicious_ratio = n_suspicious / len(segments) if segments else 0
        anomaly_score = (suspicious_ratio * 0.6 + max_prob * 0.4)

        return {
            "total_segments": len(segments),
            "suspicious_segments": n_suspicious,
            "normal_segments": status_counts.get("NORMAL", 0),
            "status_distribution": status_counts,
            "mean_probability": mean_prob,
            "max_probability": max_prob,
            "anomaly_score": float(anomaly_score),
        }

    def create_ascii_timeline(
        self,
        analysis_result: AudioAnalysisResult,
        width: int = 80,
    ) -> str:
        """Create ASCII representation of timeline.

        Args:
            analysis_result: Analysis result.
            width: Output width in characters.

        Returns:
            ASCII timeline string.
        """
        duration = analysis_result.properties.duration
        if duration <= 0:
            return "No audio duration available"

        lines = []
        lines.append("=" * width)
        lines.append("AUDIO FORENSIC TIMELINE")
        lines.append("=" * width)

        # Header
        lines.append(f"Duration: {duration:.1f}s")
        lines.append(f"Segments: {analysis_result.total_segments}")
        lines.append("")

        # Timeline bar
        bar_width = width - 20
        lines.append("0s" + " " * (bar_width - 10) + f"{duration:.1f}s")
        lines.append("-" * width)

        # Create bar
        bar = ["."] * bar_width

        for segment in analysis_result.segments:
            start_pos = int((segment.start_time / duration) * bar_width)
            end_pos = int((segment.end_time / duration) * bar_width)

            # Get character based on status
            status = segment.prediction.value
            if status == "NORMAL":
                char = "="
            elif status == "LOW_SUSPICION":
                char = "-"
            elif status == "SUSPICIOUS":
                char = "~"
            elif status == "POTENTIAL_MANIPULATION":
                char = "!"
            else:
                char = "?"

            # Fill bar
            for i in range(start_pos, min(end_pos, bar_width)):
                bar[i] = char

        lines.append("[" + "".join(bar) + "]")

        # Legend
        lines.append("")
        lines.append("Legend:")
        lines.append("  = Normal")
        lines.append("  - Low Suspicion")
        lines.append("  ~ Suspicious")
        lines.append("  ! Potential Manipulation")
        lines.append("")

        # Segment details
        lines.append("Segment Details:")
        lines.append("-" * width)

        for i, segment in enumerate(analysis_result.segments[:10]):
            status = segment.prediction.value
            lines.append(
                f"  {i+1:3d}. [{segment.start_time:6.1f}s - "
                f"{segment.end_time:6.1f}s] "
                f"{status:25s} "
                f"prob={segment.probability:.3f} "
                f"conf={segment.confidence:.3f}"
            )

        if len(analysis_result.segments) > 10:
            lines.append(
                f"  ... and {len(analysis_result.segments) - 10} more segments"
            )

        lines.append("=" * width)

        return "\n".join(lines)

    def create_html_timeline(
        self, analysis_result: AudioAnalysisResult
    ) -> str:
        """Create HTML representation of timeline.

        Args:
            analysis_result: Analysis result.

        Returns:
            HTML string.
        """
        duration = analysis_result.properties.duration
        segments = analysis_result.segments

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        .timeline-container {{
            font-family: Arial, sans-serif;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        .timeline-header {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .timeline-bar {{
            position: relative;
            height: 40px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }}
        .timeline-segment {{
            position: absolute;
            height: 100%;
            opacity: 0.8;
        }}
        .timeline-segment:hover {{
            opacity: 1;
        }}
        .legend {{
            margin-top: 15px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        .stats {{
            margin-top: 15px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="timeline-container">
        <div class="timeline-header">Audio Forensic Timeline</div>
        <p>Duration: {duration:.1f}s | Segments: {len(segments)}</p>

        <div class="timeline-bar">
"""

        # Add segments (guard zero-duration timelines)
        duration = duration if duration > 0 else 1e-9
        for segment in segments:
            start_pct = (segment.start_time / duration) * 100
            width_pct = ((segment.end_time - segment.start_time) / duration) * 100
            status = segment.prediction.value
            color = self.STATUS_COLORS.get(status, "#9E9E9E")

            html += f"""
            <div class="timeline-segment"
                 style="left: {start_pct:.2f}%; width: {width_pct:.2f}%; background: {color};"
                 title="{status}: {segment.start_time:.1f}s - {segment.end_time:.1f}s (prob: {segment.probability:.3f})">
            </div>
"""

        html += """
        </div>

        <div class="legend">
"""
        # Add legend
        for status, color in self.STATUS_COLORS.items():
            label = self.STATUS_LABELS.get(status, status)
            html += f"""
            <div class="legend-item">
                <div class="legend-color" style="background: {color};"></div>
                <span>{label}</span>
            </div>
"""
        html += """
        </div>

        <div class="stats">
"""
        # Add stats
        summary = self._generate_summary(analysis_result)
        html += f"""
            <p><strong>Total Segments:</strong> {summary['total_segments']}</p>
            <p><strong>Suspicious Segments:</strong> {summary['suspicious_segments']}</p>
            <p><strong>Mean Probability:</strong> {summary['mean_probability']:.3f}</p>
            <p><strong>Max Probability:</strong> {summary['max_probability']:.3f}</p>
            <p><strong>Anomaly Score:</strong> {summary['anomaly_score']:.3f}</p>
"""
        html += """
        </div>
    </div>
</body>
</html>
"""
        return html


def get_timeline(
    config: Optional[AudioForensicsConfig] = None,
) -> AudioForensicTimelineGenerator:
    """Factory function to get timeline generator."""
    return AudioForensicTimelineGenerator(config)
