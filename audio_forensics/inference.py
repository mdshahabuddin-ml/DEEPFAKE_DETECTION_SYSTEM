"""Audio inference pipeline for deepfake detection.

This module orchestrates the complete audio analysis pipeline:
    Audio Input -> Preprocessing -> Segmentation -> Feature Extraction ->
    Detection -> Aggregation -> Result

The pipeline is completely isolated from the existing visual detection system.
"""

import hashlib
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import AudioForensicsConfig
from .detector import AudioDetector, DetectionResult, get_detector
from .features import AudioFeatureExtractor, get_feature_extractor
from .preprocessing import AudioPreprocessor, get_preprocessor
from .schemas import (
    AudioAnalysisResult,
    AudioProperties,
    AudioStatus,
    SegmentResult,
    SpeechType,
)
from .segmentation import AudioSegmenter, get_segmenter

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Custom exception for inference errors."""
    pass


class AudioInferencePipeline:
    """Complete audio deepfake detection pipeline.

    This class orchestrates the entire audio analysis workflow:
    1. Preprocessing
    2. Segmentation
    3. Feature extraction
    4. Detection
    5. Aggregation

    Usage:
        config = AudioForensicsConfig()
        pipeline = AudioInferencePipeline(config)
        result = pipeline.analyze("path/to/audio.wav")
    """

    def __init__(self, config: Optional[AudioForensicsConfig] = None):
        """Initialize inference pipeline.

        Args:
            config: Audio forensics configuration.
        """
        self.config = config or AudioForensicsConfig()

        # Initialize components
        self._preprocessor = get_preprocessor(self.config)
        self._segmenter = get_segmenter(self.config)
        self._feature_extractor = get_feature_extractor(self.config)
        self._detector = get_detector(self.config)

        # Try to load model
        self._model_loaded = self._detector.load_model()

    @property
    def is_ready(self) -> bool:
        """Check if pipeline is ready for analysis."""
        return True  # Pipeline works with or without model

    @property
    def model_loaded(self) -> bool:
        """Check if detection model is loaded."""
        return self._model_loaded

    def analyze(
        self,
        audio_path: str,
        return_segments: bool = True,
        working_dir: Optional[str] = None,
    ) -> AudioAnalysisResult:
        """Analyze audio file for deepfake detection.

        Args:
            audio_path: Path to audio file.
            return_segments: Whether to include segment details.
            working_dir: Optional directory for working files.

        Returns:
            AudioAnalysisResult with complete analysis.
        """
        start_time = time.time()
        analysis_id = f"AUD-{uuid.uuid4().hex[:12]}"

        logger.info(f"Starting analysis: {analysis_id}")

        try:
            # Step 1: Preprocessing
            logger.debug("Step 1: Preprocessing")
            preprocess_result = self._preprocessor.preprocess(
                audio_path, working_dir
            )

            # Step 2: Segmentation
            logger.debug("Step 2: Segmentation")
            seg_result = self._segmenter.segment(
                preprocess_result.audio,
                preprocess_result.sample_rate,
            )

            # Step 3: Segment-level analysis
            logger.debug("Step 3: Segment analysis")
            segment_results = []
            all_features = []

            for i, segment in enumerate(seg_result.segments):
                # Extract segment audio
                start_sample = segment.start_sample
                end_sample = segment.end_sample
                segment_audio = preprocess_result.audio[start_sample:end_sample]

                if len(segment_audio) < self.config.min_speech_duration * preprocess_result.sample_rate:
                    continue

                # Extract features
                features = self._feature_extractor.extract(
                    segment_audio,
                    preprocess_result.sample_rate,
                )

                # Detect
                feature_array = features.to_array()
                detection = self._detector.detect(
                    feature_array,
                    features.feature_names,
                )

                # Create segment result
                seg_result_item = SegmentResult(
                    segment_id=i,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    duration=segment.duration,
                    speech_type=segment.speech_type,
                    prediction=detection.prediction,
                    probability=detection.probability,
                    confidence=detection.confidence,
                    features_used=features.feature_names,
                )

                segment_results.append(seg_result_item)
                all_features.append(feature_array)

            # Step 4: Aggregate results
            logger.debug("Step 4: Aggregation")
            overall = self._aggregate_results(
                segment_results,
                preprocess_result.properties,
            )

            # Step 5: Build timeline
            timeline = self._build_timeline(segment_results)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Build final result
            result = AudioAnalysisResult(
                file_path=audio_path,
                file_hash=preprocess_result.file_hash,
                analysis_id=analysis_id,
                properties=preprocess_result.properties,
                overall_prediction=overall["prediction"],
                overall_probability=overall["probability"],
                overall_confidence=overall["confidence"],
                segments=segment_results if return_segments else [],
                total_segments=len(segment_results),
                suspicious_segments=sum(
                    1 for s in segment_results
                    if s.prediction in [
                        AudioStatus.SUSPICIOUS,
                        AudioStatus.POTENTIAL_MANIPULATION,
                    ]
                ),
                timeline=timeline,
                model_name=self._detector.architecture.value,
                processing_time=processing_time,
                evidence=overall.get("evidence", []),
                warnings=preprocess_result.warnings,
                recommendations=overall.get("recommendations", []),
            )

            logger.info(
                f"Analysis complete: {analysis_id} - "
                f"{result.overall_prediction.value} "
                f"(prob={result.overall_probability:.3f})"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Analysis failed: {e}")

            return AudioAnalysisResult(
                file_path=audio_path,
                analysis_id=analysis_id,
                overall_prediction=AudioStatus.REQUIRES_REVIEW,
                overall_probability=0.0,
                overall_confidence=0.0,
                model_name=self._detector.architecture.value,
                processing_time=processing_time,
                warnings=[f"Analysis failed: {str(e)}"],
                recommendations=["Please check the audio file and try again"],
            )

    def analyze_audio_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        file_hash: Optional[str] = None,
    ) -> AudioAnalysisResult:
        """Analyze audio array directly.

        Args:
            audio: Audio signal as numpy array.
            sample_rate: Audio sample rate.
            file_hash: Optional pre-computed hash.

        Returns:
            AudioAnalysisResult with analysis.
        """
        start_time = time.time()
        analysis_id = f"AUD-{uuid.uuid4().hex[:12]}"

        try:
            # Ensure correct format
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            # Create properties
            properties = AudioProperties(
                duration=len(audio) / sample_rate,
                sample_rate=sample_rate,
                channels=1,
            )

            # Segmentation
            seg_result = self._segmenter.segment(audio, sample_rate)

            # Segment analysis
            segment_results = []
            for i, segment in enumerate(seg_result.segments):
                segment_audio = audio[segment.start_sample:segment.end_sample]

                features = self._feature_extractor.extract(
                    segment_audio, sample_rate
                )

                detection = self._detector.detect(
                    features.to_array(),
                    features.feature_names,
                )

                seg_result_item = SegmentResult(
                    segment_id=i,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    duration=segment.duration,
                    speech_type=segment.speech_type,
                    prediction=detection.prediction,
                    probability=detection.probability,
                    confidence=detection.confidence,
                )

                segment_results.append(seg_result_item)

            # Aggregate
            overall = self._aggregate_results(segment_results, properties)
            timeline = self._build_timeline(segment_results)

            processing_time = time.time() - start_time

            return AudioAnalysisResult(
                file_hash=file_hash or "",
                analysis_id=analysis_id,
                properties=properties,
                overall_prediction=overall["prediction"],
                overall_probability=overall["probability"],
                overall_confidence=overall["confidence"],
                segments=segment_results,
                total_segments=len(segment_results),
                suspicious_segments=sum(
                    1 for s in segment_results
                    if s.prediction in [
                        AudioStatus.SUSPICIOUS,
                        AudioStatus.POTENTIAL_MANIPULATION,
                    ]
                ),
                timeline=timeline,
                model_name=self._detector.architecture.value,
                processing_time=processing_time,
                evidence=overall.get("evidence", []),
                recommendations=overall.get("recommendations", []),
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return AudioAnalysisResult(
                analysis_id=analysis_id,
                overall_prediction=AudioStatus.REQUIRES_REVIEW,
                processing_time=processing_time,
                warnings=[f"Analysis failed: {str(e)}"],
            )

    def _aggregate_results(
        self,
        segment_results: List[SegmentResult],
        properties: AudioProperties,
    ) -> Dict[str, Any]:
        """Aggregate segment results into overall assessment."""
        if not segment_results:
            return {
                "prediction": AudioStatus.UNKNOWN,
                "probability": 0.0,
                "confidence": 0.0,
                "evidence": ["No speech segments detected"],
                "recommendations": ["Audio may not contain speech"],
            }

        # Collect probabilities
        probabilities = [s.probability for s in segment_results]
        confidences = [s.confidence for s in segment_results]

        # Weighted average (weighted by confidence)
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weighted_prob = sum(
                p * c for p, c in zip(probabilities, confidences)
            ) / total_confidence
        else:
            weighted_prob = np.mean(probabilities)

        # Calculate overall confidence
        avg_confidence = np.mean(confidences)
        n_segments = len(segment_results)
        n_suspicious = sum(
            1 for s in segment_results
            if s.prediction in [
                AudioStatus.SUSPICIOUS,
                AudioStatus.POTENTIAL_MANIPULATION,
            ]
        )

        # Adjust confidence based on consistency
        if n_segments > 1:
            prob_std = np.std(probabilities)
            consistency_factor = 1.0 - min(prob_std, 0.5)
            overall_confidence = avg_confidence * consistency_factor
        else:
            overall_confidence = avg_confidence * 0.7  # Lower for single segment

        # Determine prediction
        if weighted_prob >= self.config.high_confidence_threshold:
            prediction = AudioStatus.POTENTIAL_MANIPULATION
        elif weighted_prob >= self.config.fake_threshold:
            prediction = AudioStatus.SUSPICIOUS
        elif weighted_prob >= self.config.suspicious_threshold:
            prediction = AudioStatus.LOW_SUSPICION
        else:
            prediction = AudioStatus.NORMAL

        # Build evidence
        evidence = []
        if n_suspicious > 0:
            evidence.append(
                f"{n_suspicious} of {n_segments} segments flagged as suspicious"
            )

        if properties.has_clipping:
            evidence.append("Audio clipping detected")

        if properties.snr_estimate < 10:
            evidence.append("Low signal-to-noise ratio")

        # Build recommendations
        recommendations = []
        if prediction in [AudioStatus.SUSPICIOUS, AudioStatus.POTENTIAL_MANIPULATION]:
            recommendations.append("Manual review recommended")
            recommendations.append("Consider additional analysis")

        if n_segments < 3:
            recommendations.append("Limited speech segments for analysis")

        return {
            "prediction": prediction,
            "probability": float(weighted_prob),
            "confidence": float(overall_confidence),
            "evidence": evidence,
            "recommendations": recommendations,
        }

    def _build_timeline(
        self, segment_results: List[SegmentResult]
    ) -> List[Dict[str, Any]]:
        """Build timeline from segment results."""
        timeline = []

        for segment in segment_results:
            timeline.append({
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "prediction": segment.prediction.value,
                "probability": segment.probability,
                "confidence": segment.confidence,
                "speech_type": segment.speech_type.value,
            })

        return timeline

    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            "model_loaded": self._model_loaded,
            "architecture": self._detector.architecture.value,
            "config": {
                "target_sample_rate": self.config.target_sample_rate,
                "fake_threshold": self.config.fake_threshold,
                "n_mels": self.config.n_mels,
                "n_mfcc": self.config.n_mfcc,
            },
        }


def get_pipeline(
    config: Optional[AudioForensicsConfig] = None,
) -> AudioInferencePipeline:
    """Factory function to get inference pipeline."""
    return AudioInferencePipeline(config)
