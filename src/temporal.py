"""
src/temporal.py
Video temporal analysis for the Deepfake Detection System.

Provides:
- Frame-to-frame consistency analysis
- Temporal anomaly detection
- Motion pattern analysis
- Face tracking across frames
- Temporal scoring and aggregation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class TemporalAnomalyType(Enum):
    """Types of temporal anomalies."""
    JITTER = "jitter"
    FLICKER = "flicker"
    TELEPORTATION = "teleportation"
    BLUR_ARTIFACT = "blur_artifact"
    COLOR_SHIFT = "color_shift"
    GEOMETRIC_DISTORTION = "geometric_distortion"


@dataclass
class FrameAnalysis:
    """Analysis results for a single frame."""
    frame_idx: int
    timestamp: float
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_embedding: Optional[np.ndarray] = None
    prediction: float = 0.0
    confidence: float = 0.0
    motion_vector: Optional[Tuple[float, float]] = None
    optical_flow_magnitude: float = 0.0


@dataclass
class TemporalSegment:
    """A segment of consecutive frames with similar characteristics."""
    start_frame: int
    end_frame: int
    avg_prediction: float
    avg_confidence: float
    anomaly_score: float
    anomaly_types: List[TemporalAnomalyType] = field(default_factory=list)


@dataclass
class TemporalAnalysisResult:
    """Complete temporal analysis result for a video."""
    video_path: str
    total_frames: int
    analyzed_frames: int
    frame_analyses: List[FrameAnalysis]
    temporal_segments: List[TemporalSegment]
    overall_anomaly_score: float
    temporal_consistency: float
    face_tracking_score: float
    motion_smoothness: float
    manipulation_timeline: List[Dict[str, Any]]
    summary: Dict[str, Any]
    analysis_time: float = 0.0


class OpticalFlowAnalyzer:
    """Analyze optical flow between frames."""
    
    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
    
    def compute_flow(
        self,
        frame: np.ndarray,
        method: str = "farneback"
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute optical flow from previous frame.
        
        Args:
            frame: Current frame (BGR)
            method: Optical flow method ('farneback' or 'lucas_kanade')
            
        Returns:
            Tuple of (flow_field, magnitude)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0.0
        
        if method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = float(np.mean(magnitude))
        else:
            # Lucas-Kanade sparse flow
            prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=200, qualityLevel=0.01,
                minDistance=10, blockSize=7
            )
            if prev_pts is None:
                self.prev_gray = gray
                return None, 0.0
            
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_pts, None
            )
            
            if curr_pts is None or status is None:
                self.prev_gray = gray
                return None, 0.0
            
            # Filter valid points
            valid_mask = status.flatten() == 1
            if np.sum(valid_mask) == 0:
                self.prev_gray = gray
                return None, 0.0
            
            prev_valid = prev_pts[valid_mask]
            curr_valid = curr_pts[valid_mask]
            
            # Compute displacements
            displacements = curr_valid - prev_valid
            magnitude = np.sqrt(np.sum(displacements**2, axis=1))
            avg_magnitude = float(np.mean(magnitude))
            
            # Create dense flow for compatibility
            flow = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.float32)
            for pt, disp in zip(prev_valid, displacements):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
                    flow[y, x] = disp
        
        self.prev_gray = gray
        return flow, avg_magnitude
    
    def reset(self):
        """Reset the analyzer state."""
        self.prev_gray = None


class FaceTracker:
    """Track faces across video frames."""
    
    def __init__(self, max_disappearance: int = 5):
        """
        Initialize face tracker.
        
        Args:
            max_disappearance: Maximum frames a face can disappear before deregistration
        """
        self.max_disappearance = max_disappearance
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        self.disappeared: Dict[int, int] = {}
    
    def update(
        self,
        faces: List[Tuple[int, int, int, int]],
        predictions: List[float]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Update tracker with new face detections.
        
        Args:
            faces: List of face bounding boxes
            predictions: List of prediction scores for each face
            
        Returns:
            Dictionary of tracked faces
        """
        if len(faces) == 0:
            # Mark all existing tracks as disappeared
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappearance:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return self.tracks
        
        # If no existing tracks, create new ones
        if len(self.tracks) == 0:
            for i, (face, pred) in enumerate(zip(faces, predictions)):
                self.tracks[self.next_id] = {
                    "bbox": face,
                    "prediction": pred,
                    "history": [pred],
                }
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            # Match new detections to existing tracks
            track_ids = list(self.tracks.keys())
            track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]
            
            # Compute distances
            matched_ids = []
            unmatched_detections = list(range(len(faces)))
            
            for i, (face, pred) in enumerate(zip(faces, predictions)):
                best_dist = float('inf')
                best_id = None
                
                for j, (track_id, track_bbox) in enumerate(zip(track_ids, track_bboxes)):
                    # Simple IoU-based matching
                    iou = self._compute_iou(face, track_bbox)
                    dist = 1.0 - iou  # Convert to distance
                    
                    if dist < best_dist and dist < 0.5:
                        best_dist = dist
                        best_id = track_ids[j]
                
                if best_id is not None:
                    # Update existing track
                    self.tracks[best_id]["bbox"] = face
                    self.tracks[best_id]["prediction"] = pred
                    self.tracks[best_id]["history"].append(pred)
                    self.disappeared[best_id] = 0
                    matched_ids.append(best_id)
                    unmatched_detections.remove(i)
            
            # Mark unmatched tracks as disappeared
            for track_id in track_ids:
                if track_id not in matched_ids:
                    self.disappeared[track_id] += 1
                    if self.disappeared[track_id] > self.max_disappearance:
                        del self.tracks[track_id]
                        del self.disappeared[track_id]
            
            # Create new tracks for unmatched detections
            for i in unmatched_detections:
                face = faces[i]
                pred = predictions[i]
                self.tracks[self.next_id] = {
                    "bbox": face,
                    "prediction": pred,
                    "history": [pred],
                }
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        
        return self.tracks
    
    def _compute_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_tracks(self) -> Dict[int, Dict[str, Any]]:
        """Return current tracks."""
        return self.tracks
    
    def reset(self):
        """Reset the tracker."""
        self.tracks.clear()
        self.disappeared.clear()
        self.next_id = 0


class TemporalAnalyzer:
    """
    Comprehensive temporal analysis for video deepfake detection.
    
    Analyzes frame-to-frame consistency, motion patterns, and temporal anomalies.
    """
    
    def __init__(
        self,
        face_detector=None,
        device: str = "cpu",
        frame_skip: int = 1
    ):
        """
        Initialize temporal analyzer.
        
        Args:
            face_detector: Face detector instance
            device: Device for inference
            frame_skip: Process every Nth frame
        """
        self.face_detector = face_detector
        self.device = device
        self.frame_skip = frame_skip
        self.optical_flow = OpticalFlowAnalyzer()
        self.face_tracker = FaceTracker(max_disappearance=5)
    
    def analyze_video(
        self,
        video_path: str,
        detector,
        max_frames: int = 64
    ) -> TemporalAnalysisResult:
        """
        Perform comprehensive temporal analysis on a video.
        
        Args:
            video_path: Path to the video file
            detector: DeepfakeDetector instance
            max_frames: Maximum frames to analyze
            
        Returns:
            TemporalAnalysisResult with full analysis
        """
        start_time = time.time()
        logger.info("Starting temporal analysis: %s", video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        frame_analyses = []
        frame_count = 0
        processed_count = 0
        
        # Reset analyzers
        self.optical_flow.reset()
        self.face_tracker.reset()
        
        try:
            while processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Compute timestamp
                timestamp = frame_count / fps
                
                # Face detection
                face_bbox = None
                prediction = 0.0
                confidence = 0.0
                
                if self.face_detector:
                    boxes = self.face_detector.detect_faces(frame)
                    if boxes:
                        # Use largest face
                        face_bbox = max(
                            boxes,
                            key=lambda b: (b[2]-b[0]) * (b[3]-b[1])
                        )
                        
                        # Crop and predict
                        x0, y0, x1, y1 = face_bbox
                        face_crop = frame[y0:y1, x0:x1]
                        
                        if face_crop.size > 0:
                            result = detector.predict_image_array(face_crop)
                            prediction = result.get("fake_probability", 0.0)
                            confidence = result.get("confidence", 0.0)
                
                # Optical flow
                flow, flow_magnitude = self.optical_flow.compute_flow(frame)
                
                # Motion vector
                motion_vector = None
                if flow is not None:
                    # Compute average motion in center region
                    h, w = flow.shape[:2]
                    center_region = flow[h//4:3*h//4, w//4:3*w//4]
                    if center_region.size > 0:
                        avg_flow = np.mean(center_region, axis=(0, 1))
                        motion_vector = (float(avg_flow[0]), float(avg_flow[1]))
                
                # Create frame analysis
                analysis = FrameAnalysis(
                    frame_idx=frame_count,
                    timestamp=timestamp,
                    face_bbox=face_bbox,
                    prediction=prediction,
                    confidence=confidence,
                    motion_vector=motion_vector,
                    optical_flow_magnitude=flow_magnitude,
                )
                frame_analyses.append(analysis)
                
                # Update face tracker
                if face_bbox:
                    self.face_tracker.update([face_bbox], [prediction])
                
                processed_count += 1
            
            # Analyze temporal patterns
            temporal_segments = self._segment_video(frame_analyses, fps)
            anomaly_score = self._compute_anomaly_score(frame_analyses, temporal_segments)
            consistency = self._compute_temporal_consistency(frame_analyses)
            face_tracking_score = self._compute_tracking_score()
            motion_smoothness = self._compute_motion_smoothness(frame_analyses)
            manipulation_timeline = self._extract_manipulation_timeline(
                frame_analyses, fps
            )
            
            # Generate summary
            summary = {
                "total_frames": total_frames,
                "analyzed_frames": processed_count,
                "fps": fps,
                "duration": total_frames / fps,
                "overall_anomaly_score": anomaly_score,
                "temporal_consistency": consistency,
                "face_tracking_score": face_tracking_score,
                "motion_smoothness": motion_smoothness,
                "num_segments": len(temporal_segments),
                "suspicious_segments": sum(
                    1 for s in temporal_segments if s.anomaly_score > 0.5
                ),
            }
            
            analysis_time = time.time() - start_time
            
            result = TemporalAnalysisResult(
                video_path=video_path,
                total_frames=total_frames,
                analyzed_frames=processed_count,
                frame_analyses=frame_analyses,
                temporal_segments=temporal_segments,
                overall_anomaly_score=anomaly_score,
                temporal_consistency=consistency,
                face_tracking_score=face_tracking_score,
                motion_smoothness=motion_smoothness,
                manipulation_timeline=manipulation_timeline,
                summary=summary,
                analysis_time=analysis_time,
            )
            
            logger.info(
                "Temporal analysis complete: %s (anomaly=%.3f, consistency=%.3f)",
                video_path, anomaly_score, consistency
            )
            
            return result
            
        finally:
            cap.release()
    
    def _segment_video(
        self,
        analyses: List[FrameAnalysis],
        fps: float,
        segment_duration: float = 1.0
    ) -> List[TemporalSegment]:
        """Segment video into temporal segments."""
        if not analyses:
            return []
        
        frames_per_segment = int(segment_duration * fps)
        segments = []
        
        for i in range(0, len(analyses), frames_per_segment):
            segment_analyses = analyses[i:i + frames_per_segment]
            
            if not segment_analyses:
                continue
            
            predictions = [a.prediction for a in segment_analyses]
            confidences = [a.confidence for a in segment_analyses]
            
            avg_pred = float(np.mean(predictions))
            avg_conf = float(np.mean(confidences))
            
            # Compute anomaly score for segment
            pred_std = float(np.std(predictions))
            anomaly_score = pred_std  # High variance indicates anomaly
            
            # Detect specific anomalies
            anomaly_types = []
            if pred_std > 0.3:
                anomaly_types.append(TemporalAnomalyType.FLICKER)
            
            # Check for jitter (rapid changes)
            if len(predictions) > 1:
                diffs = np.diff(predictions)
                if np.max(np.abs(diffs)) > 0.5:
                    anomaly_types.append(TemporalAnomalyType.JITTER)
            
            segment = TemporalSegment(
                start_frame=segment_analyses[0].frame_idx,
                end_frame=segment_analyses[-1].frame_idx,
                avg_prediction=avg_pred,
                avg_confidence=avg_conf,
                anomaly_score=anomaly_score,
                anomaly_types=anomaly_types,
            )
            segments.append(segment)
        
        return segments
    
    def _compute_anomaly_score(
        self,
        analyses: List[FrameAnalysis],
        segments: List[TemporalSegment]
    ) -> float:
        """Compute overall anomaly score."""
        if not analyses:
            return 0.0
        
        # Base anomaly from prediction variance
        predictions = [a.prediction for a in analyses]
        base_anomaly = float(np.std(predictions))
        
        # Segment-based anomaly
        segment_anomalies = [s.anomaly_score for s in segments]
        segment_anomaly = float(np.mean(segment_anomalies)) if segment_anomalies else 0.0
        
        # Motion-based anomaly
        flow_magnitudes = [a.optical_flow_magnitude for a in analyses if a.optical_flow_magnitude > 0]
        motion_anomaly = 0.0
        if flow_magnitudes:
            motion_std = float(np.std(flow_magnitudes))
            motion_anomaly = min(1.0, motion_std / 10.0)
        
        # Combined score
        anomaly_score = (
            0.5 * base_anomaly +
            0.3 * segment_anomaly +
            0.2 * motion_anomaly
        )
        
        return float(np.clip(anomaly_score, 0.0, 1.0))
    
    def _compute_temporal_consistency(
        self,
        analyses: List[FrameAnalysis]
    ) -> float:
        """Compute temporal consistency score (0=erratic, 1=smooth)."""
        if len(analyses) < 2:
            return 1.0
        
        predictions = [a.prediction for a in analyses]
        
        # Compute frame-to-frame differences
        diffs = np.abs(np.diff(predictions))
        
        # Average difference (lower = more consistent)
        avg_diff = float(np.mean(diffs))
        
        # Convert to consistency score
        consistency = 1.0 - min(1.0, avg_diff * 2)
        
        return consistency
    
    def _compute_tracking_score(self) -> float:
        """Compute face tracking score."""
        tracks = self.face_tracker.get_tracks()
        
        if not tracks:
            return 0.0
        
        # Average track length
        track_lengths = [
            len(track.get("history", []))
            for track in tracks.values()
        ]
        
        avg_length = float(np.mean(track_lengths)) if track_lengths else 0.0
        
        # Normalize by expected length
        score = min(1.0, avg_length / 10.0)
        
        return score
    
    def _compute_motion_smoothness(
        self,
        analyses: List[FrameAnalysis]
    ) -> float:
        """Compute motion smoothness score."""
        flow_magnitudes = [
            a.optical_flow_magnitude
            for a in analyses
            if a.optical_flow_magnitude > 0
        ]
        
        if len(flow_magnitudes) < 2:
            return 1.0
        
        # Compute acceleration (change in magnitude)
        accelerations = np.abs(np.diff(flow_magnitudes))
        
        # Lower acceleration = smoother motion
        avg_acceleration = float(np.mean(accelerations))
        
        # Convert to smoothness score
        smoothness = 1.0 - min(1.0, avg_acceleration / 5.0)
        
        return smoothness
    
    def _extract_manipulation_timeline(
        self,
        analyses: List[FrameAnalysis],
        fps: float
    ) -> List[Dict[str, Any]]:
        """Extract timeline of potential manipulations."""
        timeline = []
        
        threshold = 0.6  # Threshold for suspicious predictions
        
        for analysis in analyses:
            if analysis.prediction >= threshold:
                timeline.append({
                    "frame": analysis.frame_idx,
                    "timestamp": analysis.timestamp,
                    "fake_probability": analysis.prediction,
                    "confidence": analysis.confidence,
                })
        
        return timeline


def analyze_video_temporal(
    video_path: str,
    detector,
    face_detector=None,
    max_frames: int = 64
) -> TemporalAnalysisResult:
    """
    Convenience function for temporal video analysis.
    
    Args:
        video_path: Path to the video file
        detector: DeepfakeDetector instance
        face_detector: Optional face detector
        max_frames: Maximum frames to analyze
        
    Returns:
        TemporalAnalysisResult
    """
    analyzer = TemporalAnalyzer(
        face_detector=face_detector,
        device=detector.device,
    )
    return analyzer.analyze_video(video_path, detector, max_frames)
