"""
src/metadata.py
Media metadata extraction and evidence hashing for the Deepfake Detection System.

Provides:
- File hashing (MD5, SHA-256, SHA-512)
- EXIF metadata extraction
- Video metadata extraction (codec, resolution, FPS, etc.)
- Evidence chain of custody tracking
- Content fingerprinting
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FileHash:
    """File hash information."""
    md5: str
    sha256: str
    sha512: str
    file_size: int
    file_name: str


@dataclass
class ImageMetadata:
    """Extracted image metadata."""
    file_name: str
    file_path: str
    file_hash: FileHash
    width: int
    height: int
    channels: int
    dtype: str
    exif_data: Dict[str, Any] = field(default_factory=dict)
    creation_time: Optional[str] = None
    modification_time: Optional[str] = None
    camera_model: Optional[str] = None
    gps_info: Optional[Dict[str, float]] = None


@dataclass
class VideoMetadata:
    """Extracted video metadata."""
    file_name: str
    file_path: str
    file_hash: FileHash
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec_name: str
    codec_long_name: str
    pixel_format: str
    bit_rate: Optional[int] = None
    creation_time: Optional[str] = None
    rotation: Optional[int] = None


@dataclass
class EvidenceRecord:
    """Chain of custody record for forensic analysis."""
    evidence_id: str
    file_hash: FileHash
    extraction_time: str
    metadata: Dict[str, Any]
    prediction_result: Optional[Dict[str, Any]] = None
    analysis_notes: Optional[str] = None
    analyst: Optional[str] = None


class FileHasher:
    """Compute file hashes for evidence integrity."""
    
    def __init__(self, chunk_size: int = 8192):
        """
        Initialize file hasher.
        
        Args:
            chunk_size: Size of chunks for reading large files
        """
        self.chunk_size = chunk_size
    
    def compute_hashes(self, file_path: str) -> FileHash:
        """
        Compute MD5, SHA-256, and SHA-512 hashes for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileHash with all computed hashes
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        md5_hash = hashlib.md5()
        sha256_hash = hashlib.sha256()
        sha512_hash = hashlib.sha512()
        
        file_size = 0
        
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                md5_hash.update(chunk)
                sha256_hash.update(chunk)
                sha512_hash.update(chunk)
                file_size += len(chunk)
        
        file_hash = FileHash(
            md5=md5_hash.hexdigest(),
            sha256=sha256_hash.hexdigest(),
            sha512=sha512_hash.hexdigest(),
            file_size=file_size,
            file_name=os.path.basename(file_path),
        )
        
        logger.info(
            "File hashes computed: %s (size=%d bytes)",
            file_hash.file_name, file_size
        )
        
        return file_hash
    
    def compute_content_hash(self, data: bytes) -> str:
        """
        Compute SHA-256 hash of raw data content.
        
        Args:
            data: Raw bytes to hash
            
        Returns:
            SHA-256 hex digest
        """
        return hashlib.sha256(data).hexdigest()
    
    def verify_hash(self, file_path: str, expected_hash: str, algorithm: str = "sha256") -> bool:
        """
        Verify file hash matches expected value.
        
        Args:
            file_path: Path to file
            expected_hash: Expected hash value
            algorithm: Hash algorithm (md5, sha256, sha512)
            
        Returns:
            True if hash matches
        """
        file_hash = self.compute_hashes(file_path)
        
        hash_map = {
            "md5": file_hash.md5,
            "sha256": file_hash.sha256,
            "sha512": file_hash.sha512,
        }
        
        actual_hash = hash_map.get(algorithm)
        if actual_hash is None:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        match = actual_hash.lower() == expected_hash.lower()
        
        if not match:
            logger.warning(
                "Hash mismatch for %s: expected=%s, actual=%s",
                file_path, expected_hash[:16], actual_hash[:16]
            )
        
        return match


class MetadataExtractor:
    """Extract metadata from image and video files."""
    
    def __init__(self):
        self.hasher = FileHasher()
    
    def extract_image_metadata(self, image_path: str) -> ImageMetadata:
        """
        Extract metadata from an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ImageMetadata with extracted information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Compute file hashes
        file_hash = self.hasher.compute_hashes(image_path)
        
        # Read image with OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        # Get file timestamps
        stat = os.stat(image_path)
        creation_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
        modification_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        # Try to extract EXIF data
        exif_data = self._extract_exif(image_path)
        
        metadata = ImageMetadata(
            file_name=os.path.basename(image_path),
            file_path=image_path,
            file_hash=file_hash,
            width=w,
            height=h,
            channels=channels,
            dtype=str(image.dtype),
            exif_data=exif_data,
            creation_time=creation_time,
            modification_time=modification_time,
        )
        
        logger.info(
            "Image metadata extracted: %s (%dx%d, %d channels)",
            metadata.file_name, w, h, channels
        )
        
        return metadata
    
    def extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoMetadata with extracted information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Compute file hashes
        file_hash = self.hasher.compute_hashes(video_path)
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_name = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Calculate duration
            duration = total_frames / fps if fps > 0 else 0
            
            # Get file timestamps
            stat = os.stat(video_path)
            creation_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
            
            metadata = VideoMetadata(
                file_name=os.path.basename(video_path),
                file_path=video_path,
                file_hash=file_hash,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                codec_name=codec_name,
                codec_long_name=codec_name,
                pixel_format="unknown",
                creation_time=creation_time,
            )
            
            logger.info(
                "Video metadata extracted: %s (%dx%d, %.2f FPS, %.1f seconds)",
                metadata.file_name, width, height, fps, duration
            )
            
            return metadata
            
        finally:
            cap.release()
    
    def _extract_exif(self, file_path: str) -> Dict[str, Any]:
        """
        Extract EXIF data from an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary of EXIF data
        """
        exif_data = {}
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            with Image.open(file_path) as img:
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        # Convert non-serializable values to strings
                        try:
                            json.dumps(value)
                            exif_data[tag] = value
                        except (TypeError, ValueError):
                            exif_data[tag] = str(value)
                            
        except ImportError:
            logger.debug("PIL not available for EXIF extraction")
        except Exception as e:
            logger.debug("EXIF extraction failed: %s", e)
        
        return exif_data


class EvidenceTracker:
    """
    Track evidence chain of custody for forensic analysis.
    
    Maintains a record of all analysis performed on media files.
    """
    
    def __init__(self, output_dir: str = "results/evidence"):
        """
        Initialize evidence tracker.
        
        Args:
            output_dir: Directory to store evidence records
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hasher = FileHasher()
    
    def create_evidence_record(
        self,
        file_path: str,
        prediction_result: Optional[Dict[str, Any]] = None,
        analysis_notes: Optional[str] = None,
        analyst: Optional[str] = None,
    ) -> EvidenceRecord:
        """
        Create an evidence record for a file.
        
        Args:
            file_path: Path to the evidence file
            prediction_result: Optional prediction results to attach
            analysis_notes: Optional notes about the analysis
            analyst: Optional name of the analyst
            
        Returns:
            EvidenceRecord with all information
        """
        # Compute file hash
        file_hash = self.hasher.compute_hashes(file_path)
        
        # Generate evidence ID
        evidence_id = f"EV-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{file_hash.sha256[:8]}"
        
        # Extract basic metadata
        metadata = {}
        try:
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                extractor = MetadataExtractor()
                img_meta = extractor.extract_image_metadata(file_path)
                metadata = {
                    "width": img_meta.width,
                    "height": img_meta.height,
                    "channels": img_meta.channels,
                }
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                extractor = MetadataExtractor()
                vid_meta = extractor.extract_video_metadata(file_path)
                metadata = {
                    "width": vid_meta.width,
                    "height": vid_meta.height,
                    "fps": vid_meta.fps,
                    "duration": vid_meta.duration_seconds,
                }
        except Exception as e:
            logger.warning("Failed to extract metadata: %s", e)
        
        record = EvidenceRecord(
            evidence_id=evidence_id,
            file_hash=file_hash,
            extraction_time=datetime.now().isoformat(),
            metadata=metadata,
            prediction_result=prediction_result,
            analysis_notes=analysis_notes,
            analyst=analyst,
        )
        
        # Save evidence record
        self._save_record(record)
        
        logger.info(
            "Evidence record created: %s for %s",
            evidence_id, file_hash.file_name
        )
        
        return record
    
    def _save_record(self, record: EvidenceRecord) -> None:
        """
        Save evidence record to disk.
        
        Args:
            record: EvidenceRecord to save
        """
        output_file = self.output_dir / f"{record.evidence_id}.json"
        
        with open(output_file, "w") as f:
            json.dump(asdict(record), f, indent=2, default=str)
        
        logger.debug("Evidence record saved: %s", output_file)
    
    def load_record(self, evidence_id: str) -> Optional[EvidenceRecord]:
        """
        Load an evidence record by ID.
        
        Args:
            evidence_id: Evidence ID to load
            
        Returns:
            EvidenceRecord if found, None otherwise
        """
        output_file = self.output_dir / f"{evidence_id}.json"
        
        if not output_file.exists():
            logger.warning("Evidence record not found: %s", evidence_id)
            return None
        
        with open(output_file, "r") as f:
            data = json.load(f)
        
        # Reconstruct FileHash
        data["file_hash"] = FileHash(**data["file_hash"])
        
        return EvidenceRecord(**data)
    
    def list_records(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[EvidenceRecord]:
        """
        List evidence records with optional date filtering.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            List of EvidenceRecord objects
        """
        records = []
        
        for file_path in self.output_dir.glob("EV-*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Reconstruct FileHash
                data["file_hash"] = FileHash(**data["file_hash"])
                record = EvidenceRecord(**data)
                
                # Apply date filtering
                if start_date and record.extraction_time < start_date:
                    continue
                if end_date and record.extraction_time > end_date:
                    continue
                
                records.append(record)
            except Exception as e:
                logger.warning("Failed to load record %s: %s", file_path, e)
        
        return sorted(records, key=lambda r: r.extraction_time, reverse=True)


class ContentFingerprint:
    """
    Generate content fingerprints for media deduplication and similarity detection.
    
    Uses perceptual hashing for images and temporal signatures for videos.
    """
    
    def __init__(self):
        pass
    
    def compute_image_fingerprint(self, image: np.ndarray) -> str:
        """
        Compute a perceptual fingerprint for an image.
        
        Uses a simplified pHash-like approach.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Hex string representing the fingerprint
        """
        # Resize to small dimensions
        small = cv2.resize(image, (32, 32))
        
        # Convert to grayscale
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small
        
        # Compute DCT
        dct = cv2.dct(np.float32(gray))
        
        # Keep top-left corner (low frequencies)
        dct_low = dct[:8, :8]
        
        # Compute median
        med = np.median(dct_low)
        
        # Create binary fingerprint
        fingerprint = (dct_low > med).flatten()
        
        # Convert to hex
        fingerprint_int = 0
        for bit in fingerprint:
            fingerprint_int = (fingerprint_int << 1) | int(bit)
        
        return format(fingerprint_int, '032x')
    
    def compute_video_fingerprint(
        self,
        video_path: str,
        n_frames: int = 16
    ) -> str:
        """
        Compute a temporal fingerprint for a video.
        
        Args:
            video_path: Path to the video file
            n_frames: Number of frames to sample
            
        Returns:
            Hex string representing the video fingerprint
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Video has no frames")
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, n_frames).astype(int)
            
            fingerprints = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    fp = self.compute_image_fingerprint(frame)
                    fingerprints.append(fp)
            
            # Combine fingerprints
            combined = ''.join(fingerprints)
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
            
        finally:
            cap.release()
    
    def compute_similarity(self, fp1: str, fp2: str) -> float:
        """
        Compute similarity between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert hex to binary
        fp1_int = int(fp1, 16)
        fp2_int = int(fp2, 16)
        
        # XOR to find differences
        xor = fp1_int ^ fp2_int
        
        # Count different bits (Hamming distance)
        diff_bits = bin(xor).count('1')
        total_bits = len(fp1) * 4  # 4 bits per hex char
        
        # Similarity = 1 - normalized hamming distance
        similarity = 1.0 - (diff_bits / total_bits)
        
        return similarity


def compute_file_hashes(file_path: str) -> FileHash:
    """
    Convenience function to compute file hashes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        FileHash with all computed hashes
    """
    hasher = FileHasher()
    return hasher.compute_hashes(file_path)


def extract_media_metadata(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract metadata from any media file.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        Dictionary with extracted metadata
    """
    extractor = MetadataExtractor()
    
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        metadata = extractor.extract_image_metadata(file_path)
        return asdict(metadata)
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        metadata = extractor.extract_video_metadata(file_path)
        return asdict(metadata)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
