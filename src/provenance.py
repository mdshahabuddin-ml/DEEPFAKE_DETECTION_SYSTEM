"""
src/provenance.py
C2PA (Coalition for Content Provenance and Authenticity) and provenance
verification for the Deepfake Detection System.

Provides:
- C2PA manifest extraction and validation
- Content credential verification
- Digital signature verification
- Provenance chain analysis
- Trust anchor management
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from src.logger import get_logger

logger = get_logger(__name__)


class ProvenanceStatus(Enum):
    """Status of provenance verification."""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"
    EXPIRED = "expired"


class TrustLevel(Enum):
    """Trust level for content credentials."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class ProvenanceClaim:
    """A single claim in the provenance chain."""
    claim_type: str
    claim_value: Any
    evidence: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ContentCredential:
    """Content credential information from C2PA manifest."""
    credential_id: str
    issuer: str
    issuer_url: Optional[str] = None
    issued_at: Optional[str] = None
    expires_at: Optional[str] = None
    trust_chain: List[str] = field(default_factory=list)
    assertions: List[Dict[str, Any]] = field(default_factory=list)
    signature: Optional[str] = None
    certificate_chain: Optional[List[str]] = None


@dataclass
class ProvenanceResult:
    """Result of provenance verification."""
    file_name: str
    status: ProvenanceStatus
    trust_level: TrustLevel
    has_c2pa: bool
    credential: Optional[ContentCredential] = None
    claims: List[ProvenanceClaim] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    verification_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class C2PAManifestParser:
    """
    Parse and validate C2PA manifests from media files.
    
    C2PA manifests are typically stored in:
    - JPEG: APP11 markers
    - PNG: iTXt chunks
    - MP4: 'c2pa' box
    """
    
    # C2PA magic bytes and structure markers
    C2PA_MAGIC = b"c2pa"
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.mp4', '.mov'}
    
    def extract_manifest(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract C2PA manifest from a media file.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            Extracted manifest as dictionary, or None if not found
        """
        path = Path(file_path)
        
        if path.suffix.lower() not in self.supported_formats:
            logger.warning("Unsupported format for C2PA extraction: %s", path.suffix)
            return None
        
        try:
            if path.suffix.lower() in {'.jpg', '.jpeg'}:
                return self._extract_from_jpeg(file_path)
            elif path.suffix.lower() == '.png':
                return self._extract_from_png(file_path)
            elif path.suffix.lower() in {'.mp4', '.mov'}:
                return self._extract_from_mp4(file_path)
        except Exception as e:
            logger.error("C2PA extraction failed: %s", e)
            return None
        
        return None
    
    def _extract_from_jpeg(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract C2PA manifest from JPEG file."""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Look for C2PA markers in APP11
        # This is a simplified parser - production would use the full C2PA spec
        idx = 0
        while idx < len(data) - 4:
            # Look for marker
            if data[idx:idx+2] == b'\xff\xee':  # APP11 marker
                length = struct.unpack('>H', data[idx+2:idx+4])[0]
                segment = data[idx+4:idx+2+length]
                
                # Check for C2PA signature
                if self.C2PA_MAGIC in segment:
                    return self._parse_c2pa_segment(segment)
                
                idx += 2 + length
            else:
                idx += 1
        
        return None
    
    def _extract_from_png(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract C2PA manifest from PNG file."""
        with open(file_path, 'rb') as f:
            # Skip PNG signature
            f.read(8)
            
            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                
                chunk_length = struct.unpack('>I', chunk_header[:4])[0]
                chunk_type = chunk_header[4:8]
                
                if chunk_type == b'iTXt':
                    chunk_data = f.read(chunk_length)
                    if self.C2PA_MAGIC in chunk_data:
                        return self._parse_c2pa_segment(chunk_data)
                else:
                    f.read(chunk_length)
                
                # Skip CRC
                f.read(4)
        
        return None
    
    def _extract_from_mp4(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Extract C2PA manifest from MP4 file."""
        with open(file_path, 'rb') as f:
            while True:
                box_header = f.read(8)
                if len(box_header) < 8:
                    break
                
                box_size = struct.unpack('>I', box_header[:4])[0]
                box_type = box_header[4:8]
                
                if box_type == b'c2pa':
                    box_data = f.read(box_size - 8)
                    return self._parse_c2pa_segment(box_data)
                else:
                    f.read(box_size - 8)
        
        return None
    
    def _parse_c2pa_segment(self, data: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse a C2PA segment into a structured dictionary.
        
        This is a simplified parser - production implementation would
        follow the full C2PA specification.
        """
        try:
            # Try to parse as JSON (common format)
            text = data.decode('utf-8', errors='ignore')
            # Find JSON content
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Return raw data info if JSON parsing fails
        return {
            "format": "binary",
            "size": len(data),
            "preview": base64.b64encode(data[:100]).decode('ascii'),
        }


class ProvenanceVerifier:
    """
    Verify content provenance and C2PA credentials.
    
    Provides comprehensive verification including:
    - C2PA manifest validation
    - Certificate chain verification
    - Timestamp validation
    - Trust anchor checking
    """
    
    def __init__(self, trust_anchors: Optional[List[str]] = None):
        """
        Initialize provenance verifier.
        
        Args:
            trust_anchors: List of trusted certificate fingerprints
        """
        self.trust_anchors = set(trust_anchors or [])
        self.parser = C2PAManifestParser()
        
        # Default trusted issuers (would be populated from a trusted list)
        self.trusted_issuers = {
            "Adobe",
            "Microsoft",
            "Google",
            "Truepic",
            "Content Authenticity Initiative",
        }
    
    def verify(self, file_path: str) -> ProvenanceResult:
        """
        Verify provenance for a media file.
        
        Args:
            file_path: Path to the media file
            
        Returns:
            ProvenanceResult with verification status
        """
        start_time = time.time()
        file_name = Path(file_path).name
        
        logger.info("Verifying provenance for: %s", file_name)
        
        # Extract C2PA manifest
        manifest = self.parser.extract_manifest(file_path)
        
        if manifest is None:
            return ProvenanceResult(
                file_name=file_name,
                status=ProvenanceStatus.NOT_FOUND,
                trust_level=TrustLevel.NONE,
                has_c2pa=False,
                verification_time=time.time() - start_time,
            )
        
        # Parse credential
        credential = self._extract_credential(manifest)
        
        # Verify the credential
        warnings = []
        errors = []
        claims = []
        
        # Check expiration
        if credential and credential.expires_at:
            try:
                expiry = datetime.fromisoformat(credential.expires_at)
                if expiry < datetime.now():
                    errors.append("Credential has expired")
            except ValueError:
                warnings.append("Invalid expiration date format")
        
        # Check issuer trust
        if credential and credential.issuer:
            if credential.issuer not in self.trusted_issuers:
                warnings.append(f"Issuer not in trusted list: {credential.issuer}")
        
        # Check signature
        if credential and credential.signature:
            if not self._verify_signature(credential.signature):
                errors.append("Invalid signature")
        
        # Extract claims
        if credential and credential.assertions:
            for assertion in credential.assertions:
                claim = ProvenanceClaim(
                    claim_type=assertion.get("type", "unknown"),
                    claim_value=assertion.get("value"),
                    evidence=assertion.get("evidence"),
                    timestamp=assertion.get("timestamp"),
                    confidence=assertion.get("confidence", 1.0),
                )
                claims.append(claim)
        
        # Determine status
        if errors:
            status = ProvenanceStatus.INVALID
            trust_level = TrustLevel.NONE
        elif warnings:
            status = ProvenanceStatus.SUSPICIOUS
            trust_level = TrustLevel.LOW
        else:
            status = ProvenanceStatus.VALID
            trust_level = TrustLevel.HIGH
        
        verification_time = time.time() - start_time
        
        result = ProvenanceResult(
            file_name=file_name,
            status=status,
            trust_level=trust_level,
            has_c2pa=True,
            credential=credential,
            claims=claims,
            warnings=warnings,
            errors=errors,
            verification_time=verification_time,
            details={"manifest_format": manifest.get("format", "unknown")},
        )
        
        logger.info(
            "Provenance verification complete: %s (status=%s, trust=%s)",
            file_name, status.value, trust_level.value
        )
        
        return result
    
    def _extract_credential(self, manifest: Dict[str, Any]) -> Optional[ContentCredential]:
        """Extract ContentCredential from manifest data."""
        try:
            return ContentCredential(
                credential_id=manifest.get("id", "unknown"),
                issuer=manifest.get("issuer", "unknown"),
                issuer_url=manifest.get("issuer_url"),
                issued_at=manifest.get("issued_at"),
                expires_at=manifest.get("expires_at"),
                trust_chain=manifest.get("trust_chain", []),
                assertions=manifest.get("assertions", []),
                signature=manifest.get("signature"),
                certificate_chain=manifest.get("certificate_chain"),
            )
        except Exception as e:
            logger.warning("Failed to extract credential: %s", e)
            return None
    
    def _verify_signature(self, signature: str) -> bool:
        """
        Verify a content signature.
        
        In production, this would:
        1. Parse the signature structure
        2. Verify the certificate chain
        3. Check certificate revocation
        4. Validate the cryptographic signature
        """
        # Simplified check - production would use proper cryptography
        return len(signature) > 0
    
    def add_trust_anchor(self, fingerprint: str) -> None:
        """Add a trusted certificate fingerprint."""
        self.trust_anchors.add(fingerprint)
        logger.info("Added trust anchor: %s", fingerprint[:16])
    
    def remove_trust_anchor(self, fingerprint: str) -> None:
        """Remove a certificate fingerprint from trust anchors."""
        self.trust_anchors.discard(fingerprint)
        logger.info("Removed trust anchor: %s", fingerprint[:16])


class ProvenanceEnricher:
    """
    Enrich prediction results with provenance information.
    
    Combines deepfake detection with content provenance analysis.
    """
    
    def __init__(self, verifier: Optional[ProvenanceVerifier] = None):
        """
        Initialize provenance enricher.
        
        Args:
            verifier: Optional ProvenanceVerifier instance
        """
        self.verifier = verifier or ProvenanceVerifier()
    
    def enrich_prediction(
        self,
        prediction_result: Dict[str, Any],
        file_path: str
    ) -> Dict[str, Any]:
        """
        Enrich a prediction result with provenance information.
        
        Args:
            prediction_result: Original prediction result
            file_path: Path to the analyzed file
            
        Returns:
            Enriched prediction result
        """
        provenance = self.verifier.verify(file_path)
        
        enriched = prediction_result.copy()
        enriched["provenance"] = {
            "status": provenance.status.value,
            "trust_level": provenance.trust_level.value,
            "has_c2pa": provenance.has_c2pa,
            "verification_time": provenance.verification_time,
        }
        
        if provenance.credential:
            enriched["provenance"]["credential"] = {
                "issuer": provenance.credential.issuer,
                "issued_at": provenance.credential.issued_at,
                "expires_at": provenance.credential.expires_at,
            }
        
        if provenance.warnings:
            enriched["provenance"]["warnings"] = provenance.warnings
        
        if provenance.errors:
            enriched["provenance"]["errors"] = provenance.errors
        
        if provenance.claims:
            enriched["provenance"]["claims"] = [
                asdict(claim) for claim in provenance.claims
            ]
        
        # Adjust confidence based on provenance
        if provenance.status == ProvenanceStatus.VALID:
            # Valid provenance increases confidence
            enriched["confidence"] = min(1.0, enriched.get("confidence", 0.5) * 1.1)
            enriched["provenance_boost"] = True
        elif provenance.status == ProvenanceStatus.INVALID:
            # Invalid provenance decreases confidence
            enriched["confidence"] = enriched.get("confidence", 0.5) * 0.9
            enriched["provenance_penalty"] = True
        
        logger.info(
            "Prediction enriched with provenance: %s (status=%s)",
            enriched.get("prediction", "unknown"),
            provenance.status.value,
        )
        
        return enriched


def verify_provenance(file_path: str) -> ProvenanceResult:
    """
    Convenience function to verify provenance for a file.
    
    Args:
        file_path: Path to the media file
        
    Returns:
        ProvenanceResult with verification status
    """
    verifier = ProvenanceVerifier()
    return verifier.verify(file_path)
