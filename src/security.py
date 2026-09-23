"""
src/security.py
Privacy and security hardening for the Deepfake Detection System.

Provides:
- Input sanitization and validation
- Rate limiting and abuse prevention
- Data encryption at rest
- Secure inference pipeline
- Audit logging
- API security enhancements
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key_required: bool = True
    rate_limit_per_minute: int = 60
    max_upload_size_mb: int = 100
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    encryption_enabled: bool = False
    audit_logging_enabled: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15


@dataclass
class AuditEntry:
    """Audit log entry."""
    timestamp: str
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    action: str
    resource: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


class InputSanitizer:
    """Sanitize and validate inputs."""

    DANGEROUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
    ]

    @classmethod
    def sanitize_string(cls, text: str, max_length: int = 1000) -> str:
        """Sanitize a string input."""
        if not isinstance(text, str):
            text = str(text)

        text = text[:max_length]

        for pattern in cls.DANGEROUS_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        text = text.strip()

        return text

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize a filename."""
        filename = os.path.basename(filename)
        filename = re.sub(r'[^\w\-.]', '_', filename)
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_.')
        if not filename:
            filename = "unnamed_file"
        return filename

    @classmethod
    def validate_file_content(cls, file_path: str, expected_type: str) -> bool:
        """
        Validate file content matches expected type.

        Args:
            file_path: Path to the file
            expected_type: Expected file type (image, video, audio)

        Returns:
            True if content matches expected type
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)

            if expected_type == "image":
                image_signatures = [
                    b'\xff\xd8\xff',  # JPEG
                    b'\x89PNG',       # PNG
                    b'GIF87a',        # GIF
                    b'GIF89a',        # GIF
                    b'RIFF',          # WEBP (starts with RIFF)
                ]
                return any(header.startswith(sig) for sig in image_signatures)

            elif expected_type == "video":
                video_signatures = [
                    b'\x00\x00\x00',  # MP4/MOV (ftyp box)
                    b'RIFF',          # AVI
                    b'\x1aE\xdf\xa3', # MKV/WebM
                ]
                return any(header.startswith(sig) for sig in video_signatures)

            elif expected_type == "audio":
                audio_signatures = [
                    b'RIFF',          # WAV
                    b'fLaC',         # FLAC
                    b'ID3',          # MP3
                    b'\xff\xfb',     # MP3
                    b'OggS',         # OGG
                ]
                return any(header.startswith(sig) for sig in audio_signatures)

        except Exception as e:
            logger.warning("File content validation failed: %s", e)
            return False

        return True


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        if client_id not in self.requests:
            self.requests[client_id] = []

        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > window_start
        ]

        if len(self.requests[client_id]) >= self.max_requests:
            return False

        self.requests[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for a client."""
        now = time.time()
        window_start = now - self.window_seconds

        if client_id not in self.requests:
            return self.max_requests

        recent = [t for t in self.requests[client_id] if t > window_start]
        return max(0, self.max_requests - len(recent))

    def reset(self, client_id: Optional[str] = None):
        """Reset rate limiter."""
        if client_id:
            self.requests.pop(client_id, None)
        else:
            self.requests.clear()


class DataEncryptor:
    """Encrypt sensitive data at rest."""

    def __init__(self, key: Optional[str] = None):
        """
        Initialize encryptor.

        Args:
            key: Encryption key (if None, generated randomly)
        """
        if key is None:
            key = secrets.token_hex(32)
        self.key = key.encode() if isinstance(key, str) else key

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using XOR cipher (simple, for demo)."""
        key_len = len(self.key)
        encrypted = bytes(
            b ^ self.key[i % key_len]
            for i, b in enumerate(data)
        )
        return encrypted

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        return self.encrypt(encrypted_data)

    def hash_data(self, data: str) -> str:
        """Create HMAC hash of data."""
        return hmac.new(
            self.key, data.encode(), hashlib.sha256
        ).hexdigest()


class AuditLogger:
    """Security audit logging."""

    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[AuditEntry] = []

    def log(
        self,
        event_type: str,
        action: str,
        resource: str,
        status: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a security event."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            action=action,
            resource=resource,
            status=status,
            details=details or {},
        )

        self.entries.append(entry)

        log_line = (
            f"{entry.timestamp} | {entry.event_type} | {entry.action} | "
            f"{entry.resource} | {entry.status}"
        )
        if entry.user_id:
            log_line += f" | user={entry.user_id}"
        if entry.ip_address:
            log_line += f" | ip={entry.ip_address}"

        logger.info(log_line)

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Log an API request."""
        details = {}
        if duration is not None:
            details["duration_ms"] = round(duration * 1000, 2)

        self.log(
            event_type="api_request",
            action=f"{method} {path}",
            resource=path,
            status="success" if 200 <= status_code < 400 else "error",
            user_id=user_id,
            ip_address=ip_address,
            details=details,
        )

    def log_authentication(
        self,
        success: bool,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Log an authentication attempt."""
        self.log(
            event_type="authentication",
            action="login",
            resource="auth",
            status="success" if success else "failure",
            user_id=user_id,
            ip_address=ip_address,
            details={"reason": reason} if reason else {},
        )

    def log_prediction(
        self,
        file_name: str,
        prediction: str,
        confidence: float,
        user_id: Optional[str] = None,
    ) -> None:
        """Log a prediction request."""
        self.log(
            event_type="prediction",
            action="predict",
            resource=file_name,
            status="completed",
            user_id=user_id,
            details={
                "prediction": prediction,
                "confidence": confidence,
            },
        )

    def save_to_file(self) -> Path:
        """Save audit log to file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"audit_{timestamp}.json"

        import json
        with open(log_file, "w") as f:
            json.dump(
                [self._entry_to_dict(e) for e in self.entries],
                f,
                indent=2,
                default=str,
            )

        return log_file

    def _entry_to_dict(self, entry: AuditEntry) -> dict:
        return {
            "timestamp": entry.timestamp,
            "event_type": entry.event_type,
            "user_id": entry.user_id,
            "ip_address": entry.ip_address,
            "action": entry.action,
            "resource": entry.resource,
            "status": entry.status,
            "details": entry.details,
        }


class SecureInferencePipeline:
    """Secure inference pipeline with input validation."""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit_per_minute
        )
        self.audit_logger = AuditLogger()

    def validate_input(
        self,
        file_path: str,
        expected_type: str,
        client_id: str,
    ) -> Dict[str, Any]:
        """
        Validate input for secure inference.

        Args:
            file_path: Path to input file
            expected_type: Expected file type
            client_id: Client identifier

        Returns:
            Validation result
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        if not self.rate_limiter.is_allowed(client_id):
            result["valid"] = False
            result["errors"].append("Rate limit exceeded")
            self.audit_logger.log(
                "security", "rate_limit_exceeded", file_path,
                "blocked", ip_address=client_id,
            )
            return result

        if not os.path.exists(file_path):
            result["valid"] = False
            result["errors"].append("File not found")
            return result

        file_size = os.path.getsize(file_path)
        max_size = self.config.max_upload_size_mb * 1024 * 1024
        if file_size > max_size:
            result["valid"] = False
            result["errors"].append(f"File too large: {file_size} bytes")
            return result

        if not self.sanitizer.validate_file_content(file_path, expected_type):
            result["warnings"].append("File content may not match expected type")

        sanitized_name = self.sanitizer.sanitize_filename(
            os.path.basename(file_path)
        )
        if sanitized_name != os.path.basename(file_path):
            result["warnings"].append("Filename was sanitized")

        return result

    def log_prediction(
        self,
        file_name: str,
        prediction: str,
        confidence: float,
        client_id: Optional[str] = None,
    ) -> None:
        """Log a prediction for audit."""
        self.audit_logger.log_prediction(
            file_name=file_name,
            prediction=prediction,
            confidence=confidence,
            user_id=client_id,
        )


def require_api_key(api_key: str):
    """Decorator to require API key."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from fastapi import Request, HTTPException
            request = kwargs.get("request") or (args[0] if args else None)

            if request and hasattr(request, "headers"):
                provided_key = request.headers.get("x-api-key")
                if not provided_key or provided_key != api_key:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid or missing API key"
                    )

            return func(*args, **kwargs)
        return wrapper
    return decorator


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    """Hash a password with salt."""
    if salt is None:
        salt = secrets.token_hex(16)

    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt.encode(),
        100000,
    )

    return hashed.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify a password against its hash."""
    new_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(new_hash, hashed)
