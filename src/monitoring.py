"""
src/monitoring.py
Production monitoring and health checks for the Deepfake Detection System.

Provides:
- Application health monitoring
- Performance metrics collection
- Resource usage tracking
- Alerting thresholds
- Dashboard data aggregation
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

try:
    import psutil
except ImportError:  # optional dependency - metrics degrade gracefully
    psutil = None

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Application health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    model_loaded: bool
    gpu_available: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_percent: float
    uptime_seconds: float
    total_predictions: int
    avg_inference_time: float
    error_rate: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: str
    requests_per_minute: float
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_count: int
    success_count: int
    active_connections: int


class PerformanceMonitor:
    """Track application performance metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.request_timestamps: deque = deque(maxlen=window_size * 10)
        self.error_count = 0
        self.success_count = 0
        self._lock = threading.Lock()
        self.start_time = time.time()

    def record_request(self, response_time: float, success: bool = True) -> None:
        """Record a request."""
        with self._lock:
            self.response_times.append(response_time)
            self.request_timestamps.append(time.time())

            if success:
                self.success_count += 1
            else:
                self.error_count += 1

    def get_requests_per_minute(self) -> float:
        """Calculate requests per minute."""
        with self._lock:
            now = time.time()
            one_minute_ago = now - 60
            recent = [t for t in self.request_timestamps if t > one_minute_ago]
            return len(recent)

    def get_avg_response_time(self) -> float:
        """Get average response time."""
        with self._lock:
            if not self.response_times:
                return 0.0
            return sum(self.response_times) / len(self.response_times)

    def get_p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        with self._lock:
            if not self.response_times:
                return 0.0
            sorted_times = sorted(self.response_times)
            idx = int(len(sorted_times) * 0.95)
            return sorted_times[min(idx, len(sorted_times) - 1)]

    def get_p99_response_time(self) -> float:
        """Get 99th percentile response time."""
        with self._lock:
            if not self.response_times:
                return 0.0
            sorted_times = sorted(self.response_times)
            idx = int(len(sorted_times) * 0.99)
            return sorted_times[min(idx, len(sorted_times) - 1)]

    def get_error_rate(self) -> float:
        """Get error rate."""
        with self._lock:
            total = self.error_count + self.success_count
            if total == 0:
                return 0.0
            return self.error_count / total

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            requests_per_minute=self.get_requests_per_minute(),
            avg_response_time=self.get_avg_response_time(),
            p95_response_time=self.get_p95_response_time(),
            p99_response_time=self.get_p99_response_time(),
            error_count=self.error_count,
            success_count=self.success_count,
            active_connections=0,
        )

    def reset(self) -> None:
        """Reset metrics."""
        with self._lock:
            self.response_times.clear()
            self.request_timestamps.clear()
            self.error_count = 0
            self.success_count = 0
            self.start_time = time.time()


class ResourceMonitor:
    """Monitor system resources."""

    def __init__(self):
        self.gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_memory_usage(self) -> float:
        """Get memory usage in MB."""
        if psutil is None:
            return 0.0
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        if psutil is None:
            return 0.0
        return psutil.cpu_percent(interval=0.1)

    def get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        if psutil is None:
            return 0.0
        disk = psutil.disk_usage('/')
        return disk.percent

    def get_gpu_usage(self) -> Optional[Dict[str, float]]:
        """Get GPU usage if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "memory_used_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "memory_total_mb": torch.cuda.get_device_properties(0).total_mem / 1024 / 1024,
                }
        except Exception:
            pass
        return None


class HealthChecker:
    """Comprehensive health checking."""

    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.start_time = time.time()
        self.total_predictions = 0
        self.prediction_times: List[float] = []

    def check_health(self, model_loaded: bool = True) -> HealthStatus:
        """
        Check application health.

        Args:
            model_loaded: Whether the model is loaded

        Returns:
            HealthStatus
        """
        memory_mb = self.resource_monitor.get_memory_usage()
        cpu_percent = self.resource_monitor.get_cpu_usage()
        disk_percent = self.resource_monitor.get_disk_usage()

        uptime = time.time() - self.start_time

        avg_inference = (
            sum(self.prediction_times) / len(self.prediction_times)
            if self.prediction_times else 0.0
        )

        error_rate = self.performance_monitor.get_error_rate()

        if not model_loaded:
            status = "unhealthy"
        elif error_rate > 0.1:
            status = "degraded"
        elif memory_mb > 8000:
            status = "degraded"
        elif cpu_percent > 90:
            status = "degraded"
        else:
            status = "healthy"

        return HealthStatus(
            status=status,
            model_loaded=model_loaded,
            gpu_available=self.resource_monitor.gpu_available,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            disk_usage_percent=disk_percent,
            uptime_seconds=uptime,
            total_predictions=self.total_predictions,
            avg_inference_time=avg_inference,
            error_rate=error_rate,
            details={
                "gpu_usage": self.resource_monitor.get_gpu_usage(),
                "performance": {
                    "rpm": self.performance_monitor.get_requests_per_minute(),
                    "avg_response_time": self.performance_monitor.get_avg_response_time(),
                    "p95_response_time": self.performance_monitor.get_p95_response_time(),
                },
            },
        )

    def record_prediction(self, inference_time: float) -> None:
        """Record a prediction."""
        self.total_predictions += 1
        self.prediction_times.append(inference_time)

        if len(self.prediction_times) > 1000:
            self.prediction_times = self.prediction_times[-1000:]


class AlertManager:
    """Manage monitoring alerts."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "memory_mb": 8000,
            "cpu_percent": 90,
            "disk_percent": 90,
            "error_rate": 0.1,
            "response_time_p95": 5.0,
        }
        self.alerts: List[Dict[str, Any]] = []

    def check_alerts(self, health: HealthStatus) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        new_alerts = []

        if health.memory_usage_mb > self.thresholds["memory_mb"]:
            alert = {
                "type": "memory_high",
                "severity": "warning",
                "message": f"Memory usage high: {health.memory_usage_mb:.1f} MB",
                "timestamp": datetime.now().isoformat(),
            }
            new_alerts.append(alert)

        if health.cpu_usage_percent > self.thresholds["cpu_percent"]:
            alert = {
                "type": "cpu_high",
                "severity": "warning",
                "message": f"CPU usage high: {health.cpu_usage_percent:.1f}%",
                "timestamp": datetime.now().isoformat(),
            }
            new_alerts.append(alert)

        if health.error_rate > self.thresholds["error_rate"]:
            alert = {
                "type": "error_rate_high",
                "severity": "critical",
                "message": f"Error rate high: {health.error_rate:.2%}",
                "timestamp": datetime.now().isoformat(),
            }
            new_alerts.append(alert)

        self.alerts.extend(new_alerts)

        return new_alerts

    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [
            a for a in self.alerts
            if datetime.fromisoformat(a["timestamp"]) > cutoff
        ]


# Global instances
_performance_monitor = PerformanceMonitor()
_health_checker = HealthChecker()
_alert_manager = AlertManager()


def get_performance_monitor() -> PerformanceMonitor:
    return _performance_monitor


def get_health_checker() -> HealthChecker:
    return _health_checker


def get_alert_manager() -> AlertManager:
    return _alert_manager
