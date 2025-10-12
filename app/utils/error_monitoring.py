"""
Error monitoring and alerting system
"""
import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorMetric:
    """Error metric for monitoring"""

    error_type: str
    count: int
    first_occurrence: datetime
    last_occurrence: datetime
    severity: str
    service: str
    category: str


@dataclass
class Alert:
    """Alert for error monitoring"""

    alert_id: str
    level: AlertLevel
    title: str
    message: str
    error_type: str
    count: int
    threshold: int
    time_window: int
    service: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["level"] = self.level.value
        data["timestamp"] = self.timestamp.isoformat()
        if self.resolved_at:
            data["resolved_at"] = self.resolved_at.isoformat()
        return data


class ErrorMonitor:
    """Error monitoring and alerting system"""

    def __init__(self, alert_thresholds: Dict[str, int] = None):
        self.alert_thresholds = alert_thresholds or {
            "error": 10,  # 10 errors in 5 minutes
            "warning": 20,  # 20 warnings in 5 minutes
            "critical": 5,  # 5 critical errors in 5 minutes
        }

        # Error tracking
        self.error_counts = defaultdict(lambda: defaultdict(int))
        self.error_history = deque(maxlen=1000)  # Keep last 1000 errors
        self.service_metrics = defaultdict(lambda: defaultdict(int))

        # Alerting
        self.active_alerts = {}
        self.alert_history = deque(maxlen=500)

        # Time windows for monitoring (in minutes)
        self.time_windows = [5, 15, 60, 1440]  # 5min, 15min, 1hr, 24hr

        # Start monitoring task
        self.monitoring_task = None
        self.is_monitoring = False

    def start_monitoring(self):
        """Start error monitoring background task"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Error monitoring started")

    def stop_monitoring(self):
        """Stop error monitoring"""
        if self.is_monitoring and self.monitoring_task:
            self.is_monitoring = False
            self.monitoring_task.cancel()
            logger.info("Error monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                await self._check_alerts()
                await self._cleanup_old_data()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    def record_error(
        self,
        error_type: str,
        severity: str,
        service: str,
        category: str,
        error_id: str = None,
    ):
        """Record an error occurrence"""
        timestamp = datetime.now(timezone.utc)

        # Update counts
        self.error_counts[error_type][timestamp] += 1
        self.service_metrics[service][error_type] += 1

        # Add to history
        error_record = {
            "error_id": error_id,
            "error_type": error_type,
            "severity": severity,
            "service": service,
            "category": category,
            "timestamp": timestamp,
        }
        self.error_history.append(error_record)

        # Check for immediate alerts
        asyncio.create_task(self._check_immediate_alerts(error_type, severity, service))

    async def _check_immediate_alerts(
        self, error_type: str, severity: str, service: str
    ):
        """Check for immediate alert conditions"""
        # Check if this error type has exceeded threshold
        threshold = self.alert_thresholds.get(severity, 10)
        recent_count = self._get_recent_error_count(error_type, 5)  # 5 minutes

        if recent_count >= threshold:
            await self._create_alert(
                error_type=error_type,
                severity=severity,
                service=service,
                count=recent_count,
                threshold=threshold,
                time_window=5,
            )

    async def _check_alerts(self):
        """Check for alert conditions across all error types"""
        for severity, threshold in self.alert_thresholds.items():
            for error_type in self.error_counts:
                for time_window in self.time_windows:
                    recent_count = self._get_recent_error_count(error_type, time_window)

                    if recent_count >= threshold:
                        # Check if alert already exists
                        alert_key = f"{error_type}_{severity}_{time_window}"
                        if alert_key not in self.active_alerts:
                            await self._create_alert(
                                error_type=error_type,
                                severity=severity,
                                service="unknown",  # Will be updated with actual service
                                count=recent_count,
                                threshold=threshold,
                                time_window=time_window,
                            )

    async def _create_alert(
        self,
        error_type: str,
        severity: str,
        service: str,
        count: int,
        threshold: int,
        time_window: int,
    ):
        """Create a new alert"""
        alert_id = (
            f"ALERT_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{error_type}"
        )

        alert = Alert(
            alert_id=alert_id,
            level=AlertLevel(severity.upper()),
            title=f"High {severity} error rate: {error_type}",
            message=f"{error_type} errors exceeded threshold: {count}/{threshold} in {time_window} minutes",
            error_type=error_type,
            count=count,
            threshold=threshold,
            time_window=time_window,
            service=service,
            timestamp=datetime.now(timezone.utc),
        )

        # Store alert
        alert_key = f"{error_type}_{severity}_{time_window}"
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Send alert (implement your alerting mechanism here)
        await self._send_alert(alert)

        logger.warning(f"Alert created: {alert.title}")

    async def _send_alert(self, alert: Alert):
        """Send alert to monitoring system (implement based on your needs)"""
        # This is where you would integrate with your alerting system
        # Examples: Slack, PagerDuty, email, etc.

        alert_data = alert.to_dict()

        # Log the alert
        logger.critical("ALERT: %s - %s", alert.title, alert.message)

        # Here you could send to external systems:
        # - Slack webhook
        # - PagerDuty API
        # - Email service
        # - Custom monitoring dashboard

        # For now, just log the structured alert data
        try:
            formatted = json.dumps(alert_data, indent=2, default=str)
        except Exception:
            formatted = str(alert_data)
        logger.info("Alert data: %s", formatted)

    def _get_recent_error_count(self, error_type: str, minutes: int) -> int:
        """Get error count for specific type in recent time window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        count = 0

        for timestamp, error_count in self.error_counts[error_type].items():
            if timestamp >= cutoff_time:
                count += error_count

        return count

    async def _cleanup_old_data(self):
        """Clean up old error data to prevent memory leaks"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

        # Clean error counts
        for error_type in list(self.error_counts.keys()):
            old_timestamps = [
                ts for ts in self.error_counts[error_type].keys() if ts < cutoff_time
            ]
            for ts in old_timestamps:
                del self.error_counts[error_type][ts]

            # Remove empty error types
            if not self.error_counts[error_type]:
                del self.error_counts[error_type]

    def get_error_metrics(
        self, service: str = None, hours: int = 24
    ) -> List[ErrorMetric]:
        """Get error metrics for monitoring dashboard"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        metrics = []

        for error_record in self.error_history:
            if error_record["timestamp"] >= cutoff_time:
                if service and error_record["service"] != service:
                    continue

                # Find or create metric
                metric_key = f"{error_record['error_type']}_{error_record['service']}"
                existing_metric = next(
                    (m for m in metrics if f"{m.error_type}_{m.service}" == metric_key),
                    None,
                )

                if existing_metric:
                    existing_metric.count += 1
                    existing_metric.last_occurrence = error_record["timestamp"]
                else:
                    metrics.append(
                        ErrorMetric(
                            error_type=error_record["error_type"],
                            count=1,
                            first_occurrence=error_record["timestamp"],
                            last_occurrence=error_record["timestamp"],
                            severity=error_record["severity"],
                            service=error_record["service"],
                            category=error_record["category"],
                        )
                    )

        return sorted(metrics, key=lambda x: x.count, reverse=True)

    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert_key, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                del self.active_alerts[alert_key]
                logger.info(f"Alert resolved: {alert_id}")
                break

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        error_alerts = [a for a in active_alerts if a.level == AlertLevel.ERROR]

        # Determine overall health
        if critical_alerts:
            health_status = "critical"
        elif error_alerts:
            health_status = "degraded"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "error_alerts": len(error_alerts),
            "total_errors_24h": len(
                [
                    e
                    for e in self.error_history
                    if e["timestamp"]
                    >= datetime.now(timezone.utc) - timedelta(hours=24)
                ]
            ),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }


# Global error monitor instance
error_monitor = ErrorMonitor()
