"""
Structured logging configuration with correlation IDs and performance metrics.
"""

import logging
import sys
import time
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

import structlog
from structlog.stdlib import LoggerFactory

from ..config.settings import settings


def setup_logging() -> None:
    """Setup structured logging configuration."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class CorrelationIDFilter(logging.Filter):
    """Filter to add correlation IDs to log records."""
    
    def __init__(self):
        super().__init__()
        self._correlation_id: Optional[str] = None
    
    def set_correlation_id(self, correlation_id: str):
        """Set the correlation ID for this filter."""
        self._correlation_id = correlation_id
    
    def filter(self, record):
        """Add correlation ID to log record."""
        record.correlation_id = self._correlation_id or "unknown"
        return True


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)
        self._start_times: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str, **context) -> str:
        """Start timing an operation."""
        operation_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        self._start_times[operation_id] = time.time()
        
        self.logger.info(
            "Operation started",
            operation_id=operation_id,
            operation_name=operation_name,
            **context
        )
        
        return operation_id
    
    def end_operation(
        self, 
        operation_id: str, 
        success: bool = True,
        error: Optional[str] = None,
        **context
    ):
        """End timing an operation."""
        if operation_id not in self._start_times:
            self.logger.warning("Operation ID not found", operation_id=operation_id)
            return
        
        duration = time.time() - self._start_times[operation_id]
        del self._start_times[operation_id]
        
        log_data = {
            "operation_id": operation_id,
            "duration_seconds": duration,
            "success": success,
            **context
        }
        
        if error:
            log_data["error"] = error
        
        if success:
            self.logger.info("Operation completed", **log_data)
        else:
            self.logger.error("Operation failed", **log_data)
    
    @contextmanager
    def operation_timer(self, operation_name: str, **context):
        """Context manager for timing operations."""
        operation_id = self.start_operation(operation_name, **context)
        
        try:
            yield operation_id
            self.end_operation(operation_id, success=True)
        except Exception as e:
            self.end_operation(operation_id, success=False, error=str(e))
            raise


class AuditLogger:
    """Logger for audit trails and compliance."""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(f"{logger_name}.audit")
    
    def log_agent_created(
        self,
        agent_type: str,
        project_id: str,
        user_id: Optional[str] = None,
        **context
    ):
        """Log agent creation event."""
        self.logger.info(
            "Agent created",
            event_type="agent_created",
            agent_type=agent_type,
            project_id=project_id,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            **context
        )
    
    def log_agent_executed(
        self,
        agent_type: str,
        project_id: str,
        task_description: str,
        success: bool,
        duration_seconds: float,
        user_id: Optional[str] = None,
        **context
    ):
        """Log agent execution event."""
        self.logger.info(
            "Agent executed",
            event_type="agent_executed",
            agent_type=agent_type,
            project_id=project_id,
            task_description=task_description[:100],  # Truncate for security
            success=success,
            duration_seconds=duration_seconds,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            **context
        )
    
    def log_context_handoff(
        self,
        source_agent: str,
        target_agent: str,
        project_id: str,
        handoff_id: str,
        quality_score: float,
        user_id: Optional[str] = None,
        **context
    ):
        """Log context handoff event."""
        self.logger.info(
            "Context handoff",
            event_type="context_handoff",
            source_agent=source_agent,
            target_agent=target_agent,
            project_id=project_id,
            handoff_id=handoff_id,
            quality_score=quality_score,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            **context
        )
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        **context
    ):
        """Log security-related events."""
        self.logger.warning(
            "Security event",
            event_type=event_type,
            severity=severity,
            description=description,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            **context
        )


class MetricsCollector:
    """Collect and log performance metrics."""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(f"{logger_name}.metrics")
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
    
    def increment_counter(self, metric_name: str, value: int = 1, **labels):
        """Increment a counter metric."""
        key = f"{metric_name}:{':'.join(f'{k}={v}' for k, v in labels.items())}"
        self._counters[key] = self._counters.get(key, 0) + value
        
        self.logger.debug(
            "Counter incremented",
            metric_name=metric_name,
            value=self._counters[key],
            increment=value,
            labels=labels
        )
    
    def set_gauge(self, metric_name: str, value: float, **labels):
        """Set a gauge metric value."""
        key = f"{metric_name}:{':'.join(f'{k}={v}' for k, v in labels.items())}"
        self._gauges[key] = value
        
        self.logger.debug(
            "Gauge set",
            metric_name=metric_name,
            value=value,
            labels=labels
        )
    
    def record_histogram(self, metric_name: str, value: float, **labels):
        """Record a histogram metric (implemented as gauge for now)."""
        self.set_gauge(f"{metric_name}_histogram", value, **labels)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        return {
            "counters": self._counters.copy(),
            "gauges": self._gauges.copy(),
            "timestamp": datetime.now().isoformat()
        }


# Global instances
_performance_logger: Optional[PerformanceLogger] = None
_audit_logger: Optional[AuditLogger] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_performance_logger(logger_name: str = "subagent.performance") -> PerformanceLogger:
    """Get global performance logger instance."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger(logger_name)
    return _performance_logger


def get_audit_logger(logger_name: str = "subagent.audit") -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(logger_name)
    return _audit_logger


def get_metrics_collector(logger_name: str = "subagent.metrics") -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(logger_name)
    return _metrics_collector