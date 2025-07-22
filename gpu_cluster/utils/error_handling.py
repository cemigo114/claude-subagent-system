"""
GPU Cluster Error Handling

This module provides a comprehensive exception hierarchy and error handling system
for the GPU cluster management system. It includes GPU vendor-specific error mapping,
performance-critical error paths, and recovery strategies for different failure modes.

Key features:
- Centralized exception hierarchy
- GPU vendor-specific error code mapping  
- Performance-critical error paths (no logging in hot path)
- Recovery strategies and error categorization
- Consistent error reporting across all components
"""

from typing import Dict, Optional, Any, Union, List, Tuple
from enum import Enum
import traceback
import time
from dataclasses import dataclass


class ErrorSeverity(str, Enum):
    """Error severity levels for categorization and handling"""
    LOW = "low"           # Minor issues, system continues normally
    MEDIUM = "medium"     # Degraded performance, requires attention
    HIGH = "high"         # System functionality impaired
    CRITICAL = "critical" # System failure, immediate action required


class ErrorCategory(str, Enum):
    """Error categories for systematic handling"""
    HARDWARE = "hardware"           # GPU/device-related errors
    SCHEDULING = "scheduling"       # Task scheduling and resource allocation
    NETWORKING = "networking"       # Network and communication errors
    DISTRIBUTED = "distributed"    # Raft consensus and cluster management
    SECURITY = "security"          # Authentication and authorization
    PERFORMANCE = "performance"     # Performance requirement violations
    CONFIGURATION = "configuration" # Configuration and setup errors
    SYSTEM = "system"              # General system errors


@dataclass
class ErrorContext:
    """Additional context information for error analysis and debugging"""
    timestamp: float
    node_id: Optional[str] = None
    gpu_id: Optional[str] = None
    task_id: Optional[str] = None
    tenant_id: Optional[str] = None
    operation: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create_current(
        cls, 
        node_id: Optional[str] = None,
        gpu_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ) -> "ErrorContext":
        """Create error context with current timestamp"""
        return cls(
            timestamp=time.time(),
            node_id=node_id,
            gpu_id=gpu_id,
            task_id=task_id,
            tenant_id=tenant_id,
            operation=operation,
            additional_info=kwargs if kwargs else None
        )


class GPUClusterError(Exception):
    """
    Base exception class for all GPU cluster errors.
    
    This base class provides common functionality for error categorization,
    severity assessment, and context tracking across all cluster components.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        recoverable: bool = True,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.error_code = error_code
        self.context = context or ErrorContext.create_current()
        self.recoverable = recoverable
        self.cause = cause
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "error_code": self.error_code,
            "recoverable": self.recoverable,
            "timestamp": self.context.timestamp,
            "context": {
                "node_id": self.context.node_id,
                "gpu_id": self.context.gpu_id,
                "task_id": self.context.task_id,
                "tenant_id": self.context.tenant_id,
                "operation": self.context.operation,
                "additional_info": self.context.additional_info
            }
        }


# Hardware-related exceptions

class HardwareError(GPUClusterError):
    """Base class for hardware-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.HARDWARE,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            **kwargs
        )


class GPUNotFoundError(HardwareError):
    """Raised when a specified GPU device cannot be found"""
    
    def __init__(self, gpu_id: str, **kwargs):
        super().__init__(
            f"GPU device not found: {gpu_id}",
            error_code="GPU_NOT_FOUND",
            **kwargs
        )
        self.gpu_id = gpu_id


class GPUInitializationError(HardwareError):
    """Raised when GPU initialization fails"""
    
    def __init__(self, gpu_id: str, vendor: str, reason: str, **kwargs):
        super().__init__(
            f"GPU initialization failed for {vendor} device {gpu_id}: {reason}",
            error_code="GPU_INIT_FAILED",
            recoverable=False,
            **kwargs
        )
        self.gpu_id = gpu_id
        self.vendor = vendor
        self.reason = reason


class GPUMemoryError(HardwareError):
    """Raised for GPU memory allocation/deallocation errors"""
    
    def __init__(self, gpu_id: str, operation: str, size_bytes: int, available_bytes: int, **kwargs):
        super().__init__(
            f"GPU memory {operation} failed on {gpu_id}: "
            f"requested {size_bytes} bytes, available {available_bytes} bytes",
            error_code="GPU_MEMORY_ERROR",
            **kwargs
        )
        self.gpu_id = gpu_id
        self.operation = operation
        self.size_bytes = size_bytes
        self.available_bytes = available_bytes


class ThermalThrottleError(HardwareError):
    """Raised when GPU thermal throttling occurs"""
    
    def __init__(self, gpu_id: str, temperature: float, threshold: float, **kwargs):
        super().__init__(
            f"GPU {gpu_id} thermal throttling: {temperature}°C exceeds threshold {threshold}°C",
            error_code="THERMAL_THROTTLE",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.gpu_id = gpu_id
        self.temperature = temperature
        self.threshold = threshold


# Vendor-specific error mapping

class NVMLError(HardwareError):
    """NVIDIA NVML-specific errors"""
    
    # NVML return codes mapping
    NVML_ERROR_CODES = {
        1: ("NVML_ERROR_UNINITIALIZED", "NVML was not first initialized"),
        2: ("NVML_ERROR_INVALID_ARGUMENT", "Invalid argument passed"),
        3: ("NVML_ERROR_NOT_SUPPORTED", "Operation not supported"),
        4: ("NVML_ERROR_NO_PERMISSION", "Insufficient permissions"),
        5: ("NVML_ERROR_ALREADY_INITIALIZED", "NVML already initialized"),
        6: ("NVML_ERROR_NOT_FOUND", "Device or resource not found"),
        7: ("NVML_ERROR_INSUFFICIENT_SIZE", "Buffer too small"),
        8: ("NVML_ERROR_INSUFFICIENT_POWER", "Insufficient power"),
        9: ("NVML_ERROR_DRIVER_NOT_LOADED", "NVIDIA driver not loaded"),
        10: ("NVML_ERROR_TIMEOUT", "Operation timeout"),
        11: ("NVML_ERROR_IRQ_ISSUE", "Interrupt request issue"),
        12: ("NVML_ERROR_LIBRARY_NOT_FOUND", "NVML library not found"),
        13: ("NVML_ERROR_FUNCTION_NOT_FOUND", "Function not found in library"),
        14: ("NVML_ERROR_CORRUPTED_INFOROM", "Corrupted infoROM"),
        15: ("NVML_ERROR_GPU_IS_LOST", "GPU is lost"),
        999: ("NVML_ERROR_UNKNOWN", "Unknown error")
    }
    
    def __init__(self, nvml_code: int, gpu_id: Optional[str] = None, **kwargs):
        error_name, error_desc = self.NVML_ERROR_CODES.get(
            nvml_code, 
            ("NVML_ERROR_UNKNOWN", f"Unknown NVML error code: {nvml_code}")
        )
        
        super().__init__(
            f"NVML Error: {error_desc}",
            error_code=error_name,
            severity=self._get_severity_from_code(nvml_code),
            recoverable=self._is_recoverable(nvml_code),
            **kwargs
        )
        self.nvml_code = nvml_code
        self.gpu_id = gpu_id
    
    @staticmethod
    def _get_severity_from_code(code: int) -> ErrorSeverity:
        """Map NVML error code to severity"""
        critical_codes = {15}  # GPU_IS_LOST
        high_codes = {9, 12, 13, 14}  # Driver issues, library problems
        return (ErrorSeverity.CRITICAL if code in critical_codes else
                ErrorSeverity.HIGH if code in high_codes else
                ErrorSeverity.MEDIUM)
    
    @staticmethod
    def _is_recoverable(code: int) -> bool:
        """Determine if NVML error is recoverable"""
        non_recoverable_codes = {9, 12, 13, 14, 15}  # System-level issues
        return code not in non_recoverable_codes


class ROCmError(HardwareError):
    """AMD ROCm-specific errors"""
    
    # ROCm SMI return codes mapping
    ROCM_ERROR_CODES = {
        0: ("RSMI_STATUS_SUCCESS", "Success"),
        1: ("RSMI_STATUS_INVALID_ARGS", "Invalid arguments"),
        2: ("RSMI_STATUS_NOT_SUPPORTED", "Operation not supported"),
        3: ("RSMI_STATUS_FILE_ERROR", "File operation error"),
        4: ("RSMI_STATUS_PERMISSION", "Permission error"),
        5: ("RSMI_STATUS_OUT_OF_RESOURCES", "Out of resources"),
        6: ("RSMI_STATUS_INTERNAL_EXCEPTION", "Internal exception"),
        7: ("RSMI_STATUS_INPUT_OUT_OF_BOUNDS", "Input out of bounds"),
        8: ("RSMI_STATUS_INIT_ERROR", "Initialization error"),
        0xFFFFFFFF: ("RSMI_STATUS_UNKNOWN_ERROR", "Unknown error")
    }
    
    def __init__(self, rocm_code: int, gpu_id: Optional[str] = None, **kwargs):
        error_name, error_desc = self.ROCM_ERROR_CODES.get(
            rocm_code,
            ("RSMI_STATUS_UNKNOWN", f"Unknown ROCm error code: {rocm_code}")
        )
        
        super().__init__(
            f"ROCm Error: {error_desc}",
            error_code=error_name,
            severity=self._get_severity_from_code(rocm_code),
            **kwargs
        )
        self.rocm_code = rocm_code
        self.gpu_id = gpu_id
    
    @staticmethod
    def _get_severity_from_code(code: int) -> ErrorSeverity:
        """Map ROCm error code to severity"""
        high_codes = {4, 5, 6, 8}  # Permission, resources, internal, init errors
        return ErrorSeverity.HIGH if code in high_codes else ErrorSeverity.MEDIUM


# Scheduling-related exceptions

class SchedulingError(GPUClusterError):
    """Base class for scheduling-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SCHEDULING,
            **kwargs
        )


class SchedulingLatencyViolation(SchedulingError):
    """Raised when scheduling latency exceeds requirements"""
    
    def __init__(self, actual_latency_us: float, max_latency_us: float, **kwargs):
        super().__init__(
            f"Scheduling latency violation: {actual_latency_us}μs exceeds limit {max_latency_us}μs",
            error_code="SCHEDULING_LATENCY_VIOLATION",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.actual_latency_us = actual_latency_us
        self.max_latency_us = max_latency_us


class ResourceExhaustedError(SchedulingError):
    """Raised when cluster resources are exhausted"""
    
    def __init__(self, resource_type: str, requested: int, available: int, **kwargs):
        super().__init__(
            f"Resource exhausted: {resource_type} requested={requested}, available={available}",
            error_code="RESOURCE_EXHAUSTED",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.resource_type = resource_type
        self.requested = requested
        self.available = available


class TaskTimeoutError(SchedulingError):
    """Raised when task execution exceeds timeout"""
    
    def __init__(self, task_id: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Task {task_id} exceeded timeout of {timeout_seconds} seconds",
            error_code="TASK_TIMEOUT",
            **kwargs
        )
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds


# Distributed system exceptions

class DistributedSystemError(GPUClusterError):
    """Base class for distributed system errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DISTRIBUTED,
            **kwargs
        )


class RaftConsensusError(DistributedSystemError):
    """Raft consensus-related errors"""
    
    def __init__(self, message: str, term: int, node_id: str, **kwargs):
        super().__init__(
            f"Raft consensus error (term={term}, node={node_id}): {message}",
            error_code="RAFT_CONSENSUS_ERROR",
            **kwargs
        )
        self.term = term
        self.node_id = node_id


class SplitBrainError(DistributedSystemError):
    """Raised when split-brain scenario is detected"""
    
    def __init__(self, leader_nodes: List[str], **kwargs):
        super().__init__(
            f"Split-brain detected: multiple leaders {leader_nodes}",
            error_code="SPLIT_BRAIN",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )
        self.leader_nodes = leader_nodes


class NodeFailureError(DistributedSystemError):
    """Raised when cluster node failure is detected"""
    
    def __init__(self, failed_node_id: str, last_heartbeat: float, **kwargs):
        super().__init__(
            f"Node failure detected: {failed_node_id}, last heartbeat {time.time() - last_heartbeat:.1f}s ago",
            error_code="NODE_FAILURE",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.failed_node_id = failed_node_id
        self.last_heartbeat = last_heartbeat


# Networking exceptions

class NetworkingError(GPUClusterError):
    """Base class for networking errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORKING,
            **kwargs
        )


class RDMAError(NetworkingError):
    """RDMA transport errors"""
    
    def __init__(self, operation: str, error_details: str, **kwargs):
        super().__init__(
            f"RDMA {operation} error: {error_details}",
            error_code="RDMA_ERROR",
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.operation = operation


class ConnectionTimeoutError(NetworkingError):
    """Network connection timeout errors"""
    
    def __init__(self, target: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Connection timeout to {target} after {timeout_seconds} seconds",
            error_code="CONNECTION_TIMEOUT",
            **kwargs
        )
        self.target = target
        self.timeout_seconds = timeout_seconds


# Security and multi-tenancy exceptions

class SecurityError(GPUClusterError):
    """Base class for security-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=kwargs.get('severity', ErrorSeverity.HIGH),
            **kwargs
        )


class AuthenticationError(SecurityError):
    """Authentication failure errors"""
    
    def __init__(self, user_id: str, reason: str, **kwargs):
        super().__init__(
            f"Authentication failed for user {user_id}: {reason}",
            error_code="AUTHENTICATION_FAILED",
            **kwargs
        )
        self.user_id = user_id
        self.reason = reason


class QuotaExceededError(SecurityError):
    """Resource quota exceeded errors"""
    
    def __init__(self, tenant_id: str, resource_type: str, requested: int, limit: int, **kwargs):
        super().__init__(
            f"Quota exceeded for tenant {tenant_id}: {resource_type} requested={requested}, limit={limit}",
            error_code="QUOTA_EXCEEDED",
            **kwargs
        )
        self.tenant_id = tenant_id
        self.resource_type = resource_type
        self.requested = requested
        self.limit = limit


class TenantIsolationViolation(SecurityError):
    """Tenant isolation violation errors"""
    
    def __init__(self, tenant_id: str, violation_type: str, details: str, **kwargs):
        super().__init__(
            f"Tenant isolation violation for {tenant_id}: {violation_type} - {details}",
            error_code="ISOLATION_VIOLATION",
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.tenant_id = tenant_id
        self.violation_type = violation_type
        self.details = details


# Performance exceptions

class PerformanceError(GPUClusterError):
    """Base class for performance-related errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PERFORMANCE,
            **kwargs
        )


class UtilizationThresholdError(PerformanceError):
    """GPU utilization below threshold"""
    
    def __init__(self, gpu_id: str, actual_utilization: float, threshold: float, **kwargs):
        super().__init__(
            f"GPU {gpu_id} utilization {actual_utilization:.1f}% below threshold {threshold:.1f}%",
            error_code="UTILIZATION_LOW",
            **kwargs
        )
        self.gpu_id = gpu_id
        self.actual_utilization = actual_utilization
        self.threshold = threshold


# Error handling utilities

class ErrorHandler:
    """
    Centralized error handling and recovery system.
    
    Provides error categorization, recovery strategy selection, and
    performance-optimized error reporting for critical paths.
    """
    
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, callable] = {}
        
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> bool:
        """
        Handle error with appropriate recovery strategy.
        
        Args:
            error: Exception to handle
            context: Additional context for error handling
            
        Returns:
            True if error was recovered, False if not recoverable
        """
        if isinstance(error, GPUClusterError):
            cluster_error = error
        else:
            # Wrap non-cluster errors
            cluster_error = GPUClusterError(
                str(error),
                context=context,
                cause=error
            )
        
        # Update error counts for monitoring
        error_key = f"{cluster_error.category.value}:{cluster_error.__class__.__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error if not in performance-critical path
        if self.enable_logging and cluster_error.severity != ErrorSeverity.LOW:
            self._log_error(cluster_error)
        
        # Attempt recovery if error is recoverable
        if cluster_error.recoverable:
            return self._attempt_recovery(cluster_error)
        
        return False
    
    def register_recovery_strategy(self, error_code: str, strategy: callable) -> None:
        """Register custom recovery strategy for specific error codes"""
        self.recovery_strategies[error_code] = strategy
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring and analysis"""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "error_types": len(self.error_counts),
            "registered_strategies": len(self.recovery_strategies)
        }
    
    def _log_error(self, error: GPUClusterError) -> None:
        """Log error with appropriate detail level"""
        # In production, this would use proper logging infrastructure
        # For now, we avoid logging to maintain performance in hot paths
        pass
    
    def _attempt_recovery(self, error: GPUClusterError) -> bool:
        """Attempt error recovery using registered strategies"""
        if error.error_code and error.error_code in self.recovery_strategies:
            try:
                strategy = self.recovery_strategies[error.error_code]
                return strategy(error)
            except Exception:
                return False
        
        # Default recovery strategies based on category
        if error.category == ErrorCategory.HARDWARE:
            return self._recover_hardware_error(error)
        elif error.category == ErrorCategory.NETWORKING:
            return self._recover_network_error(error)
        elif error.category == ErrorCategory.SCHEDULING:
            return self._recover_scheduling_error(error)
        
        return False
    
    def _recover_hardware_error(self, error: HardwareError) -> bool:
        """Default hardware error recovery"""
        if isinstance(error, ThermalThrottleError):
            # Reduce GPU workload temporarily
            return True
        elif isinstance(error, GPUMemoryError):
            # Trigger garbage collection and retry
            return True
        return False
    
    def _recover_network_error(self, error: NetworkingError) -> bool:
        """Default network error recovery"""
        if isinstance(error, ConnectionTimeoutError):
            # Retry with exponential backoff
            return True
        return False
    
    def _recover_scheduling_error(self, error: SchedulingError) -> bool:
        """Default scheduling error recovery"""
        if isinstance(error, ResourceExhaustedError):
            # Queue task for later execution
            return True
        return False


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(error: Exception, context: Optional[ErrorContext] = None) -> bool:
    """Global error handling function"""
    return global_error_handler.handle_error(error, context)


def create_error_context(**kwargs) -> ErrorContext:
    """Convenience function to create error context"""
    return ErrorContext.create_current(**kwargs)


def map_vendor_error(vendor: str, error_code: int, gpu_id: Optional[str] = None) -> HardwareError:
    """
    Map vendor-specific error codes to GPU cluster exceptions.
    
    Args:
        vendor: GPU vendor name
        error_code: Vendor-specific error code
        gpu_id: Optional GPU identifier
        
    Returns:
        Appropriate HardwareError subclass
    """
    vendor_lower = vendor.lower()
    
    if vendor_lower == "nvidia":
        return NVMLError(error_code, gpu_id=gpu_id)
    elif vendor_lower == "amd":
        return ROCmError(error_code, gpu_id=gpu_id)
    else:
        return HardwareError(
            f"Unknown error from {vendor} GPU {gpu_id}: code {error_code}",
            error_code=f"{vendor_upper}_ERROR_{error_code}",
            context=ErrorContext.create_current(gpu_id=gpu_id)
        )