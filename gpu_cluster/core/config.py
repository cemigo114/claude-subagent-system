"""
GPU Cluster Configuration Management

This module provides comprehensive configuration management for the GPU cluster system,
including environment variable loading, performance tuning parameters, and vendor-specific
GPU detection and enumeration settings.

Key features:
- Pydantic settings with validation
- Performance tuning parameters for <100μs scheduling latency
- Multi-vendor GPU support configuration
- Distributed system settings (Raft, RDMA)
- Security and multi-tenancy configuration
"""

import os
from typing import List, Dict, Optional, Union
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class GPUVendor(str, Enum):
    """Supported GPU vendors"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"


class SchedulingPolicy(str, Enum):
    """Available scheduling policies"""
    FIFO = "fifo"
    PRIORITY = "priority"
    AFFINITY = "affinity"
    GANG = "gang"
    REAL_TIME = "real_time"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PerformanceConfig(BaseModel):
    """Performance-critical configuration parameters"""
    
    # Scheduling performance requirements
    max_scheduling_latency_us: int = Field(
        default=100,
        description="Maximum scheduling decision latency in microseconds"
    )
    
    memory_allocation_latency_us: int = Field(
        default=10,
        description="Maximum memory allocation latency in microseconds"
    )
    
    target_gpu_utilization_percent: float = Field(
        default=95.0,
        ge=0.0,
        le=100.0,
        description="Target GPU utilization percentage"
    )
    
    # Memory pool configuration
    memory_pool_size_bytes: int = Field(
        default=8 * 1024 * 1024 * 1024,  # 8GB
        description="Size of GPU memory pool per device"
    )
    
    memory_alignment_bytes: int = Field(
        default=256,
        description="Memory alignment requirement (vendor-specific)"
    )
    
    # Queue and buffer sizes
    task_queue_capacity: int = Field(
        default=10000,
        description="Maximum number of tasks in ready queue"
    )
    
    completion_queue_capacity: int = Field(
        default=1000,
        description="Maximum number of completion notifications"
    )
    
    # Lock-free data structure parameters
    atomic_retry_count: int = Field(
        default=1000,
        description="Maximum retries for atomic operations"
    )
    
    atomic_backoff_ns: int = Field(
        default=100,
        description="Backoff time in nanoseconds for atomic operations"
    )


class HardwareConfig(BaseModel):
    """Hardware-specific configuration"""
    
    # Multi-vendor GPU support
    enabled_vendors: List[GPUVendor] = Field(
        default=[GPUVendor.NVIDIA, GPUVendor.AMD, GPUVendor.INTEL],
        description="List of enabled GPU vendors"
    )
    
    # NVIDIA-specific settings
    nvidia_driver_path: str = Field(
        default="/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
        description="Path to NVIDIA Management Library"
    )
    
    cuda_driver_path: str = Field(
        default="/usr/local/cuda/lib64/libcuda.so.1",
        description="Path to CUDA Driver library"
    )
    
    # AMD-specific settings
    rocm_smi_path: str = Field(
        default="/opt/rocm/lib/librocm_smi64.so.1",
        description="Path to ROCm SMI library"
    )
    
    # Intel-specific settings
    intel_gpu_driver_path: str = Field(
        default="/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
        description="Path to Intel GPU driver library"
    )
    
    # Thermal management
    thermal_threshold_celsius: float = Field(
        default=85.0,
        description="Temperature threshold for thermal management"
    )
    
    power_limit_watts: Optional[float] = Field(
        default=None,
        description="Power limit per GPU in watts"
    )
    
    # NUMA configuration
    numa_aware_allocation: bool = Field(
        default=True,
        description="Enable NUMA-aware memory allocation"
    )


class DistributedConfig(BaseModel):
    """Distributed system configuration"""
    
    # Raft consensus parameters
    election_timeout_ms: int = Field(
        default=150,
        description="Raft election timeout in milliseconds"
    )
    
    heartbeat_interval_ms: int = Field(
        default=50,
        description="Raft heartbeat interval in milliseconds"
    )
    
    log_replication_batch_size: int = Field(
        default=100,
        description="Number of log entries to replicate in a batch"
    )
    
    snapshot_threshold_entries: int = Field(
        default=10000,
        description="Number of log entries before triggering snapshot"
    )
    
    # Gossip protocol settings
    gossip_interval_ms: int = Field(
        default=1000,
        description="Gossip protocol interval in milliseconds"
    )
    
    failure_detection_timeout_ms: int = Field(
        default=5000,
        description="Failure detection timeout in milliseconds"
    )
    
    # Network configuration
    cluster_port: int = Field(
        default=8080,
        description="Port for cluster communication"
    )
    
    max_connections: int = Field(
        default=1000,
        description="Maximum number of concurrent connections"
    )


class NetworkingConfig(BaseModel):
    """High-performance networking configuration"""
    
    # RDMA settings
    enable_rdma: bool = Field(
        default=True,
        description="Enable RDMA transport when available"
    )
    
    rdma_device_name: str = Field(
        default="mlx5_0",
        description="RDMA device name (InfiniBand/RoCE)"
    )
    
    rdma_port: int = Field(
        default=1,
        description="RDMA port number"
    )
    
    rdma_queue_depth: int = Field(
        default=1024,
        description="RDMA queue pair depth"
    )
    
    rdma_completion_queue_size: int = Field(
        default=2048,
        description="RDMA completion queue size"
    )
    
    # Message protocol settings
    message_compression: bool = Field(
        default=True,
        description="Enable message compression for large payloads"
    )
    
    max_message_size_bytes: int = Field(
        default=64 * 1024 * 1024,  # 64MB
        description="Maximum message size in bytes"
    )


class SecurityConfig(BaseModel):
    """Security and multi-tenancy configuration"""
    
    # Multi-tenant isolation
    enable_multi_tenancy: bool = Field(
        default=True,
        description="Enable multi-tenant resource isolation"
    )
    
    max_tenants: int = Field(
        default=100,
        description="Maximum number of tenants supported"
    )
    
    # Resource quotas
    default_memory_quota_gb: float = Field(
        default=8.0,
        description="Default memory quota per tenant in GB"
    )
    
    default_compute_quota_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Default compute quota per tenant as percentage"
    )
    
    # Authentication and authorization
    enable_authentication: bool = Field(
        default=True,
        description="Enable authentication for API access"
    )
    
    jwt_secret_key: str = Field(
        default="",
        description="JWT secret key for authentication"
    )
    
    jwt_expiration_hours: int = Field(
        default=24,
        description="JWT token expiration time in hours"
    )
    
    # Audit logging
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging for compliance"
    )
    
    audit_log_path: Path = Field(
        default=Path("/var/log/gpu-cluster/audit.log"),
        description="Path to audit log file"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration"""
    
    # Metrics collection
    metrics_collection_interval_ms: int = Field(
        default=1000,
        description="Metrics collection interval in milliseconds"
    )
    
    metrics_retention_hours: int = Field(
        default=168,  # 1 week
        description="Metrics retention time in hours"
    )
    
    # Prometheus integration
    prometheus_port: int = Field(
        default=9090,
        description="Prometheus metrics export port"
    )
    
    prometheus_endpoint: str = Field(
        default="/metrics",
        description="Prometheus metrics endpoint path"
    )
    
    # Performance monitoring
    enable_kernel_tracing: bool = Field(
        default=True,
        description="Enable CUDA kernel execution tracing"
    )
    
    trace_buffer_size_mb: int = Field(
        default=256,
        description="Kernel trace buffer size in MB"
    )
    
    # Alerting
    enable_alerting: bool = Field(
        default=True,
        description="Enable performance and health alerting"
    )
    
    alert_latency_threshold_us: int = Field(
        default=200,
        description="Alert threshold for scheduling latency in microseconds"
    )


class GPUClusterConfig(BaseSettings):
    """
    Main configuration class for GPU cluster system.
    
    This class combines all configuration sections and provides environment
    variable loading with validation. Configuration can be loaded from:
    1. Environment variables (prefixed with GPU_CLUSTER_)
    2. Configuration files (YAML/JSON)
    3. Command line arguments
    """
    
    # Basic cluster identification
    cluster_name: str = Field(
        default="gpu-cluster-01",
        description="Unique cluster identifier"
    )
    
    node_id: str = Field(
        default="",
        description="Unique node identifier (auto-generated if empty)"
    )
    
    # Logging configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level for the cluster"
    )
    
    log_file_path: Optional[Path] = Field(
        default=None,
        description="Path to log file (stdout if not specified)"
    )
    
    # Default scheduling policy
    default_scheduling_policy: SchedulingPolicy = Field(
        default=SchedulingPolicy.REAL_TIME,
        description="Default scheduling policy for new tasks"
    )
    
    # Configuration sections
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance-related configuration"
    )
    
    hardware: HardwareConfig = Field(
        default_factory=HardwareConfig,
        description="Hardware-specific configuration"
    )
    
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Distributed system configuration"
    )
    
    networking: NetworkingConfig = Field(
        default_factory=NetworkingConfig,
        description="High-performance networking configuration"
    )
    
    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security and multi-tenancy configuration"
    )
    
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring and observability configuration"
    )
    
    class Config:
        """Pydantic configuration"""
        env_prefix = "GPU_CLUSTER_"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    @validator('node_id', always=True)
    def generate_node_id(cls, v):
        """Generate unique node ID if not provided"""
        if not v:
            import socket
            import uuid
            hostname = socket.gethostname()
            unique_id = str(uuid.uuid4())[:8]
            return f"{hostname}-{unique_id}"
        return v
    
    @validator('security')
    def validate_security_config(cls, v):
        """Validate security configuration"""
        if v.enable_authentication and not v.jwt_secret_key:
            import secrets
            v.jwt_secret_key = secrets.token_urlsafe(32)
        return v
    
    def validate_hardware_paths(self) -> Dict[str, bool]:
        """
        Validate that required hardware libraries are available.
        
        Returns:
            Dict mapping vendor to availability status
        """
        availability = {}
        
        for vendor in self.hardware.enabled_vendors:
            if vendor == GPUVendor.NVIDIA:
                nvidia_available = (
                    os.path.exists(self.hardware.nvidia_driver_path) and
                    os.path.exists(self.hardware.cuda_driver_path)
                )
                availability[vendor.value] = nvidia_available
                
            elif vendor == GPUVendor.AMD:
                amd_available = os.path.exists(self.hardware.rocm_smi_path)
                availability[vendor.value] = amd_available
                
            elif vendor == GPUVendor.INTEL:
                intel_available = os.path.exists(self.hardware.intel_gpu_driver_path)
                availability[vendor.value] = intel_available
        
        return availability
    
    def get_performance_requirements(self) -> Dict[str, Union[int, float]]:
        """
        Get performance requirements for validation.
        
        Returns:
            Dictionary of performance requirements with their target values
        """
        return {
            "max_scheduling_latency_us": self.performance.max_scheduling_latency_us,
            "memory_allocation_latency_us": self.performance.memory_allocation_latency_us,
            "target_gpu_utilization_percent": self.performance.target_gpu_utilization_percent,
            "system_overhead_percent": 5.0,  # <5% system overhead requirement
        }
    
    @classmethod
    def load_from_file(cls, config_path: Union[str, Path]) -> "GPUClusterConfig":
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured GPUClusterConfig instance
        """
        import yaml
        import json
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
            format: File format ('yaml' or 'json')
        """
        import yaml
        import json
        
        config_path = Path(config_path)
        config_data = self.dict()
        
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")


# Global configuration instance
config: Optional[GPUClusterConfig] = None


def get_config() -> GPUClusterConfig:
    """
    Get global configuration instance.
    
    Returns:
        Global GPUClusterConfig instance
    """
    global config
    if config is None:
        config = GPUClusterConfig()
    return config


def set_config(new_config: GPUClusterConfig) -> None:
    """
    Set global configuration instance.
    
    Args:
        new_config: New configuration instance to use globally
    """
    global config
    config = new_config


def load_config_from_file(config_path: Union[str, Path]) -> GPUClusterConfig:
    """
    Load configuration from file and set as global instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration instance
    """
    global config
    config = GPUClusterConfig.load_from_file(config_path)
    return config