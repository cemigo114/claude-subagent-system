"""
Advanced GPU Cluster Manager

A production-ready distributed GPU cluster management system providing:
- Direct hardware control with <100¼s scheduling latency
- Multi-vendor GPU support (NVIDIA, AMD, Intel)
- Distributed consensus with automatic failover
- Real-time monitoring and multi-tenant isolation
"""

__version__ = "1.0.0"
__author__ = "GPU Cluster Manager Team"

from .core.cluster_manager import ClusterManager
from .core.node_manager import NodeManager
from .hardware.gpu_manager import GPUManager
from .scheduling.real_time_scheduler import RealTimeScheduler

__all__ = [
    "ClusterManager",
    "NodeManager", 
    "GPUManager",
    "RealTimeScheduler"
]