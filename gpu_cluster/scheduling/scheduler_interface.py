"""
GPU Cluster Scheduler Interface

This module defines the abstract base class and interfaces for GPU task scheduling
with strict timing contracts. All scheduler implementations must meet the <100μs
scheduling latency requirement and provide consistent interfaces for task management.

Key features:
- Abstract base class with timing contracts
- Performance requirements documentation
- Metrics collection interfaces  
- Plugin architecture for custom schedulers
- Consistent scheduling decision format
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core.config import get_config
from ..utils.error_handling import SchedulingError, SchedulingLatencyViolation, ResourceExhaustedError, ErrorContext


T = TypeVar('T')


class SchedulingPolicy(str, Enum):
    """Available scheduling policies"""
    FIFO = "fifo"                    # First-in, first-out
    PRIORITY = "priority"            # Priority-based scheduling
    AFFINITY = "affinity"           # GPU affinity-aware scheduling
    GANG = "gang"                   # Gang scheduling for multi-GPU tasks
    REAL_TIME = "real_time"         # Real-time scheduling with preemption
    SHORTEST_JOB_FIRST = "sjf"      # Shortest job first
    ROUND_ROBIN = "round_robin"     # Round robin scheduling


class TaskState(str, Enum):
    """Task execution states"""
    PENDING = "pending"             # Waiting to be scheduled
    SCHEDULED = "scheduled"         # Assigned to GPU but not started
    RUNNING = "running"             # Currently executing
    COMPLETED = "completed"         # Successfully completed
    FAILED = "failed"              # Execution failed
    CANCELLED = "cancelled"        # Cancelled before completion
    TIMEOUT = "timeout"            # Exceeded maximum execution time


class ResourceType(str, Enum):
    """GPU resource types for allocation"""
    COMPUTE = "compute"             # GPU compute units
    MEMORY = "memory"              # GPU memory
    BANDWIDTH = "bandwidth"        # Memory bandwidth
    TENSOR_CORES = "tensor_cores"  # Specialized tensor processing units
    RT_CORES = "rt_cores"          # Ray tracing cores (if available)


@dataclass
class ResourceRequirement:
    """Resource requirements for a task"""
    resource_type: ResourceType
    amount: float                   # Amount of resource needed (0.0-1.0 for ratios)
    minimum: Optional[float] = None # Minimum acceptable amount
    maximum: Optional[float] = None # Maximum amount that can be utilized
    exclusive: bool = False         # Whether resource needs exclusive access
    
    def __post_init__(self):
        """Validate resource requirement parameters"""
        if self.amount < 0.0:
            raise ValueError("Resource amount cannot be negative")
        if self.minimum is not None and self.minimum > self.amount:
            raise ValueError("Minimum requirement cannot exceed requested amount")
        if self.maximum is not None and self.maximum < self.amount:
            raise ValueError("Maximum limit cannot be less than requested amount")


@dataclass 
class TaskRequest:
    """
    Comprehensive task request with scheduling metadata.
    
    This class represents a complete task scheduling request including
    resource requirements, timing constraints, and scheduling preferences.
    """
    # Basic task identification
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    tenant_id: Optional[str] = None
    
    # Resource requirements
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    memory_required_bytes: int = 0
    compute_required_ratio: float = 1.0  # 0.0-1.0, fraction of GPU compute needed
    
    # Timing constraints
    priority: int = 0                    # Higher number = higher priority
    max_execution_time_seconds: Optional[float] = None
    deadline: Optional[datetime] = None
    earliest_start_time: Optional[datetime] = None
    
    # Scheduling preferences
    gpu_affinity: Optional[List[str]] = None    # Preferred GPU IDs
    node_affinity: Optional[List[str]] = None   # Preferred node IDs
    anti_affinity: Optional[List[str]] = None   # GPUs/nodes to avoid
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.REAL_TIME
    
    # Task metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_by: Optional[str] = None
    task_type: Optional[str] = None
    estimated_duration: Optional[float] = None  # Estimated execution time in seconds
    
    # Advanced scheduling hints
    gang_size: int = 1                          # Number of GPUs needed simultaneously
    communication_pattern: Optional[str] = None # e.g., "all-reduce", "parameter-server"
    checkpointing_enabled: bool = False         # Whether task supports checkpointing
    preemptible: bool = True                   # Whether task can be preempted
    
    # Performance requirements
    max_scheduling_latency_us: Optional[float] = None  # Override global latency limit
    
    def add_resource_requirement(
        self, 
        resource_type: ResourceType, 
        amount: float, 
        **kwargs
    ) -> None:
        """Add a resource requirement to the task"""
        req = ResourceRequirement(resource_type, amount, **kwargs)
        self.resource_requirements.append(req)
    
    def get_resource_requirement(self, resource_type: ResourceType) -> Optional[ResourceRequirement]:
        """Get resource requirement by type"""
        for req in self.resource_requirements:
            if req.resource_type == resource_type:
                return req
        return None
    
    def validate(self) -> None:
        """Validate task request parameters"""
        if self.compute_required_ratio < 0.0 or self.compute_required_ratio > 1.0:
            raise ValueError("Compute requirement ratio must be between 0.0 and 1.0")
        
        if self.gang_size < 1:
            raise ValueError("Gang size must be at least 1")
        
        if self.memory_required_bytes < 0:
            raise ValueError("Memory requirement cannot be negative")
        
        if self.max_execution_time_seconds is not None and self.max_execution_time_seconds <= 0:
            raise ValueError("Maximum execution time must be positive")


@dataclass
class SchedulingDecision:
    """
    Result of a scheduling decision.
    
    Contains all information about where and when a task was scheduled,
    including performance metrics and resource allocation details.
    """
    # Basic scheduling information
    task_id: str
    node_id: str
    gpu_id: str
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    
    # Resource allocation
    allocated_memory_bytes: int = 0
    allocated_compute_ratio: float = 0.0
    allocated_resources: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Timing information
    decision_latency_us: float = 0.0            # Time to make scheduling decision
    estimated_start_time: Optional[datetime] = None
    estimated_completion_time: Optional[datetime] = None
    expected_duration_seconds: Optional[float] = None
    
    # Gang scheduling information
    gang_members: List[str] = field(default_factory=list)  # Other tasks in gang
    gang_coordinator: bool = False                         # Whether this task coordinates gang
    
    # Advanced scheduling metadata
    preemption_allowed: bool = True
    checkpoint_interval_seconds: Optional[float] = None
    scheduler_used: str = ""                              # Which scheduler made decision
    scheduling_reason: str = ""                           # Why this allocation was chosen
    alternative_placements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance validation
    meets_latency_requirement: bool = True
    performance_score: float = 1.0                       # Quality of placement (0.0-1.0)
    
    def validate_latency_requirement(self, max_latency_us: float) -> None:
        """Validate that scheduling decision meets latency requirement"""
        if self.decision_latency_us > max_latency_us:
            self.meets_latency_requirement = False
            raise SchedulingLatencyViolation(
                self.decision_latency_us,
                max_latency_us,
                context=ErrorContext.create_current(
                    task_id=self.task_id,
                    gpu_id=self.gpu_id,
                    operation="validate_latency"
                )
            )


@dataclass
class SchedulerMetrics:
    """Performance and operational metrics for scheduler"""
    
    # Latency metrics (in microseconds)
    avg_scheduling_latency_us: float = 0.0
    p50_scheduling_latency_us: float = 0.0
    p95_scheduling_latency_us: float = 0.0
    p99_scheduling_latency_us: float = 0.0
    max_scheduling_latency_us: float = 0.0
    
    # Throughput metrics
    tasks_scheduled_per_second: float = 0.0
    total_tasks_scheduled: int = 0
    successful_schedules: int = 0
    failed_schedules: int = 0
    
    # Resource utilization
    avg_gpu_utilization_percent: float = 0.0
    avg_memory_utilization_percent: float = 0.0
    resource_fragmentation_ratio: float = 0.0
    
    # Queue metrics
    avg_queue_depth: float = 0.0
    max_queue_depth: int = 0
    avg_wait_time_seconds: float = 0.0
    
    # Error metrics
    scheduling_errors: int = 0
    latency_violations: int = 0
    resource_exhaustion_events: int = 0
    
    # Timing
    metrics_collected_at: datetime = field(default_factory=datetime.utcnow)
    collection_period_seconds: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate scheduling success rate"""
        total = self.successful_schedules + self.failed_schedules
        return self.successful_schedules / total if total > 0 else 0.0
    
    def meets_latency_sla(self, sla_latency_us: float = 100.0) -> bool:
        """Check if scheduler meets latency SLA"""
        return (self.p99_scheduling_latency_us <= sla_latency_us and 
                self.max_scheduling_latency_us <= sla_latency_us * 2)  # Allow 2x burst


class SchedulerCallback(Protocol):
    """Protocol for scheduler event callbacks"""
    
    def on_task_scheduled(self, task: TaskRequest, decision: SchedulingDecision) -> None:
        """Called when task is successfully scheduled"""
        ...
    
    def on_task_failed(self, task: TaskRequest, error: Exception) -> None:
        """Called when task scheduling fails"""
        ...
    
    def on_scheduling_latency_violation(self, latency_us: float, limit_us: float) -> None:
        """Called when scheduling latency exceeds limits"""
        ...
    
    def on_resource_exhausted(self, resource_type: str, available: float, requested: float) -> None:
        """Called when resources are exhausted"""
        ...


class SchedulerInterface(ABC, Generic[T]):
    """
    Abstract base class for GPU task schedulers.
    
    All scheduler implementations must inherit from this class and implement
    the required methods while meeting the performance contracts specified.
    
    Performance Requirements:
    - Scheduling decisions must complete in <100μs (configurable)
    - Memory allocation decisions must complete in <10μs
    - System overhead must be <5% of direct GPU access latency
    - Support concurrent scheduling from multiple threads
    """
    
    def __init__(self, scheduler_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize scheduler with configuration.
        
        Args:
            scheduler_id: Unique identifier for this scheduler instance
            config: Optional configuration dictionary
        """
        self.scheduler_id = scheduler_id
        self.config = config or {}
        self.metrics = SchedulerMetrics()
        self.callbacks: List[SchedulerCallback] = []
        self._is_started = False
        
        # Performance configuration
        cluster_config = get_config()
        self.max_scheduling_latency_us = cluster_config.performance.max_scheduling_latency_us
        self.max_memory_allocation_latency_us = cluster_config.performance.memory_allocation_latency_us
        
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the scheduler.
        
        This method should set up any required data structures, connections,
        or resources needed for scheduling operations.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the scheduler and cleanup resources.
        
        This method should properly clean up all resources and ensure
        graceful termination of scheduling operations.
        """
        pass
    
    @abstractmethod
    async def schedule_task(self, task: TaskRequest) -> Optional[SchedulingDecision]:
        """
        Schedule a single task.
        
        This is the core scheduling method that must meet the <100μs latency
        requirement. The implementation must use lock-free data structures
        and avoid any blocking operations in the hot path.
        
        Args:
            task: Task to schedule
            
        Returns:
            Scheduling decision if successful, None if task cannot be scheduled
            
        Raises:
            SchedulingLatencyViolation: If scheduling exceeds latency limits
            ResourceExhaustedError: If insufficient resources available
        """
        pass
    
    @abstractmethod
    async def schedule_gang(self, tasks: List[TaskRequest]) -> Optional[List[SchedulingDecision]]:
        """
        Schedule a gang of tasks that must run simultaneously.
        
        Gang scheduling is used for multi-GPU tasks that require coordination.
        All tasks in the gang must be scheduled together or not at all.
        
        Args:
            tasks: List of tasks to schedule as a gang
            
        Returns:
            List of scheduling decisions if successful, None if gang cannot be scheduled
        """
        pass
    
    @abstractmethod
    async def preempt_task(self, task_id: str, reason: str = "") -> bool:
        """
        Preempt a running task to make room for higher priority tasks.
        
        Args:
            task_id: ID of task to preempt
            reason: Reason for preemption
            
        Returns:
            True if task was successfully preempted
        """
        pass
    
    @abstractmethod
    async def get_available_resources(self) -> Dict[str, Dict[ResourceType, float]]:
        """
        Get currently available resources by node and GPU.
        
        Returns:
            Dictionary mapping node_id -> gpu_id -> resource_type -> available_amount
        """
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[TaskState]:
        """
        Get current status of a scheduled task.
        
        Args:
            task_id: ID of task to query
            
        Returns:
            Current task state or None if task not found
        """
        pass
    
    # Optional methods with default implementations
    
    async def start(self) -> None:
        """Start the scheduler"""
        if self._is_started:
            return
        
        await self.initialize()
        self._is_started = True
    
    async def stop(self) -> None:
        """Stop the scheduler"""
        if not self._is_started:
            return
        
        await self.shutdown()
        self._is_started = False
    
    def add_callback(self, callback: SchedulerCallback) -> None:
        """Add event callback"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: SchedulerCallback) -> None:
        """Remove event callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def get_metrics(self) -> SchedulerMetrics:
        """Get current scheduler metrics"""
        return self.metrics
    
    async def reset_metrics(self) -> None:
        """Reset scheduler metrics"""
        self.metrics = SchedulerMetrics()
    
    def supports_policy(self, policy: SchedulingPolicy) -> bool:
        """Check if scheduler supports a specific scheduling policy"""
        return True  # Default implementation supports all policies
    
    def get_supported_policies(self) -> List[SchedulingPolicy]:
        """Get list of supported scheduling policies"""
        return list(SchedulingPolicy)
    
    def validate_task_request(self, task: TaskRequest) -> None:
        """
        Validate task request parameters.
        
        Args:
            task: Task request to validate
            
        Raises:
            ValueError: If task request is invalid
        """
        task.validate()
        
        # Additional scheduler-specific validation can be implemented by subclasses
        if not self.supports_policy(task.scheduling_policy):
            raise ValueError(f"Scheduling policy {task.scheduling_policy} not supported")
    
    def _record_scheduling_decision(
        self, 
        task: TaskRequest, 
        decision: SchedulingDecision, 
        start_time_ns: int
    ) -> None:
        """Record metrics for a scheduling decision"""
        end_time_ns = time.perf_counter_ns()
        latency_us = (end_time_ns - start_time_ns) / 1000
        
        decision.decision_latency_us = latency_us
        
        # Update metrics
        self.metrics.total_tasks_scheduled += 1
        
        # Check latency requirement
        try:
            max_latency = task.max_scheduling_latency_us or self.max_scheduling_latency_us
            decision.validate_latency_requirement(max_latency)
            self.metrics.successful_schedules += 1
        except SchedulingLatencyViolation as e:
            self.metrics.latency_violations += 1
            self.metrics.failed_schedules += 1
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback.on_scheduling_latency_violation(latency_us, max_latency)
                except Exception:
                    pass  # Don't let callback errors affect scheduling
            
            raise e
        
        # Update latency statistics (simplified)
        if self.metrics.max_scheduling_latency_us < latency_us:
            self.metrics.max_scheduling_latency_us = latency_us
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback.on_task_scheduled(task, decision)
            except Exception:
                pass  # Don't let callback errors affect scheduling
    
    def _record_scheduling_failure(self, task: TaskRequest, error: Exception) -> None:
        """Record metrics for a scheduling failure"""
        self.metrics.total_tasks_scheduled += 1
        self.metrics.failed_schedules += 1
        self.metrics.scheduling_errors += 1
        
        # Specific error type tracking
        if isinstance(error, ResourceExhaustedError):
            self.metrics.resource_exhaustion_events += 1
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback.on_task_failed(task, error)
            except Exception:
                pass
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(id={self.scheduler_id})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"{self.__class__.__name__}(id={self.scheduler_id}, "
                f"started={self._is_started}, metrics={self.metrics})")


class SchedulerFactory:
    """Factory for creating scheduler instances"""
    
    _scheduler_classes: Dict[str, type] = {}
    
    @classmethod
    def register_scheduler(cls, name: str, scheduler_class: type) -> None:
        """Register a scheduler implementation"""
        cls._scheduler_classes[name] = scheduler_class
    
    @classmethod
    def create_scheduler(
        cls, 
        scheduler_type: str, 
        scheduler_id: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> SchedulerInterface:
        """Create scheduler instance by type"""
        if scheduler_type not in cls._scheduler_classes:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        scheduler_class = cls._scheduler_classes[scheduler_type]
        return scheduler_class(scheduler_id, config)
    
    @classmethod
    def get_available_schedulers(cls) -> List[str]:
        """Get list of available scheduler types"""
        return list(cls._scheduler_classes.keys())