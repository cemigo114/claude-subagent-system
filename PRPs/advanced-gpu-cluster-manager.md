name: "Advanced GPU Cluster Manager - Production-Ready Distributed System"
description: |

## Purpose
Build a high-performance GPU cluster management system that provides direct hardware control, microsecond-level scheduling, and distributed coordination for enterprise-scale deployments. This system must achieve <100 microsecond scheduling latency, >95% GPU utilization, and support clusters with 1000+ GPUs across 100+ nodes.

## Core Principles
1. **Direct Hardware Control**: Bypass abstraction layers using NVML/ROCm/OpenCL APIs directly
2. **Microsecond Precision**: Lock-free data structures and event-driven architecture for real-time scheduling
3. **Distributed Resilience**: Raft consensus with automatic failover and split-brain prevention
4. **Production Ready**: Comprehensive monitoring, security, and multi-tenant isolation
5. **Performance Critical**: Every component optimized for minimal latency and maximum throughput

---

## Goal
Create a production-ready advanced GPU cluster management system that significantly outperforms existing solutions like GPUStack by providing direct hardware control, custom scheduling algorithms with microsecond-level decision making, advanced memory management, distributed peer-to-peer architecture, real-time monitoring, multi-tenant isolation, and high-performance networking with RDMA support.

## Why
- **Critical Performance Gap**: Existing solutions add 10-50ms scheduling overhead, unacceptable for real-time AI inference
- **Hardware Utilization**: Current systems achieve only 60-80% GPU utilization due to poor scheduling
- **Scale Limitations**: Most solutions fail at 100+ node scale due to centralized architectures
- **Multi-vendor Support**: Need to manage heterogeneous clusters (NVIDIA + AMD + Intel GPUs)
- **Enterprise Requirements**: Hardware-level isolation, compliance logging, and security needed for production

## What
A distributed GPU cluster management system providing:

### Core Capabilities
- **Direct Hardware Integration**: NVML/ROCm/OpenCL APIs for maximum performance control
- **Advanced Scheduling**: FIFO, priority-based, GPU affinity-aware algorithms with <100μs decisions
- **Memory Virtualization**: Custom GPU memory pools with defragmentation and precise tracking
- **Distributed Architecture**: Raft consensus, gossip protocols, automatic failover
- **Real-time Monitoring**: Nanosecond precision metrics, thermal/power management
- **Multi-tenant Security**: Hardware-enforced boundaries, quota management, audit logging
- **High-Speed Networking**: RDMA support for minimal latency inference

### Success Criteria
- [ ] Scheduling decisions complete in <100 microseconds
- [ ] Memory allocation/deallocation in <10 microseconds
- [ ] System overhead adds <5% latency vs direct GPU access
- [ ] Achieve >95% GPU utilization under mixed workloads
- [ ] Support 1000+ GPUs across 100+ nodes
- [ ] Node failure recovery in <30 seconds
- [ ] Zero split-brain scenarios during network partitions
- [ ] Multi-vendor GPU support (NVIDIA + AMD + Intel)

## All Needed Context

### Documentation & References (CRITICAL - Include these in context window)
```yaml
# HARDWARE INTEGRATION - MUST READ
- url: https://docs.nvidia.com/deploy/nvml-api/index.html
  why: Complete NVML API reference for direct hardware control
  critical: Device enumeration, memory management, thermal monitoring
  
- url: https://docs.nvidia.com/cuda/cuda-driver-api/index.html
  why: Low-level CUDA Driver API for maximum control
  critical: Context management, memory allocation, kernel execution
  
- url: https://rocm.docs.amd.com/projects/rocm_smi_lib/
  why: AMD ROCm SMI library for AMD GPU management
  critical: Multi-vendor support patterns
  
- url: https://dgpu-docs.intel.com/
  why: Intel GPU management documentation
  critical: Heterogeneous cluster support

# DISTRIBUTED SYSTEMS - ESSENTIAL
- url: https://raft.github.io/
  why: Raft consensus algorithm specification
  critical: Leader election, log replication, safety properties
  
- url: https://pkg.go.dev/go.etcd.io/etcd/raft/v3
  why: Production-tested Raft implementation patterns
  critical: Configuration changes, snapshotting, performance optimizations

# PERFORMANCE OPTIMIZATION - CRITICAL
- url: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
  why: CUDA performance best practices
  critical: Memory coalescing, occupancy optimization, latency hiding
  
- url: https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/
  why: ROCm performance tuning guidelines
  critical: Memory bandwidth optimization, compute unit utilization
  
- url: https://www.rdmamojo.com/
  why: RDMA programming guide for high-speed networking
  critical: Queue pair management, memory registration, low-latency patterns

# REAL-TIME SCHEDULING RESEARCH - MUST UNDERSTAND
- url: https://dl.acm.org/doi/10.1145/3600006.3613163
  why: Paella - Software-defined GPU scheduling for low latency
  critical: Preemptive scheduling, context switching optimization
  
- url: https://arxiv.org/html/2401.16529v1
  why: Real-time GPU task scheduling with priority-based preemption
  critical: Hardware scheduler control, driver-level implementation

# API COMPATIBILITY - INTEGRATION
- url: https://platform.openai.com/docs/api-reference
  why: OpenAI API compatibility layer
  critical: Chat completions, embeddings endpoints, streaming responses
```

### Current Codebase Tree
```bash
.
├── CLAUDE.md                           # Global project rules and conventions
├── INITIAL.md                          # Feature requirements and examples
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md                # PRP template structure
│   └── EXAMPLE_multi_agent_prp.md     # Multi-agent system example
├── examples/                           # (Referenced but empty - need to create)
└── use-cases/
    ├── pydantic-ai/                   # Pydantic AI examples for patterns
    │   └── examples/
    │       └── main_agent_reference/  # Agent structure patterns
    └── mcp-server/                    # MCP server patterns
```

### Desired Codebase Tree with Implementation
```bash
.
├── gpu_cluster/                        # Main package
│   ├── __init__.py                    # Package initialization
│   ├── core/                          # Core system components
│   │   ├── __init__.py
│   │   ├── cluster_manager.py         # Main cluster orchestration
│   │   ├── node_manager.py            # Individual node management
│   │   └── config.py                  # Configuration management
│   ├── hardware/                      # Hardware abstraction layer
│   │   ├── __init__.py
│   │   ├── gpu_manager.py             # Multi-vendor GPU management
│   │   ├── nvml_wrapper.py           # NVIDIA NVML integration
│   │   ├── rocm_wrapper.py           # AMD ROCm integration
│   │   ├── intel_wrapper.py          # Intel GPU integration
│   │   ├── memory_pool.py            # Custom memory management
│   │   ├── device_topology.py        # PCIe/NUMA topology discovery
│   │   └── thermal_manager.py        # Temperature/power management
│   ├── scheduling/                    # Scheduling algorithms
│   │   ├── __init__.py
│   │   ├── scheduler_interface.py     # Abstract scheduler base
│   │   ├── fifo_scheduler.py         # FIFO implementation
│   │   ├── priority_scheduler.py     # Priority-based scheduling
│   │   ├── affinity_scheduler.py     # GPU affinity-aware scheduling
│   │   ├── gang_scheduler.py         # Multi-GPU coordination
│   │   └── real_time_scheduler.py    # Microsecond-precision scheduler
│   ├── distributed/                   # Distributed system components
│   │   ├── __init__.py
│   │   ├── raft_consensus.py         # Raft implementation
│   │   ├── gossip_protocol.py        # Node health monitoring
│   │   ├── distributed_locks.py      # Resource locking
│   │   ├── failover_manager.py       # Failure detection/recovery
│   │   └── cluster_state.py          # Shared cluster state
│   ├── networking/                    # High-performance networking
│   │   ├── __init__.py
│   │   ├── rdma_transport.py         # RDMA implementation
│   │   ├── message_protocol.py       # Custom cluster protocol
│   │   └── load_balancer.py          # Request distribution
│   ├── monitoring/                    # Real-time monitoring
│   │   ├── __init__.py
│   │   ├── metrics_collector.py      # Hardware metrics collection
│   │   ├── kernel_tracer.py          # CUDA kernel profiling
│   │   ├── performance_monitor.py    # System performance tracking
│   │   └── alerting.py               # Alert management
│   ├── security/                      # Security and isolation
│   │   ├── __init__.py
│   │   ├── multi_tenant.py           # Multi-tenant isolation
│   │   ├── resource_quotas.py        # Resource quota management
│   │   ├── audit_logger.py           # Compliance logging
│   │   └── auth_manager.py           # Authentication/authorization
│   ├── apis/                          # API layers
│   │   ├── __init__.py
│   │   ├── grpc_server.py            # High-performance gRPC API
│   │   ├── rest_endpoints.py         # RESTful management API
│   │   ├── openai_compat.py          # OpenAI API compatibility
│   │   └── websocket_streaming.py    # Real-time streaming API
│   └── utils/                         # Utility modules
│       ├── __init__.py
│       ├── lock_free.py              # Lock-free data structures
│       ├── numa_allocator.py         # NUMA-aware allocation
│       ├── batch_processor.py        # Dynamic batching optimization
│       └── error_handling.py         # Centralized error handling
├── tests/                             # Comprehensive test suite
│   ├── __init__.py
│   ├── unit/                         # Unit tests
│   │   ├── test_hardware/
│   │   ├── test_scheduling/
│   │   ├── test_distributed/
│   │   ├── test_networking/
│   │   └── test_apis/
│   ├── integration/                  # Integration tests
│   │   ├── test_cluster_operations.py
│   │   ├── test_failover_scenarios.py
│   │   └── test_performance.py
│   ├── performance/                  # Performance benchmarks
│   │   ├── latency_benchmarks.py
│   │   ├── throughput_tests.py
│   │   └── stress_tests.py
│   └── chaos/                        # Chaos engineering tests
│       ├── network_partition_tests.py
│       ├── node_failure_tests.py
│       └── resource_exhaustion_tests.py
├── examples/                          # Usage examples
│   ├── basic_cluster_setup.py
│   ├── multi_tenant_deployment.py
│   └── benchmark_suite.py
├── docs/                             # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   └── performance_tuning.md
├── scripts/                          # Deployment and management
│   ├── cluster_deploy.py
│   ├── performance_benchmark.py
│   └── monitoring_setup.py
├── configs/                          # Configuration files
│   ├── cluster_config.yaml
│   ├── security_config.yaml
│   └── monitoring_config.yaml
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup
├── pyproject.toml                    # Modern Python packaging
├── .env.example                      # Environment variables template
└── README.md                         # Comprehensive documentation
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: NVML/CUDA Driver API requires careful initialization
# Must call cuInit() before any CUDA Driver API calls
# NVML requires nvmlInit() before device queries
# Example: Always check return codes - GPU operations can fail silently

# CRITICAL: Memory alignment requirements differ by vendor
# NVIDIA: 256-byte alignment for optimal performance
# AMD: 4KB page alignment for ROCm
# Intel: 64-byte alignment for cache efficiency

# CRITICAL: GPU context switching is expensive (10-100μs)
# Minimize context switches in scheduler hot path
# Use single persistent context per GPU when possible
# Preemption requires driver-level coordination

# CRITICAL: RDMA programming requires careful memory management
# Memory regions must be registered before use
# Queue pairs need proper flow control
# Completion queues must be polled, not interrupt-driven for low latency

# CRITICAL: Raft implementation gotchas
# Network partitions can cause split elections
# Log compaction must preserve state machine safety
# Configuration changes require special two-phase protocol

# CRITICAL: Multi-vendor GPU coordination challenges
# Different vendors have incompatible memory spaces
# Peer-to-peer transfers may not work across vendors
# Unified virtual addressing not available on all platforms

# CRITICAL: Real-time scheduling constraints
# No blocking operations in scheduler thread
# Use lock-free data structures (atomic operations only)
# Interrupt handling must complete in <10μs
# Memory allocations must be pre-allocated pools

# CRITICAL: Thermal management across vendors
# NVIDIA: NVML provides temperature/power APIs
# AMD: ROCm SMI has different temperature semantics
# Intel: Limited thermal exposure in current drivers
# Must implement vendor-specific thermal policies

# CRITICAL: Security isolation limitations
# GPU contexts provide limited isolation
# Memory protection relies on driver enforcement
# Side-channel attacks possible through timing
# Multi-tenant workloads need careful resource bounds
```

## Implementation Blueprint

### Data Models and Structure

```python
# Core data models ensuring type safety and performance
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
from enum import Enum
import uuid
from datetime import datetime

class GPUVendor(str, Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"

class GPUStatus(str, Enum):
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class SchedulingPolicy(str, Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    AFFINITY = "affinity"
    GANG = "gang"
    REAL_TIME = "real_time"

class NodeHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    FAILED = "failed"

class GPUDevice(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")
    vendor: GPUVendor
    model: str
    memory_total: int = Field(..., description="Total memory in bytes")
    memory_available: int = Field(..., description="Available memory in bytes")
    compute_capability: str
    pcie_bus_id: str
    numa_node: int
    temperature: float = Field(..., description="Temperature in Celsius")
    power_usage: float = Field(..., description="Power usage in Watts")
    utilization: float = Field(..., ge=0.0, le=100.0)
    status: GPUStatus = GPUStatus.AVAILABLE

class TaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    priority: int = Field(default=0, ge=0, le=100)
    memory_required: int = Field(..., description="Memory requirement in bytes")
    compute_required: float = Field(..., description="Compute requirement (0.0-1.0)")
    max_execution_time: float = Field(..., description="Max execution time in seconds")
    gpu_affinity: Optional[List[str]] = None
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.FIFO
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ClusterNode(BaseModel):
    node_id: str = Field(..., description="Unique node identifier")
    hostname: str
    ip_address: str
    port: int = Field(default=8080)
    gpus: List[GPUDevice]
    health: NodeHealth = NodeHealth.HEALTHY
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    raft_role: str = Field(default="follower")  # leader, follower, candidate
    
class SchedulingDecision(BaseModel):
    task_id: str
    node_id: str
    gpu_id: str
    scheduled_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_estimate: float
    decision_latency_us: float = Field(..., description="Scheduling decision latency in microseconds")

class PerformanceMetrics(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scheduling_latency_us: float
    memory_allocation_latency_us: float
    gpu_utilization_percent: float
    cluster_throughput_tasks_per_second: float
    active_tasks: int
    queued_tasks: int
```

### List of Tasks to be Completed (Implementation Order)

```yaml
Task 1: Core Infrastructure Setup
CREATE gpu_cluster/core/config.py:
  - PATTERN: Use pydantic-settings for configuration management
  - Environment variable loading with validation
  - Multi-vendor GPU detection and enumeration
  - Performance tuning parameters (scheduling timeouts, memory pool sizes)

CREATE gpu_cluster/utils/error_handling.py:
  - PATTERN: Centralized exception hierarchy
  - GPU vendor-specific error mapping
  - Performance-critical error paths (no logging in hot path)
  - Recovery strategies for different failure modes

Task 2: Hardware Abstraction Layer
CREATE gpu_cluster/hardware/nvml_wrapper.py:
  - PATTERN: Direct NVML API calls with ctypes/cffi for performance
  - Device enumeration, memory management, thermal monitoring
  - Context management for multi-process access
  - Error code mapping to Python exceptions

CREATE gpu_cluster/hardware/rocm_wrapper.py:
  - PATTERN: Mirror NVML wrapper structure for consistency
  - ROCm SMI library integration
  - AMD-specific memory alignment and allocation
  - Temperature and power monitoring via ROCm APIs

CREATE gpu_cluster/hardware/intel_wrapper.py:
  - PATTERN: Consistent interface with NVML/ROCm wrappers
  - Intel GPU control library integration
  - Limited feature set handling gracefully
  - Future-proofing for Intel GPU expansion

CREATE gpu_cluster/hardware/gpu_manager.py:
  - PATTERN: Factory pattern for vendor-specific managers
  - Unified interface abstracting vendor differences
  - Device topology discovery and NUMA awareness
  - Multi-vendor memory space coordination

CREATE gpu_cluster/hardware/memory_pool.py:
  - PATTERN: Custom allocator with compaction algorithms
  - Lock-free allocation for performance critical paths
  - Vendor-specific alignment requirements
  - Memory fragmentation tracking and defragmentation

Task 3: Lock-Free Data Structures
CREATE gpu_cluster/utils/lock_free.py:
  - PATTERN: Atomic operations for queue and hash table
  - Memory ordering guarantees for x86/ARM
  - Lock-free FIFO queue for task scheduling
  - Atomic reference counting for resource management
  - CAS (Compare-And-Swap) based data structures

Task 4: Real-Time Scheduling Engine
CREATE gpu_cluster/scheduling/scheduler_interface.py:
  - PATTERN: Abstract base class with timing contracts
  - Performance requirements documentation
  - Metrics collection interfaces
  - Plugin architecture for custom schedulers

CREATE gpu_cluster/scheduling/real_time_scheduler.py:
  - PATTERN: Lock-free implementation using atomic operations
  - Pre-allocated data structures to avoid malloc in hot path
  - Microsecond-precision timing with clock_gettime
  - Interrupt-driven preemption support
  - Hardware scheduler coordination for NVIDIA

CREATE gpu_cluster/scheduling/priority_scheduler.py:
  - PATTERN: Priority queue with O(log n) operations
  - Aging algorithm to prevent starvation
  - Dynamic priority adjustment based on resource availability
  - Integration with real-time scheduler for preemption

Task 5: Distributed Consensus Layer
CREATE gpu_cluster/distributed/raft_consensus.py:
  - PATTERN: Follow etcd raft implementation structure
  - Leader election with randomized timeouts
  - Log replication with batching for performance
  - Snapshot mechanism for large cluster state
  - Configuration change handling (adding/removing nodes)

CREATE gpu_cluster/distributed/cluster_state.py:
  - PATTERN: Immutable data structures for consistency
  - Copy-on-write semantics for state updates
  - Merkle tree for state verification
  - Efficient serialization for network transport

CREATE gpu_cluster/distributed/gossip_protocol.py:
  - PATTERN: Epidemic-style information dissemination
  - Failure detection with configurable timeouts
  - Network partition handling
  - Membership protocol for dynamic clusters

Task 6: High-Performance Networking
CREATE gpu_cluster/networking/rdma_transport.py:
  - PATTERN: InfiniBand/RoCE programming model
  - Memory region registration and management
  - Queue pair setup and completion queue polling
  - Zero-copy data transfer for large payloads
  - Reliable connection with error recovery

CREATE gpu_cluster/networking/message_protocol.py:
  - PATTERN: Binary protocol with message framing
  - Protocol buffer or capnproto for serialization
  - Message compression for large cluster state
  - Flow control and congestion avoidance

Task 7: Monitoring and Observability
CREATE gpu_cluster/monitoring/metrics_collector.py:
  - PATTERN: Time-series data collection with minimal overhead
  - Hardware counter integration (PCIe, memory bandwidth)
  - Vendor-specific metric gathering
  - Prometheus metrics export format

CREATE gpu_cluster/monitoring/kernel_tracer.py:
  - PATTERN: CUDA profiling API integration
  - Kernel execution timeline tracking
  - Memory transfer monitoring
  - Performance bottleneck identification

Task 8: Security and Multi-Tenancy
CREATE gpu_cluster/security/multi_tenant.py:
  - PATTERN: Resource namespace isolation
  - GPU context separation by tenant
  - Memory protection using GPU virtual addressing
  - API rate limiting per tenant

CREATE gpu_cluster/security/audit_logger.py:
  - PATTERN: Structured logging for compliance
  - Tamper-evident log storage
  - Resource access tracking
  - Security event correlation

Task 9: API Layer Implementation
CREATE gpu_cluster/apis/grpc_server.py:
  - PATTERN: High-performance gRPC with streaming
  - Protocol buffer definitions for all APIs
  - Connection pooling and load balancing
  - Async request handling with proper backpressure

CREATE gpu_cluster/apis/openai_compat.py:
  - PATTERN: FastAPI with OpenAI API specification
  - Streaming responses for chat completions
  - Load balancing across GPU nodes
  - Request routing based on model requirements

Task 10: Integration and Testing
CREATE tests/performance/latency_benchmarks.py:
  - PATTERN: Microbenchmarks for critical paths
  - Scheduling latency measurement (<100μs validation)
  - Memory allocation latency testing
  - End-to-end request latency measurement

CREATE tests/chaos/network_partition_tests.py:
  - PATTERN: Chaos engineering with controlled failures
  - Raft consensus validation during partitions
  - Split-brain prevention testing
  - Recovery time measurement (<30s validation)

Task 11: Deployment and Documentation
CREATE scripts/cluster_deploy.py:
  - PATTERN: Infrastructure as code deployment
  - Multi-node cluster orchestration
  - Configuration validation and health checks
  - Rolling upgrade procedures

CREATE docs/performance_tuning.md:
  - PATTERN: Comprehensive performance guide
  - Vendor-specific optimization settings
  - Network and storage tuning recommendations
  - Troubleshooting performance issues
```

### Per Task Pseudocode (Critical Implementation Details)

```python
# Task 4: Real-Time Scheduler Implementation
class RealTimeScheduler:
    def __init__(self):
        # CRITICAL: Pre-allocate all data structures
        self.ready_queue = LockFreeQueue(capacity=10000)
        self.running_tasks = AtomicHashMap(capacity=1000)
        self.gpu_states = AtomicArray(size=1000)
        
        # CRITICAL: Use high-resolution timer
        self.timer_fd = create_timer_fd(CLOCK_MONOTONIC)
        
    async def schedule_task(self, task: TaskRequest) -> SchedulingDecision:
        start_time = time.perf_counter_ns()
        
        # CRITICAL: No memory allocation in hot path
        available_gpus = self._get_available_gpus_lockfree()
        
        if not available_gpus:
            # CRITICAL: Fast path for queue insertion
            self.ready_queue.enqueue_atomic(task)
            return None
            
        # CRITICAL: O(1) GPU selection for real-time constraint
        best_gpu = self._select_gpu_atomic(available_gpus, task)
        
        # CRITICAL: Atomic resource reservation
        if not self._reserve_gpu_atomic(best_gpu, task):
            self.ready_queue.enqueue_atomic(task)
            return None
            
        decision_time = time.perf_counter_ns() - start_time
        
        # CRITICAL: Must complete in <100 microseconds
        assert decision_time < 100_000, f"Scheduling took {decision_time}ns"
        
        return SchedulingDecision(
            task_id=task.task_id,
            gpu_id=best_gpu.device_id,
            decision_latency_us=decision_time / 1000
        )

# Task 5: Raft Consensus Implementation  
class RaftNode:
    def __init__(self, node_id: str, peers: List[str]):
        self.node_id = node_id
        self.peers = peers
        self.state = RaftState.FOLLOWER
        self.current_term = 0
        self.voted_for = None
        self.log = RaftLog()
        
        # CRITICAL: Randomized election timeout to prevent split votes
        self.election_timeout = random.uniform(150, 300)  # milliseconds
        
    async def append_entries(self, term: int, leader_id: str, 
                           prev_log_index: int, prev_log_term: int,
                           entries: List[LogEntry]) -> AppendEntriesResponse:
        # CRITICAL: Raft safety properties
        if term < self.current_term:
            return AppendEntriesResponse(success=False, term=self.current_term)
            
        # CRITICAL: Log consistency check
        if not self.log.contains_entry(prev_log_index, prev_log_term):
            return AppendEntriesResponse(success=False, term=self.current_term)
            
        # CRITICAL: Atomic log append for consistency
        self.log.append_entries_atomic(prev_log_index + 1, entries)
        
        return AppendEntriesResponse(success=True, term=self.current_term)

# Task 6: RDMA Transport Implementation
class RDMATransport:
    def __init__(self, device_name: str, port: int):
        # CRITICAL: InfiniBand/RoCE device initialization
        self.context = ibv.open_device(device_name)
        self.pd = ibv.alloc_pd(self.context)
        self.cq = ibv.create_cq(self.context, 1024)
        
        # CRITICAL: Memory region registration for zero-copy
        self.send_buffer = mmap.mmap(-1, 1024 * 1024)  # 1MB buffer
        self.send_mr = ibv.reg_mr(self.pd, self.send_buffer, 
                                  ibv.IBV_ACCESS_LOCAL_WRITE)
        
    async def send_message(self, dest_qp: int, message: bytes) -> None:
        # CRITICAL: Zero-copy send for performance
        memcpy(self.send_buffer, message)
        
        # CRITICAL: RDMA send with completion notification
        wr = ibv.SendWR(
            wr_id=self._next_wr_id(),
            sg_list=[ibv.SGE(addr=self.send_mr.addr, 
                           length=len(message),
                           lkey=self.send_mr.lkey)],
            opcode=ibv.IBV_WR_SEND
        )
        
        ibv.post_send(dest_qp, wr)
        
        # CRITICAL: Poll completion queue for low latency
        while True:
            wc = ibv.poll_cq(self.cq, 1)
            if wc and wc[0].wr_id == wr.wr_id:
                if wc[0].status != ibv.IBV_WC_SUCCESS:
                    raise RDMAError(f"Send failed: {wc[0].status}")
                break
            await asyncio.sleep(0)  # Yield to event loop

# Task 8: Multi-Tenant GPU Isolation
class MultiTenantIsolation:
    def __init__(self):
        self.tenant_contexts = {}  # tenant_id -> GPU contexts
        self.resource_quotas = {}  # tenant_id -> resource limits
        
    def create_tenant_context(self, tenant_id: str, gpu_id: str) -> CUcontext:
        # CRITICAL: Hardware-enforced isolation via GPU contexts
        device = cuda.Device(gpu_id)
        context = device.make_context(flags=cuda.ctx_flags.SCHED_BLOCKING_SYNC)
        
        # CRITICAL: Set memory limits at hardware level
        quota = self.resource_quotas[tenant_id]
        cuda.mem_set_limit(quota.memory_limit)
        
        # CRITICAL: Enable GPU context isolation
        context.set_limit(cuda.limit.MALLOC_HEAP_SIZE, quota.heap_size)
        context.set_limit(cuda.limit.STACK_SIZE, quota.stack_size)
        
        self.tenant_contexts[tenant_id] = context
        return context
        
    def allocate_gpu_memory(self, tenant_id: str, size: int) -> DeviceAllocation:
        context = self.tenant_contexts[tenant_id]
        
        # CRITICAL: Context switching for isolation
        with context:
            # CRITICAL: Check tenant quota before allocation
            if not self._check_memory_quota(tenant_id, size):
                raise QuotaExceededError(f"Tenant {tenant_id} quota exceeded")
                
            # CRITICAL: Hardware-isolated memory allocation
            return cuda.mem_alloc(size)
```

### Integration Points
```yaml
HARDWARE_DRIVERS:
  - nvidia_drivers: ">=525.0.0"  # Required for latest NVML features
  - amdgpu_drivers: ">=23.0.0"   # ROCm 6.0+ support
  - intel_drivers: ">=1.3.0"     # Intel GPU driver support
  
SYSTEM_DEPENDENCIES:
  - rdma_core: ">=28.0"          # InfiniBand/RoCE support
  - numactl: ">=2.0.14"          # NUMA topology management
  - hwloc: ">=2.7.0"             # Hardware topology discovery
  - liburing: ">=2.2"            # io_uring for async I/O

PYTHON_DEPENDENCIES:
  - pydantic: ">=2.0.0"          # Data validation and settings
  - uvloop: ">=0.17.0"           # High-performance async I/O
  - cffi: ">=1.15.0"             # C library bindings
  - msgpack: ">=1.0.0"           # Binary serialization
  - prometheus_client: ">=0.16.0" # Metrics export
  - grpcio: ">=1.50.0"           # High-performance RPC
  - fastapi: ">=0.100.0"         # REST API framework
  - psutil: ">=5.9.0"            # System monitoring

CONFIGURATION:
  - cluster_config: "/etc/gpu-cluster/cluster.yaml"
  - security_config: "/etc/gpu-cluster/security.yaml"
  - performance_tuning: "/etc/gpu-cluster/performance.yaml"
  
MONITORING:
  - prometheus_endpoint: "http://localhost:9090/metrics"
  - grafana_dashboard: "gpu-cluster-monitoring.json"
  - alert_manager: "gpu-cluster-alerts.yaml"
```

## Validation Loop

### Level 1: Syntax & Style & Performance
```bash
# CRITICAL: Performance-aware linting
ruff check gpu_cluster/ --fix --select=E,W,F,UP,B,SIM,I,N,PERF
mypy gpu_cluster/ --strict --warn-performance-implications

# CRITICAL: Security scanning for multi-tenant system
bandit -r gpu_cluster/ -f json -o security_report.json

# CRITICAL: Memory profiler for allocation patterns
python -m memory_profiler tests/performance/memory_benchmark.py

# Expected: No errors, no performance anti-patterns, no security issues
```

### Level 2: Unit Tests with Performance Validation
```python
# test_real_time_scheduler.py
import time
import pytest
import asyncio
from gpu_cluster.scheduling.real_time_scheduler import RealTimeScheduler

@pytest.mark.asyncio
async def test_scheduling_latency_requirement():
    """CRITICAL: Validate <100μs scheduling requirement"""
    scheduler = RealTimeScheduler()
    task = create_test_task(memory_required=1024*1024*1024)  # 1GB
    
    # Warm up to eliminate cold start effects
    for _ in range(100):
        await scheduler.schedule_task(task)
        
    # Measure actual scheduling latency
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        decision = await scheduler.schedule_task(task)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000)  # Convert to microseconds
        
    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[990]  # 99th percentile
    
    # CRITICAL: Both average and 99th percentile must be <100μs
    assert avg_latency < 100, f"Average latency {avg_latency}μs exceeds 100μs"
    assert p99_latency < 100, f"P99 latency {p99_latency}μs exceeds 100μs"

@pytest.mark.asyncio
async def test_memory_allocation_latency():
    """CRITICAL: Validate <10μs memory allocation requirement"""
    memory_pool = GPUMemoryPool(total_size=8*1024*1024*1024)  # 8GB
    
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        allocation = memory_pool.allocate(1024*1024)  # 1MB
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000)
        memory_pool.deallocate(allocation)
        
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 10, f"Memory allocation latency {avg_latency}μs exceeds 10μs"

@pytest.mark.asyncio
async def test_raft_consensus_correctness():
    """CRITICAL: Validate Raft safety properties"""
    cluster = RaftCluster(nodes=5)
    
    # Test leader election
    leader = await cluster.elect_leader()
    assert leader is not None
    
    # Test log replication
    entries = [LogEntry(term=1, data=f"command_{i}") for i in range(100)]
    success = await leader.replicate_entries(entries)
    assert success
    
    # Test split-brain prevention
    await cluster.create_network_partition([0, 1], [2, 3, 4])
    
    # Majority partition should continue, minority should not
    majority_leader = await cluster.nodes[2].try_become_leader()
    minority_leader = await cluster.nodes[0].try_become_leader() 
    
    assert majority_leader is not None
    assert minority_leader is None  # Should fail due to lack of majority

def test_multi_vendor_gpu_coordination():
    """CRITICAL: Validate heterogeneous GPU cluster support"""
    nvidia_gpu = create_nvidia_gpu_mock()
    amd_gpu = create_amd_gpu_mock()
    intel_gpu = create_intel_gpu_mock()
    
    cluster = GPUCluster([nvidia_gpu, amd_gpu, intel_gpu])
    
    # Test cross-vendor memory transfers
    with pytest.raises(CrossVendorTransferError):
        cluster.transfer_memory(nvidia_gpu, amd_gpu, 1024)
        
    # Test vendor-specific optimization
    nvidia_task = TaskRequest(vendor_preference="nvidia")
    decision = cluster.schedule_task(nvidia_task)
    assert decision.gpu_id.startswith("nvidia")

def test_rdma_transport_performance():
    """CRITICAL: Validate RDMA low-latency requirements"""
    transport = RDMATransport("mlx5_0", port=1)
    
    # Test round-trip latency
    latencies = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        await transport.ping_pong("test_message")
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000)
        
    avg_latency = sum(latencies) / len(latencies)
    # RDMA should achieve <10μs round-trip on modern hardware
    assert avg_latency < 10, f"RDMA latency {avg_latency}μs too high"
```

```bash
# Run comprehensive test suite with performance validation
pytest tests/ -v --cov=gpu_cluster --cov-report=html \
  --benchmark-only --benchmark-min-rounds=1000

# CRITICAL: Chaos engineering tests
pytest tests/chaos/ -v --chaos-duration=300  # 5 minutes of chaos

# CRITICAL: 72-hour stress test for memory leaks
python tests/performance/stress_test.py --duration=259200  # 72 hours
```

### Level 3: Integration Test with Real Hardware
```bash
# CRITICAL: Deploy test cluster on real hardware
./scripts/cluster_deploy.py --config configs/test_cluster.yaml --nodes 3

# Test actual GPU scheduling performance
python tests/integration/test_scheduling_performance.py \
  --duration=3600 --target-utilization=95

# Expected output:
# ✅ Average scheduling latency: 85μs (target: <100μs)
# ✅ P99 scheduling latency: 97μs (target: <100μs)  
# ✅ GPU utilization: 96.2% (target: >95%)
# ✅ Node failure recovery: 28s (target: <30s)
# ✅ Zero split-brain incidents detected

# Test OpenAI API compatibility
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key" \
  -d '{
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Hello GPU cluster!"}],
    "stream": true
  }'

# Expected: Streaming response with <100ms first token latency

# Test multi-tenant isolation
python tests/integration/test_tenant_isolation.py \
  --tenants 10 --concurrent-tasks 100

# Expected: No cross-tenant memory access, quota enforcement working
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v --cov-min=90`
- [ ] No linting errors: `ruff check gpu_cluster/`
- [ ] No type errors: `mypy gpu_cluster/ --strict`
- [ ] Performance requirements met: `python scripts/performance_benchmark.py`
  - [ ] Scheduling latency <100μs (avg and P99)
  - [ ] Memory allocation latency <10μs
  - [ ] GPU utilization >95% under load
  - [ ] System overhead <5% vs direct GPU access
- [ ] Distributed system validation:
  - [ ] Raft consensus working correctly
  - [ ] Zero split-brain scenarios in testing
  - [ ] Node failure recovery <30s
  - [ ] Network partition handling
- [ ] Multi-vendor support validated:
  - [ ] NVIDIA GPU management working
  - [ ] AMD GPU management working  
  - [ ] Intel GPU management working
  - [ ] Cross-vendor coordination
- [ ] Security and isolation:
  - [ ] Multi-tenant isolation enforced
  - [ ] Resource quotas working
  - [ ] Audit logging comprehensive
  - [ ] No privilege escalation vulnerabilities
- [ ] API compatibility:
  - [ ] OpenAI API endpoints working
  - [ ] gRPC high-performance API working
  - [ ] WebSocket streaming working
- [ ] Documentation complete:
  - [ ] Architecture documentation
  - [ ] API reference complete
  - [ ] Deployment guide tested
  - [ ] Performance tuning guide validated

---

## Anti-Patterns to Avoid
- ❌ Don't use high-level GPU libraries (CuPy) - implement direct CUDA/ROCm calls
- ❌ Avoid blocking operations in scheduler - use async/event-driven patterns  
- ❌ Don't ignore NUMA topology - memory locality significantly impacts performance
- ❌ Don't use locks in scheduling hot path - implement lock-free data structures
- ❌ Avoid memory allocations in real-time scheduler - pre-allocate all structures
- ❌ Don't skip proper GPU context management - leads to resource leaks
- ❌ Avoid synchronous network I/O - use async/await throughout
- ❌ Don't ignore vendor-specific optimizations - each GPU vendor needs special handling
- ❌ Avoid hardcoded timeouts - make all timing parameters configurable
- ❌ Don't skip chaos testing - distributed systems fail in unexpected ways
- ❌ Avoid single points of failure - every component must be fault-tolerant
- ❌ Don't ignore thermal events - implement proper thermal management
- ❌ Avoid insecure multi-tenancy - enforce hardware-level isolation

## Confidence Score: 8/10

High confidence due to:
- Comprehensive research on NVML, CUDA Driver API, Raft, and real-time scheduling
- Clear performance requirements with measurable validation criteria
- Well-defined architecture with vendor-specific abstraction layers
- Extensive testing strategy including chaos engineering
- Real-world examples and documentation from recent research (2024)

Moderate uncertainty areas:
- Intel GPU driver maturity for production use (-1 point)
- RDMA implementation complexity across different hardware vendors (-1 point)
- Multi-vendor GPU memory coordination edge cases

The implementation blueprint provides sufficient context for building a production-ready GPU cluster manager that meets all specified performance requirements.