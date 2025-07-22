# INITIAL.md - Advanced GPU Cluster Manager

## FEATURE:
Build an advanced GPU cluster management system that provides significantly more low-level control than existing solutions like GPUStack. The system should offer:

- **Direct Hardware Control**: Bypass abstraction layers for maximum performance, with direct CUDA/ROCm/OpenCL integration
- **Custom Scheduling Algorithms**: Implement multiple scheduling strategies (FIFO, priority-based, GPU affinity-aware) with microsecond-level decision making
- **Advanced Memory Management**: Custom GPU memory pools with precise allocation tracking, defragmentation, and virtualization support
- **Distributed Architecture**: Peer-to-peer cluster management with consensus protocols and automatic failover
- **Real-time Monitoring**: Hardware-level metrics collection with nanosecond precision for GPU kernel execution, power consumption, and thermal monitoring
- **Multi-tenant Isolation**: Hardware-enforced resource boundaries with fair scheduling and quota management
- **High-Performance Networking**: Custom protocols for minimal latency inference with RDMA support for high-speed interconnects

The system must achieve <100 microsecond scheduling latency, >95% GPU utilization, and support clusters with 1000+ GPUs across 100+ nodes.

## EXAMPLES:
Reference the following examples in the `examples/` directory:

### GPU Hardware Management
- `examples/gpu_manager.py` - Direct NVIDIA NVML/AMD SMI integration patterns
- `examples/memory_pool.py` - Custom memory allocator with compaction algorithms
- `examples/device_topology.py` - PCIe topology discovery and NUMA awareness
- `examples/thermal_manager.py` - Temperature monitoring and power management

### Advanced Scheduling
- `examples/schedulers/fifo_scheduler.py` - First-in-first-out scheduling implementation
- `examples/schedulers/priority_scheduler.py` - Priority-based scheduling with preemption
- `examples/schedulers/affinity_scheduler.py` - GPU affinity and locality-aware scheduling
- `examples/schedulers/gang_scheduler.py` - Multi-GPU task coordination

### Distributed Systems
- `examples/consensus/raft.py` - Raft consensus protocol for cluster state
- `examples/networking/gossip.py` - Gossip protocol for node health monitoring
- `examples/coordination/distributed_locks.py` - Resource locking across nodes
- `examples/failover/recovery_manager.py` - Automatic failure detection and recovery

### Performance Optimization
- `examples/profiling/kernel_tracer.py` - CUDA kernel execution profiling
- `examples/optimization/batch_processor.py` - Dynamic batching optimization
- `examples/networking/rdma_transport.py` - RDMA network transport layer
- `examples/memory/numa_allocator.py` - NUMA-aware memory allocation

### API Integration
- `examples/apis/openai_compat.py` - OpenAI API compatibility layer
- `examples/apis/grpc_server.py` - High-performance gRPC API server
- `examples/apis/rest_endpoints.py` - RESTful management API patterns

## DOCUMENTATION:
Include comprehensive documentation and API references:

### Hardware Integration
- NVIDIA NVML API Documentation: https://docs.nvidia.com/deploy/nvml-api/
- AMD ROCm SMI Documentation: https://rocm.docs.amd.com/projects/rocm_smi_lib/
- CUDA Driver API Reference: https://docs.nvidia.com/cuda/cuda-driver-api/
- Intel GPU Tools Documentation: https://dgpu-docs.intel.com/

### Distributed Systems
- Raft Consensus Algorithm: https://raft.github.io/
- etcd Raft Implementation: https://pkg.go.dev/go.etcd.io/etcd/raft/v3
- Gossip Protocol Patterns: https://en.wikipedia.org/wiki/Gossip_protocol

### Performance Optimization
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- ROCm Performance Guidelines: https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/
- RDMA Programming Guide: https://www.rdmamojo.com/

### OpenAI API Compatibility
- OpenAI API Reference: https://platform.openai.com/docs/api-reference
- Chat Completions API: https://platform.openai.com/docs/api-reference/chat
- Embeddings API: https://platform.openai.com/docs/api-reference/embeddings

## OTHER CONSIDERATIONS:

### Critical Performance Requirements
- Scheduling decisions must complete in <100 microseconds to avoid inference latency impact
- Memory allocation/deallocation must be <10 microseconds to support real-time workloads
- System overhead should add <5% latency compared to direct GPU access
- Must achieve >95% GPU utilization under mixed workloads with proper scheduling

### Hardware Compatibility Challenges
- Support heterogeneous GPU clusters (NVIDIA + AMD + Intel in same cluster)
- Handle different GPU generations with varying capabilities
- Manage PCIe bandwidth limitations and topology optimization
- Deal with thermal throttling and power management across vendors

### Distributed System Complexities
- Prevent split-brain scenarios during network partitions
- Ensure data consistency during concurrent resource allocations
- Handle node failures gracefully with <30 second recovery times
- Maintain cluster state consistency across 100+ nodes

### Memory Management Gotchas
- GPU memory fragmentation can severely impact allocation success
- Different GPU vendors have different memory alignment requirements
- Memory oversubscription needs careful handling to prevent OOM failures
- Cross-GPU memory transfers have significant bandwidth implications

### Security and Multi-tenancy
- Hardware-level isolation is challenging - use GPU contexts and memory protection
- API rate limiting must be implemented at multiple layers (network, scheduler, GPU)
- Audit logging for compliance needs to capture all resource access
- Secrets management for API keys and inter-node communication certificates

### Common AI Assistant Mistakes to Avoid
- Don't use high-level GPU libraries (like CuPy) - implement direct CUDA/ROCm calls
- Avoid blocking operations in the scheduler - use async/event-driven patterns
- Don't ignore NUMA topology - memory locality significantly impacts performance  
- Implement proper resource cleanup to prevent GPU memory leaks
- Use lock-free data structures in the hot path for scheduling decisions
- Implement proper backpressure to prevent resource exhaustion
- Don't forget to handle GPU ECC errors and thermal events
- Validate all user inputs to prevent resource exhaustion attacks

### Testing and Validation Requirements
- Must test on actual GPU hardware - simulators don't capture real performance characteristics
- Implement chaos engineering tests for network partitions and node failures
- Performance regression testing must run on identical hardware configurations
- Memory leak detection requires long-running stress tests (72+ hours)
- Multi-tenant isolation testing needs to simulate adversarial workloads
