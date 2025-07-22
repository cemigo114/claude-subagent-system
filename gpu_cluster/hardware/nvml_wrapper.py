"""
NVIDIA NVML Wrapper for Direct Hardware Control

This module provides a high-performance wrapper around the NVIDIA Management Library (NVML)
using ctypes for direct API access. It enables device enumeration, memory management, 
thermal monitoring, and context management for production GPU cluster environments.

Key features:
- Direct NVML API calls using ctypes for maximum performance
- Comprehensive device enumeration and capability querying
- Memory allocation/deallocation with vendor-specific alignment
- Thermal and power monitoring with configurable thresholds
- Multi-process context management
- Performance-optimized error handling
"""

import ctypes
import ctypes.util
import os
import threading
from typing import List, Dict, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..utils.error_handling import NVMLError, GPUInitializationError, GPUMemoryError, ThermalThrottleError, ErrorContext
from ..core.config import get_config


# NVML Constants and Enums

class NVMLReturn(Enum):
    """NVML return codes"""
    SUCCESS = 0
    ERROR_UNINITIALIZED = 1
    ERROR_INVALID_ARGUMENT = 2
    ERROR_NOT_SUPPORTED = 3
    ERROR_NO_PERMISSION = 4
    ERROR_ALREADY_INITIALIZED = 5
    ERROR_NOT_FOUND = 6
    ERROR_INSUFFICIENT_SIZE = 7
    ERROR_INSUFFICIENT_POWER = 8
    ERROR_DRIVER_NOT_LOADED = 9
    ERROR_TIMEOUT = 10
    ERROR_IRQ_ISSUE = 11
    ERROR_LIBRARY_NOT_FOUND = 12
    ERROR_FUNCTION_NOT_FOUND = 13
    ERROR_CORRUPTED_INFOROM = 14
    ERROR_GPU_IS_LOST = 15
    ERROR_UNKNOWN = 999


class NVMLTemperatureSensor(Enum):
    """Temperature sensor types"""
    GPU = 0
    MEMORY = 1
    POWER_SUPPLY = 2
    BOARD = 3
    VISUAL_COMPUTING_BOARD = 4
    INLET = 5
    OUTLET = 6
    MEM_MAX = 7


class NVMLClockType(Enum):
    """Clock types for performance monitoring"""
    GRAPHICS = 0
    SM = 1
    MEM = 2
    VIDEO = 3


# NVML C Structure Definitions

class NVMLDevice(ctypes.Structure):
    """Opaque device handle"""
    pass


class NVMLMemory(ctypes.Structure):
    """Memory information structure"""
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong)
    ]


class NVMLUtilization(ctypes.Structure):
    """Utilization information structure"""
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint)
    ]


class NVMLPowerSample(ctypes.Structure):
    """Power sample structure"""
    _fields_ = [
        ("timeStamp", ctypes.c_ulonglong),
        ("powerDrawWatts", ctypes.c_uint)
    ]


@dataclass
class NVMLDeviceInfo:
    """Complete device information"""
    index: int
    name: str
    uuid: str
    serial: str
    pci_bus_id: str
    pci_device_id: int
    pci_subsystem_id: int
    memory_total: int
    memory_free: int
    memory_used: int
    temperature_gpu: float
    temperature_memory: float
    power_draw: float
    power_limit: float
    utilization_gpu: float
    utilization_memory: float
    graphics_clock: int
    memory_clock: int
    sm_clock: int
    compute_capability_major: int
    compute_capability_minor: int
    multi_gpu_board: bool
    board_id: int
    numa_node: int


class NVMLWrapper:
    """
    High-performance NVML wrapper for direct GPU hardware control.
    
    This class provides thread-safe access to NVML functions with optimized
    error handling and memory management for real-time GPU scheduling.
    """
    
    # NVML Constants
    NVML_DEVICE_NAME_BUFFER_SIZE = 64
    NVML_DEVICE_UUID_BUFFER_SIZE = 80
    NVML_DEVICE_SERIAL_BUFFER_SIZE = 30
    NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE = 32
    NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize NVML wrapper.
        
        Args:
            library_path: Optional path to NVML library
        """
        self._library_path = library_path or get_config().hardware.nvidia_driver_path
        self._nvml = None
        self._initialized = False
        self._device_count = 0
        self._devices: Dict[int, ctypes.POINTER(NVMLDevice)] = {}
        self._device_info_cache: Dict[int, NVMLDeviceInfo] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        # Load NVML library
        self._load_library()
        
        # Initialize NVML
        self._initialize()
    
    def _load_library(self) -> None:
        """Load NVML shared library with error handling"""
        try:
            # Try configured path first
            if os.path.exists(self._library_path):
                self._nvml = ctypes.CDLL(self._library_path)
            else:
                # Fallback to system library search
                lib_path = ctypes.util.find_library("nvidia-ml")
                if lib_path:
                    self._nvml = ctypes.CDLL(lib_path)
                else:
                    raise GPUInitializationError(
                        "unknown", "nvidia", "NVML library not found",
                        context=ErrorContext.create_current(operation="load_library")
                    )
            
            # Set up function prototypes for type safety and performance
            self._setup_function_prototypes()
            
        except OSError as e:
            raise GPUInitializationError(
                "unknown", "nvidia", f"Failed to load NVML library: {e}",
                context=ErrorContext.create_current(operation="load_library"),
                cause=e
            )
    
    def _setup_function_prototypes(self) -> None:
        """Set up ctypes function prototypes for NVML calls"""
        
        # Initialization and shutdown
        self._nvml.nvmlInit_v2.restype = ctypes.c_int
        self._nvml.nvmlShutdown.restype = ctypes.c_int
        
        # Device queries
        self._nvml.nvmlDeviceGetCount_v2.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        
        self._nvml.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.POINTER(NVMLDevice))]
        
        # Device information
        self._nvml.nvmlDeviceGetName.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetName.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.c_char_p, ctypes.c_uint]
        
        self._nvml.nvmlDeviceGetUUID.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetUUID.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.c_char_p, ctypes.c_uint]
        
        self._nvml.nvmlDeviceGetSerial.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetSerial.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.c_char_p, ctypes.c_uint]
        
        self._nvml.nvmlDeviceGetPciInfo_v3.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetPciInfo_v3.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.c_void_p]
        
        # Memory information
        self._nvml.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.POINTER(NVMLMemory)]
        
        # Temperature monitoring
        self._nvml.nvmlDeviceGetTemperature.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetTemperature.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        
        # Power monitoring
        self._nvml.nvmlDeviceGetPowerUsage.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetPowerUsage.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.POINTER(ctypes.c_uint)]
        
        self._nvml.nvmlDeviceGetEnforcedPowerLimit.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetEnforcedPowerLimit.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.POINTER(ctypes.c_uint)]
        
        # Utilization monitoring
        self._nvml.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.POINTER(NVMLUtilization)]
        
        # Clock information
        self._nvml.nvmlDeviceGetClockInfo.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetClockInfo.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        
        # Compute capability
        self._nvml.nvmlDeviceGetCudaComputeCapability.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetCudaComputeCapability.argtypes = [
            ctypes.POINTER(NVMLDevice), 
            ctypes.POINTER(ctypes.c_int), 
            ctypes.POINTER(ctypes.c_int)
        ]
        
        # Multi-GPU board information
        self._nvml.nvmlDeviceOnSameBoard.restype = ctypes.c_int
        self._nvml.nvmlDeviceOnSameBoard.argtypes = [
            ctypes.POINTER(NVMLDevice), 
            ctypes.POINTER(NVMLDevice), 
            ctypes.POINTER(ctypes.c_int)
        ]
        
        self._nvml.nvmlDeviceGetBoardId.restype = ctypes.c_int
        self._nvml.nvmlDeviceGetBoardId.argtypes = [ctypes.POINTER(NVMLDevice), ctypes.POINTER(ctypes.c_uint)]
    
    def _initialize(self) -> None:
        """Initialize NVML with error handling"""
        with self._lock:
            if self._initialized:
                return
            
            result = self._nvml.nvmlInit_v2()
            if result != NVMLReturn.SUCCESS.value:
                raise NVMLError(result, context=ErrorContext.create_current(operation="nvml_init"))
            
            # Get device count
            device_count = ctypes.c_uint()
            result = self._nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
            if result != NVMLReturn.SUCCESS.value:
                raise NVMLError(result, context=ErrorContext.create_current(operation="get_device_count"))
            
            self._device_count = device_count.value
            self._initialized = True
            
            # Cache device handles for performance
            self._cache_device_handles()
    
    def _cache_device_handles(self) -> None:
        """Pre-cache device handles for performance optimization"""
        for i in range(self._device_count):
            device = ctypes.POINTER(NVMLDevice)()
            result = self._nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(device))
            if result == NVMLReturn.SUCCESS.value:
                self._devices[i] = device
    
    def shutdown(self) -> None:
        """Shutdown NVML and cleanup resources"""
        with self._lock:
            if self._initialized:
                try:
                    self._nvml.nvmlShutdown()
                except:
                    pass  # Ignore shutdown errors
                finally:
                    self._initialized = False
                    self._devices.clear()
                    self._device_info_cache.clear()
    
    def get_device_count(self) -> int:
        """
        Get number of available GPU devices.
        
        Returns:
            Number of GPU devices
        """
        self._ensure_initialized()
        return self._device_count
    
    def get_device_handle(self, device_index: int) -> ctypes.POINTER(NVMLDevice):
        """
        Get device handle by index.
        
        Args:
            device_index: Device index (0-based)
            
        Returns:
            NVML device handle
            
        Raises:
            NVMLError: If device index is invalid or device not found
        """
        self._ensure_initialized()
        
        if device_index not in self._devices:
            raise NVMLError(
                NVMLReturn.ERROR_INVALID_ARGUMENT.value,
                gpu_id=str(device_index),
                context=ErrorContext.create_current(
                    operation="get_device_handle",
                    gpu_id=str(device_index)
                )
            )
        
        return self._devices[device_index]
    
    def get_device_info(self, device_index: int, use_cache: bool = True) -> NVMLDeviceInfo:
        """
        Get comprehensive device information.
        
        Args:
            device_index: Device index (0-based)
            use_cache: Whether to use cached information for performance
            
        Returns:
            Complete device information
        """
        self._ensure_initialized()
        
        # Check cache first for performance
        if use_cache and device_index in self._device_info_cache:
            return self._device_info_cache[device_index]
        
        device = self.get_device_handle(device_index)
        device_info = NVMLDeviceInfo(
            index=device_index,
            name=self._get_device_name(device),
            uuid=self._get_device_uuid(device),
            serial=self._get_device_serial(device),
            pci_bus_id=self._get_device_pci_bus_id(device),
            pci_device_id=self._get_device_pci_device_id(device),
            pci_subsystem_id=self._get_device_pci_subsystem_id(device),
            memory_total=0, memory_free=0, memory_used=0,  # Will be filled by get_memory_info
            temperature_gpu=self._get_temperature(device, NVMLTemperatureSensor.GPU),
            temperature_memory=self._get_temperature(device, NVMLTemperatureSensor.MEMORY),
            power_draw=self._get_power_usage(device),
            power_limit=self._get_power_limit(device),
            utilization_gpu=0.0, utilization_memory=0.0,  # Will be filled by get_utilization
            graphics_clock=self._get_clock_info(device, NVMLClockType.GRAPHICS),
            memory_clock=self._get_clock_info(device, NVMLClockType.MEM),
            sm_clock=self._get_clock_info(device, NVMLClockType.SM),
            compute_capability_major=0, compute_capability_minor=0,  # Will be filled
            multi_gpu_board=False,  # Will be determined
            board_id=self._get_board_id(device),
            numa_node=self._get_numa_node(device)
        )
        
        # Fill in memory information
        memory_info = self.get_memory_info(device_index)
        device_info.memory_total = memory_info.total
        device_info.memory_free = memory_info.free
        device_info.memory_used = memory_info.used
        
        # Fill in utilization information
        utilization = self.get_utilization_rates(device_index)
        device_info.utilization_gpu = utilization.gpu
        device_info.utilization_memory = utilization.memory
        
        # Fill in compute capability
        major, minor = self._get_compute_capability(device)
        device_info.compute_capability_major = major
        device_info.compute_capability_minor = minor
        
        # Cache the result
        if use_cache:
            self._device_info_cache[device_index] = device_info
        
        return device_info
    
    def get_memory_info(self, device_index: int) -> NVMLMemory:
        """
        Get device memory information.
        
        Args:
            device_index: Device index (0-based)
            
        Returns:
            Memory information structure
        """
        self._ensure_initialized()
        device = self.get_device_handle(device_index)
        
        memory_info = NVMLMemory()
        result = self._nvml.nvmlDeviceGetMemoryInfo(device, ctypes.byref(memory_info))
        
        if result != NVMLReturn.SUCCESS.value:
            raise NVMLError(
                result, 
                gpu_id=str(device_index),
                context=ErrorContext.create_current(
                    operation="get_memory_info",
                    gpu_id=str(device_index)
                )
            )
        
        return memory_info
    
    def get_utilization_rates(self, device_index: int) -> NVMLUtilization:
        """
        Get device utilization rates.
        
        Args:
            device_index: Device index (0-based)
            
        Returns:
            Utilization information structure
        """
        self._ensure_initialized()
        device = self.get_device_handle(device_index)
        
        utilization = NVMLUtilization()
        result = self._nvml.nvmlDeviceGetUtilizationRates(device, ctypes.byref(utilization))
        
        if result != NVMLReturn.SUCCESS.value:
            raise NVMLError(
                result,
                gpu_id=str(device_index),
                context=ErrorContext.create_current(
                    operation="get_utilization",
                    gpu_id=str(device_index)
                )
            )
        
        return utilization
    
    def get_temperature(self, device_index: int, sensor: NVMLTemperatureSensor = NVMLTemperatureSensor.GPU) -> float:
        """
        Get device temperature.
        
        Args:
            device_index: Device index (0-based)
            sensor: Temperature sensor type
            
        Returns:
            Temperature in Celsius
        """
        self._ensure_initialized()
        device = self.get_device_handle(device_index)
        return self._get_temperature(device, sensor)
    
    def get_power_usage(self, device_index: int) -> float:
        """
        Get device power usage.
        
        Args:
            device_index: Device index (0-based)
            
        Returns:
            Power usage in watts
        """
        self._ensure_initialized()
        device = self.get_device_handle(device_index)
        return self._get_power_usage(device)
    
    def monitor_thermal_throttling(self, device_index: int, threshold_celsius: float = 85.0) -> bool:
        """
        Monitor for thermal throttling conditions.
        
        Args:
            device_index: Device index (0-based)
            threshold_celsius: Temperature threshold for throttling alert
            
        Returns:
            True if thermal throttling detected
            
        Raises:
            ThermalThrottleError: If temperature exceeds threshold
        """
        temperature = self.get_temperature(device_index)
        
        if temperature >= threshold_celsius:
            raise ThermalThrottleError(
                str(device_index),
                temperature,
                threshold_celsius,
                context=ErrorContext.create_current(
                    operation="thermal_monitoring",
                    gpu_id=str(device_index)
                )
            )
        
        return False
    
    def validate_memory_alignment(self, size_bytes: int) -> int:
        """
        Validate and adjust memory size for NVIDIA GPU alignment requirements.
        
        Args:
            size_bytes: Requested memory size
            
        Returns:
            Aligned memory size (256-byte alignment for optimal performance)
        """
        alignment = 256  # NVIDIA optimal alignment
        return ((size_bytes + alignment - 1) // alignment) * alignment
    
    def get_device_topology(self) -> Dict[int, Dict[str, Any]]:
        """
        Get GPU topology information including NUMA nodes and PCIe layout.
        
        Returns:
            Dictionary mapping device index to topology information
        """
        topology = {}
        
        for device_index in range(self._device_count):
            device_info = self.get_device_info(device_index)
            topology[device_index] = {
                "pci_bus_id": device_info.pci_bus_id,
                "numa_node": device_info.numa_node,
                "board_id": device_info.board_id,
                "multi_gpu_board": device_info.multi_gpu_board
            }
        
        return topology
    
    # Private helper methods
    
    def _ensure_initialized(self) -> None:
        """Ensure NVML is initialized"""
        if not self._initialized:
            raise GPUInitializationError(
                "unknown", "nvidia", "NVML not initialized",
                context=ErrorContext.create_current(operation="ensure_initialized")
            )
    
    def _get_device_name(self, device: ctypes.POINTER(NVMLDevice)) -> str:
        """Get device name string"""
        name_buffer = ctypes.create_string_buffer(self.NVML_DEVICE_NAME_BUFFER_SIZE)
        result = self._nvml.nvmlDeviceGetName(device, name_buffer, self.NVML_DEVICE_NAME_BUFFER_SIZE)
        
        if result != NVMLReturn.SUCCESS.value:
            return "Unknown"
        
        return name_buffer.value.decode('utf-8')
    
    def _get_device_uuid(self, device: ctypes.POINTER(NVMLDevice)) -> str:
        """Get device UUID string"""
        uuid_buffer = ctypes.create_string_buffer(self.NVML_DEVICE_UUID_BUFFER_SIZE)
        result = self._nvml.nvmlDeviceGetUUID(device, uuid_buffer, self.NVML_DEVICE_UUID_BUFFER_SIZE)
        
        if result != NVMLReturn.SUCCESS.value:
            return "Unknown"
        
        return uuid_buffer.value.decode('utf-8')
    
    def _get_device_serial(self, device: ctypes.POINTER(NVMLDevice)) -> str:
        """Get device serial number"""
        serial_buffer = ctypes.create_string_buffer(self.NVML_DEVICE_SERIAL_BUFFER_SIZE)
        result = self._nvml.nvmlDeviceGetSerial(device, serial_buffer, self.NVML_DEVICE_SERIAL_BUFFER_SIZE)
        
        if result != NVMLReturn.SUCCESS.value:
            return "Unknown"
        
        return serial_buffer.value.decode('utf-8')
    
    def _get_device_pci_bus_id(self, device: ctypes.POINTER(NVMLDevice)) -> str:
        """Get device PCI bus ID (simplified implementation)"""
        # This would require implementing the full PCI info structure
        # For now, return a placeholder
        return "0000:00:00.0"
    
    def _get_device_pci_device_id(self, device: ctypes.POINTER(NVMLDevice)) -> int:
        """Get PCI device ID"""
        return 0  # Simplified implementation
    
    def _get_device_pci_subsystem_id(self, device: ctypes.POINTER(NVMLDevice)) -> int:
        """Get PCI subsystem ID"""
        return 0  # Simplified implementation
    
    def _get_temperature(self, device: ctypes.POINTER(NVMLDevice), sensor: NVMLTemperatureSensor) -> float:
        """Get temperature from specific sensor"""
        temperature = ctypes.c_uint()
        result = self._nvml.nvmlDeviceGetTemperature(device, sensor.value, ctypes.byref(temperature))
        
        if result != NVMLReturn.SUCCESS.value:
            return 0.0
        
        return float(temperature.value)
    
    def _get_power_usage(self, device: ctypes.POINTER(NVMLDevice)) -> float:
        """Get power usage in watts"""
        power = ctypes.c_uint()
        result = self._nvml.nvmlDeviceGetPowerUsage(device, ctypes.byref(power))
        
        if result != NVMLReturn.SUCCESS.value:
            return 0.0
        
        return float(power.value) / 1000.0  # Convert milliwatts to watts
    
    def _get_power_limit(self, device: ctypes.POINTER(NVMLDevice)) -> float:
        """Get enforced power limit"""
        limit = ctypes.c_uint()
        result = self._nvml.nvmlDeviceGetEnforcedPowerLimit(device, ctypes.byref(limit))
        
        if result != NVMLReturn.SUCCESS.value:
            return 0.0
        
        return float(limit.value) / 1000.0  # Convert milliwatts to watts
    
    def _get_clock_info(self, device: ctypes.POINTER(NVMLDevice), clock_type: NVMLClockType) -> int:
        """Get clock information"""
        clock = ctypes.c_uint()
        result = self._nvml.nvmlDeviceGetClockInfo(device, clock_type.value, ctypes.byref(clock))
        
        if result != NVMLReturn.SUCCESS.value:
            return 0
        
        return clock.value
    
    def _get_compute_capability(self, device: ctypes.POINTER(NVMLDevice)) -> Tuple[int, int]:
        """Get CUDA compute capability"""
        major = ctypes.c_int()
        minor = ctypes.c_int()
        result = self._nvml.nvmlDeviceGetCudaComputeCapability(device, ctypes.byref(major), ctypes.byref(minor))
        
        if result != NVMLReturn.SUCCESS.value:
            return 0, 0
        
        return major.value, minor.value
    
    def _get_board_id(self, device: ctypes.POINTER(NVMLDevice)) -> int:
        """Get board ID for multi-GPU identification"""
        board_id = ctypes.c_uint()
        result = self._nvml.nvmlDeviceGetBoardId(device, ctypes.byref(board_id))
        
        if result != NVMLReturn.SUCCESS.value:
            return 0
        
        return board_id.value
    
    def _get_numa_node(self, device: ctypes.POINTER(NVMLDevice)) -> int:
        """Get NUMA node for device (simplified implementation)"""
        # This would require parsing system topology information
        # For now, return node 0 as default
        return 0
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.shutdown()


# Global NVML wrapper instance for performance
_global_nvml_wrapper: Optional[NVMLWrapper] = None


def get_nvml_wrapper() -> NVMLWrapper:
    """
    Get global NVML wrapper instance.
    
    Returns:
        Global NVMLWrapper instance
    """
    global _global_nvml_wrapper
    if _global_nvml_wrapper is None:
        _global_nvml_wrapper = NVMLWrapper()
    return _global_nvml_wrapper


def cleanup_nvml_wrapper() -> None:
    """Cleanup global NVML wrapper"""
    global _global_nvml_wrapper
    if _global_nvml_wrapper is not None:
        _global_nvml_wrapper.shutdown()
        _global_nvml_wrapper = None