"""
Photon Neuromorphics SDK
========================

Silicon-photonic spiking neural network library with WebAssembly SIMD acceleration
and real-time optical training capabilities.
"""

__version__ = "0.2.0-robust"
__author__ = "Daniel Schmidt"
__email__ = "daniel@photon-neuro.io"

# Core imports
from .core import PhotonicComponent, WaveguideBase, ModulatorBase
from .networks import PhotonicSNN, MZIMesh, MicroringArray
from .simulation import PhotonicSimulator, NoiseSimulator, PowerBudgetAnalyzer
from .compiler import compile_to_photonic, ONNXParser
from .training import OpticalAdam, OpticalTrainer, spike_loss, mse_loss
from .hardware import PhotonicChip, HardwareCalibrator
from .utils import matrix_fidelity, random_unitary

# Component registration decorator
from .core.registry import register_component

# Version info - Generation 2 "Make It Robust"
VERSION_INFO = (0, 2, 0, "robust")

def get_version():
    """Get the version string."""
    return __version__

# Import WebAssembly acceleration
from .wasm import (
    PhotonicWASM, initialize_wasm, mzi_forward_pass_simd, microring_simulation_simd,
    optical_matmul_simd, phase_accumulation_simd, complex_multiply_simd,
    waveguide_propagation_simd, JSPhotonNeuro, export_wasm_module
)

# Import robust features
from .core.exceptions import (
    PhotonicError, SimulationError, ComponentError, HardwareError,
    CalibrationError, PowerBudgetError, ValidationError, ConvergenceError,
    ThermalError, NoiseModelError, DataIntegrityError, ErrorRecovery,
    validate_parameter, check_tensor_validity, check_array_validity,
    safe_execution, global_error_recovery
)

from .simulation.noise import (
    QuantumNoiseModel, CrosstalkModel, TemperatureDependentNoise,
    WaveguideCrosstalk
)

from .simulation.thermal import (
    ThermalSimulator, CoolingSystem, ThermalOpticalCoupling,
    calculate_thermal_time_constant, estimate_junction_temperature
)

from .simulation.circuit_analysis import (
    SParameterParser, MultiModeWaveguide, PolarizationModel,
    DispersionModel, NetworkAnalyzer
)

from .hardware.calibration import (
    CalibrationManager, PIDCalibrator, MLCalibrator, SafetyChecker,
    RealTimeErrorCorrector, CalibrationDatabase, global_calibration_manager
)

from .utils.logging_system import (
    PhotonLogger, MetricsCollector, ProgressTracker, DiagnosticMode,
    global_logger, log_execution_time, monitor_memory_usage, track_progress
)

__all__ = [
    # Core components
    "PhotonicComponent",
    "WaveguideBase", 
    "ModulatorBase",
    
    # Networks
    "PhotonicSNN",
    "MZIMesh",
    "MicroringArray",
    
    # Simulation
    "PhotonicSimulator",
    "NoiseSimulator", 
    "PowerBudgetAnalyzer",
    
    # Compiler
    "compile_to_photonic",
    "ONNXParser",
    
    # Training
    "OpticalAdam",
    "OpticalTrainer",
    "spike_loss",
    "mse_loss",
    
    # Hardware
    "PhotonicChip",
    "HardwareCalibrator",
    
    # Utilities
    "matrix_fidelity",
    "random_unitary",
    "register_component",
    
    # Version
    "get_version",
    "__version__",
    
    # WebAssembly Acceleration
    "PhotonicWASM", "initialize_wasm", "mzi_forward_pass_simd", "microring_simulation_simd",
    "optical_matmul_simd", "phase_accumulation_simd", "complex_multiply_simd", 
    "waveguide_propagation_simd", "JSPhotonNeuro", "export_wasm_module",
    
    # Robust Features - Error Handling
    "PhotonicError", "SimulationError", "ComponentError", "HardwareError",
    "CalibrationError", "PowerBudgetError", "ValidationError", "ConvergenceError",
    "ThermalError", "NoiseModelError", "DataIntegrityError", "ErrorRecovery",
    "validate_parameter", "check_tensor_validity", "check_array_validity",
    "safe_execution", "global_error_recovery",
    
    # Robust Features - Advanced Noise Modeling
    "QuantumNoiseModel", "CrosstalkModel", "TemperatureDependentNoise", "WaveguideCrosstalk",
    
    # Robust Features - Thermal Analysis
    "ThermalSimulator", "CoolingSystem", "ThermalOpticalCoupling",
    "calculate_thermal_time_constant", "estimate_junction_temperature",
    
    # Robust Features - Circuit Analysis
    "SParameterParser", "MultiModeWaveguide", "PolarizationModel", 
    "DispersionModel", "NetworkAnalyzer",
    
    # Robust Features - Calibration System
    "CalibrationManager", "PIDCalibrator", "MLCalibrator", "SafetyChecker",
    "RealTimeErrorCorrector", "CalibrationDatabase", "global_calibration_manager",
    
    # Robust Features - Logging & Monitoring
    "PhotonLogger", "MetricsCollector", "ProgressTracker", "DiagnosticMode",
    "global_logger", "log_execution_time", "monitor_memory_usage", "track_progress",
]