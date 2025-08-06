"""
Photon Neuromorphics SDK
========================

Silicon-photonic spiking neural network library with WebAssembly SIMD acceleration
and real-time optical training capabilities.
"""

__version__ = "0.1.0"
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

# Version info
VERSION_INFO = (0, 1, 0, "alpha")

def get_version():
    """Get the version string."""
    return __version__

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
]