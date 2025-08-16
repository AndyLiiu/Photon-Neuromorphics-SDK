"""
Hardware interface and calibration tools.
"""

from .interface import PhotonicChip, HardwareInterface

# Simplified imports for Generation 5 - focus on core functionality
HardwareCalibrator = None
CalibrationData = None
VectorNetworkAnalyzer = None
OpticalSpectrumAnalyzer = None
PowerMeter = None

__all__ = [
    "PhotonicChip",
    "HardwareInterface",
    "HardwareCalibrator", 
    "CalibrationData",
    "VectorNetworkAnalyzer",
    "OpticalSpectrumAnalyzer",
    "PowerMeter",
]