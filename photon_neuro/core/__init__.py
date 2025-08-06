"""
Core photonic components and base classes.
"""

from .components import PhotonicComponent, WaveguideBase, ModulatorBase, DetectorBase
from .waveguides import SiliconWaveguide, NitridWaveguide
from .modulators import MachZehnderModulator, MicroringModulator, PhaseShifter
from .detectors import Photodetector, PhotodetectorArray
from .sources import LaserSource, LEDSource
from .registry import register_component, ComponentRegistry

__all__ = [
    "PhotonicComponent",
    "WaveguideBase",
    "ModulatorBase", 
    "DetectorBase",
    "SiliconWaveguide",
    "NitridWaveguide",
    "MachZehnderModulator",
    "MicroringModulator", 
    "PhaseShifter",
    "Photodetector",
    "PhotodetectorArray",
    "LaserSource",
    "LEDSource",
    "register_component",
    "ComponentRegistry",
]