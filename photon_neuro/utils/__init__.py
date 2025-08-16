"""
Utility functions and helpers.
"""

from .math_utils import matrix_fidelity, random_unitary

# Simplified imports for Generation 5 - focus on core functionality
phase_unwrap = None
plot_optical_field = None
plot_transmission_spectrum = None
render_field = None
load_optical_data = None
save_calibration = None
normalize_complex = None
PhysicalConstants = None
MaterialProperties = None

__all__ = [
    "matrix_fidelity",
    "random_unitary", 
    "phase_unwrap",
    "plot_optical_field",
    "plot_transmission_spectrum",
    "render_field",
    "load_optical_data",
    "save_calibration",
    "normalize_complex",
    "PhysicalConstants",
    "MaterialProperties",
]