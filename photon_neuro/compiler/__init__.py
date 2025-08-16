"""
Model compilation and optimization tools.
"""

from .onnx_compiler import ONNXParser, compile_to_photonic

# Simplified imports for Generation 5 - focus on core functionality
PhotonicOptimizer = None
LayoutOptimizer = None
PlaceAndRoute = None
PhotonicRouter = None

__all__ = [
    "ONNXParser",
    "compile_to_photonic", 
    "PhotonicOptimizer",
    "LayoutOptimizer",
    "PlaceAndRoute",
    "PhotonicRouter",
]