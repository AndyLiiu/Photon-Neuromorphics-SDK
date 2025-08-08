"""
WebAssembly SIMD Module for Photonic Neural Networks
===================================================

High-performance WebAssembly acceleration for photonic computing operations
with SIMD optimization and JavaScript bindings.
"""

from .bindings import PhotonicWASM, initialize_wasm
from .simd_kernels import (
    mzi_forward_pass_simd, microring_simulation_simd,
    optical_matmul_simd, phase_accumulation_simd,
    complex_multiply_simd, waveguide_propagation_simd
)
from .js_interface import JSPhotonNeuro, export_wasm_module

__all__ = [
    "PhotonicWASM",
    "initialize_wasm", 
    "mzi_forward_pass_simd",
    "microring_simulation_simd",
    "optical_matmul_simd",
    "phase_accumulation_simd",
    "complex_multiply_simd", 
    "waveguide_propagation_simd",
    "JSPhotonNeuro",
    "export_wasm_module"
]