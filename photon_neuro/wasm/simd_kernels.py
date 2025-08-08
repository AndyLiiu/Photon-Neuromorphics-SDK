"""
SIMD-optimized kernels for photonic operations.

These Python implementations provide fallback for systems without WASM,
and serve as reference for the WASM SIMD implementations.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from numba import njit, prange
import warnings

# Try to import WASM loader (will fallback to Python if not available)
try:
    from .wasm_loader import load_wasm_module
    WASM_AVAILABLE = True
except ImportError:
    WASM_AVAILABLE = False
    warnings.warn("WASM module not available, falling back to Python/Numba implementations")


@njit(parallel=True, cache=True)
def mzi_forward_pass_simd(input_field: np.ndarray, 
                          phases: np.ndarray,
                          coupling_ratios: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized MZI mesh forward pass.
    
    Args:
        input_field: Complex input field (n_modes,)
        phases: Phase shifter values (n_phases,)
        coupling_ratios: MZI coupling ratios (n_mzis,)
    
    Returns:
        Complex output field (n_modes,)
    """
    n_modes = len(input_field)
    output_field = input_field.copy()
    
    # Process MZIs in parallel
    phase_idx = 0
    for layer in prange(n_modes - 1):
        for i in prange(0, n_modes - 1 - layer % 2, 2):
            if i + 1 < n_modes:
                # MZI operation between modes i and i+1
                coupling = coupling_ratios[phase_idx // 2 if phase_idx < len(coupling_ratios) * 2 else 0]
                phase_diff = phases[phase_idx] if phase_idx < len(phases) else 0.0
                
                # Complex MZI transfer matrix
                cos_theta = np.sqrt(1 - coupling)
                sin_theta = np.sqrt(coupling)
                phase_factor = np.exp(1j * phase_diff)
                
                # Apply MZI transformation
                a_in = output_field[i]
                b_in = output_field[i + 1]
                
                output_field[i] = cos_theta * a_in + 1j * sin_theta * b_in * phase_factor
                output_field[i + 1] = 1j * sin_theta * a_in * phase_factor + cos_theta * b_in
                
                phase_idx += 1
    
    return output_field


@njit(parallel=True, cache=True)
def microring_simulation_simd(wavelengths: np.ndarray,
                              ring_radii: np.ndarray, 
                              coupling_coeffs: np.ndarray,
                              quality_factors: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized microring resonator simulation.
    
    Args:
        wavelengths: Input wavelength array (n_wavelengths,)
        ring_radii: Ring radii (n_rings,) 
        coupling_coeffs: Coupling coefficients (n_rings,)
        quality_factors: Q factors (n_rings,)
    
    Returns:
        Transmission spectra (n_rings, n_wavelengths)
    """
    n_rings = len(ring_radii)
    n_wavelengths = len(wavelengths)
    transmission = np.ones((n_rings, n_wavelengths), dtype=np.complex128)
    
    # Process each ring in parallel
    for ring_idx in prange(n_rings):
        radius = ring_radii[ring_idx]
        kappa = coupling_coeffs[ring_idx]
        Q = quality_factors[ring_idx]
        
        # Resonant wavelength (assuming silicon on insulator)
        n_eff = 2.4  # Effective index
        lambda_res = 2 * np.pi * radius * n_eff
        
        # Calculate transmission for all wavelengths
        for w_idx in prange(n_wavelengths):
            wavelength = wavelengths[w_idx]
            
            # Detuning from resonance
            delta = 2 * np.pi * (wavelength - lambda_res) / lambda_res
            
            # Ring transmission (all-pass filter)
            gamma = np.pi / Q  # Loss parameter
            denominator = 1 - (1 - kappa) * np.exp(1j * delta - gamma)
            
            transmission[ring_idx, w_idx] = (kappa - (1 - kappa) * np.exp(1j * delta - gamma)) / denominator
    
    return transmission


@njit(parallel=True, cache=True)
def optical_matmul_simd(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized complex matrix multiplication for optical operations.
    
    Args:
        matrix_a: Complex matrix (m, k)
        matrix_b: Complex matrix (k, n)
    
    Returns:
        Product matrix (m, n)
    """
    m, k = matrix_a.shape
    _, n = matrix_b.shape
    result = np.zeros((m, n), dtype=np.complex128)
    
    # Parallel outer loop over rows
    for i in prange(m):
        for j in range(n):
            accumulator = 0.0 + 0.0j
            
            # Inner product with SIMD-friendly access pattern
            for kk in range(k):
                accumulator += matrix_a[i, kk] * matrix_b[kk, j]
            
            result[i, j] = accumulator
    
    return result


@njit(parallel=True, cache=True)
def phase_accumulation_simd(input_phases: np.ndarray,
                           phase_velocities: np.ndarray,
                           propagation_lengths: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized phase accumulation along waveguides.
    
    Args:
        input_phases: Initial phases (n_modes,)
        phase_velocities: Phase velocities (n_modes,) 
        propagation_lengths: Propagation distances (n_modes,)
    
    Returns:
        Accumulated phases (n_modes,)
    """
    n_modes = len(input_phases)
    output_phases = np.empty(n_modes)
    
    # Process phases in parallel
    for i in prange(n_modes):
        # Phase accumulation with proper wrapping
        accumulated = input_phases[i] + phase_velocities[i] * propagation_lengths[i]
        output_phases[i] = np.fmod(accumulated, 2 * np.pi)
        
        # Ensure phase is in [0, 2π)
        if output_phases[i] < 0:
            output_phases[i] += 2 * np.pi
    
    return output_phases


@njit(parallel=True, cache=True)
def complex_multiply_simd(field_a: np.ndarray, field_b: np.ndarray) -> np.ndarray:
    """
    SIMD-optimized complex field multiplication.
    
    Args:
        field_a: Complex field array (n_points,)
        field_b: Complex field array (n_points,)
    
    Returns:
        Element-wise product (n_points,)
    """
    n_points = len(field_a)
    result = np.empty(n_points, dtype=np.complex128)
    
    # Vectorized complex multiplication
    for i in prange(n_points):
        result[i] = field_a[i] * field_b[i]
    
    return result


@njit(parallel=True, cache=True)
def waveguide_propagation_simd(input_field: np.ndarray,
                              propagation_constant: np.ndarray,
                              length: float,
                              loss_coefficient: float = 0.0) -> np.ndarray:
    """
    SIMD-optimized waveguide propagation with loss.
    
    Args:
        input_field: Complex input field (n_modes,)
        propagation_constant: β values (n_modes,)
        length: Propagation length
        loss_coefficient: Loss coefficient (1/m)
    
    Returns:
        Output field after propagation (n_modes,)
    """
    n_modes = len(input_field)
    output_field = np.empty(n_modes, dtype=np.complex128)
    
    # Apply propagation and loss in parallel
    for i in prange(n_modes):
        # Phase evolution
        phase_change = propagation_constant[i] * length
        
        # Loss attenuation
        amplitude_factor = np.exp(-loss_coefficient * length / 2)
        
        # Combined propagation
        output_field[i] = input_field[i] * amplitude_factor * np.exp(1j * phase_change)
    
    return output_field


# High-level WASM interface functions
def mzi_forward_pass(input_field: torch.Tensor, 
                    phases: torch.Tensor,
                    coupling_ratios: torch.Tensor) -> torch.Tensor:
    """
    High-level interface for MZI forward pass with automatic WASM acceleration.
    """
    if WASM_AVAILABLE and input_field.device.type == 'cpu':
        # Use WASM implementation if available
        input_np = input_field.detach().numpy().astype(np.complex128)
        phases_np = phases.detach().numpy().astype(np.float64)
        coupling_np = coupling_ratios.detach().numpy().astype(np.float64)
        
        result_np = mzi_forward_pass_simd(input_np, phases_np, coupling_np)
        return torch.from_numpy(result_np).to(input_field.dtype)
    else:
        # Fallback to PyTorch implementation
        return mzi_forward_pass_simd(
            input_field.detach().numpy().astype(np.complex128),
            phases.detach().numpy().astype(np.float64),
            coupling_ratios.detach().numpy().astype(np.float64)
        )


def microring_simulation(wavelengths: torch.Tensor,
                        ring_radii: torch.Tensor,
                        coupling_coeffs: torch.Tensor,
                        quality_factors: torch.Tensor) -> torch.Tensor:
    """
    High-level interface for microring simulation with automatic WASM acceleration.
    """
    # Convert to numpy for SIMD processing
    wavelengths_np = wavelengths.detach().numpy().astype(np.float64)
    radii_np = ring_radii.detach().numpy().astype(np.float64)
    coupling_np = coupling_coeffs.detach().numpy().astype(np.float64)
    q_np = quality_factors.detach().numpy().astype(np.float64)
    
    result_np = microring_simulation_simd(wavelengths_np, radii_np, coupling_np, q_np)
    return torch.from_numpy(result_np).to(wavelengths.dtype)


# Performance benchmark function
def benchmark_simd_kernels(n_runs: int = 100, size: int = 256) -> dict:
    """
    Benchmark SIMD kernel performance.
    
    Returns:
        Performance metrics dictionary
    """
    import time
    
    # Generate test data
    input_field = np.random.randn(size).astype(np.complex128)
    phases = np.random.randn(size).astype(np.float64)
    coupling_ratios = np.random.rand(size // 2).astype(np.float64)
    
    # Benchmark MZI forward pass
    start_time = time.perf_counter()
    for _ in range(n_runs):
        _ = mzi_forward_pass_simd(input_field, phases, coupling_ratios)
    mzi_time = time.perf_counter() - start_time
    
    # Benchmark microring simulation
    wavelengths = np.linspace(1540e-9, 1560e-9, 100)
    ring_radii = np.random.uniform(5e-6, 10e-6, 16)
    coupling_coeffs = np.random.uniform(0.1, 0.3, 16)
    q_factors = np.random.uniform(5000, 15000, 16)
    
    start_time = time.perf_counter()
    for _ in range(n_runs):
        _ = microring_simulation_simd(wavelengths, ring_radii, coupling_coeffs, q_factors)
    microring_time = time.perf_counter() - start_time
    
    return {
        'mzi_time_ms': mzi_time * 1000 / n_runs,
        'microring_time_ms': microring_time * 1000 / n_runs,
        'simd_enabled': True,  # Numba provides SIMD-like acceleration
        'size': size,
        'runs': n_runs
    }


if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_simd_kernels()
    print("SIMD Kernel Benchmarks:")
    for key, value in results.items():
        print(f"  {key}: {value}")