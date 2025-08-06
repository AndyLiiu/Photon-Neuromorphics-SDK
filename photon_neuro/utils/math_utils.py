"""
Mathematical utilities for photonic systems.
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional
import scipy.linalg


def matrix_fidelity(matrix1: Union[torch.Tensor, np.ndarray], 
                   matrix2: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate fidelity between two unitary matrices."""
    
    # Convert to numpy if needed
    if isinstance(matrix1, torch.Tensor):
        matrix1 = matrix1.detach().numpy()
    if isinstance(matrix2, torch.Tensor):
        matrix2 = matrix2.detach().numpy()
        
    # Ensure complex type
    if not np.iscomplexobj(matrix1):
        matrix1 = matrix1.astype(complex)
    if not np.iscomplexobj(matrix2):
        matrix2 = matrix2.astype(complex)
        
    # Calculate fidelity: F = |Tr(U1† U2)| / n
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Matrix shapes don't match: {matrix1.shape} vs {matrix2.shape}")
        
    n = matrix1.shape[0]
    trace = np.trace(matrix1.conj().T @ matrix2)
    fidelity = abs(trace) / n
    
    return float(np.real(fidelity))


def random_unitary(size: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate random unitary matrix using Haar measure."""
    
    if seed is not None:
        np.random.seed(seed)
        
    # Generate random complex Gaussian matrix
    real_part = np.random.randn(size, size)
    imag_part = np.random.randn(size, size)
    gaussian_matrix = real_part + 1j * imag_part
    
    # QR decomposition
    Q, R = np.linalg.qr(gaussian_matrix)
    
    # Fix phases to get Haar-distributed unitary
    diagonal = np.diagonal(R)
    phases = diagonal / np.abs(diagonal)
    Q = Q * phases
    
    return torch.tensor(Q, dtype=torch.complex64)


def phase_unwrap(phases: Union[torch.Tensor, np.ndarray], 
                discont: float = np.pi) -> Union[torch.Tensor, np.ndarray]:
    """Unwrap phase array to remove discontinuities."""
    
    is_torch = isinstance(phases, torch.Tensor)
    
    if is_torch:
        phases_np = phases.detach().numpy()
        device = phases.device
        dtype = phases.dtype
    else:
        phases_np = phases
        
    # Use numpy's unwrap function
    unwrapped = np.unwrap(phases_np, discont=discont)
    
    if is_torch:
        return torch.tensor(unwrapped, device=device, dtype=dtype)
    else:
        return unwrapped


def gaussian_beam_profile(x: np.ndarray, y: np.ndarray, 
                         beam_waist: float, wavelength: float,
                         z: float = 0) -> np.ndarray:
    """Calculate Gaussian beam intensity profile."""
    
    # Rayleigh range
    z_R = np.pi * beam_waist**2 / wavelength
    
    # Beam waist at distance z
    w_z = beam_waist * np.sqrt(1 + (z / z_R)**2)
    
    # Radius of curvature
    if z != 0:
        R_z = z * (1 + (z_R / z)**2)
    else:
        R_z = np.inf
        
    # Gouy phase
    gouy_phase = np.arctan(z / z_R)
    
    # Radial distance
    X, Y = np.meshgrid(x, y)
    r_sq = X**2 + Y**2
    
    # Beam profile
    amplitude = (beam_waist / w_z) * np.exp(-r_sq / w_z**2)
    
    # Phase (simplified - not including all terms)
    k = 2 * np.pi / wavelength
    if R_z != np.inf:
        phase = k * r_sq / (2 * R_z) - gouy_phase
    else:
        phase = -gouy_phase
        
    return amplitude * np.exp(1j * phase)


def mode_overlap_integral(mode1: np.ndarray, mode2: np.ndarray,
                         dx: float, dy: float) -> complex:
    """Calculate overlap integral between two optical modes."""
    
    # Normalize modes
    norm1 = np.sqrt(np.sum(np.abs(mode1)**2) * dx * dy)
    norm2 = np.sqrt(np.sum(np.abs(mode2)**2) * dx * dy)
    
    mode1_norm = mode1 / (norm1 + 1e-12)
    mode2_norm = mode2 / (norm2 + 1e-12)
    
    # Calculate overlap
    overlap = np.sum(np.conj(mode1_norm) * mode2_norm) * dx * dy
    
    return complex(overlap)


def fresnel_coefficients(n1: complex, n2: complex, theta_i: float,
                        polarization: str = 'TE') -> Tuple[complex, complex]:
    """Calculate Fresnel reflection and transmission coefficients."""
    
    theta_i = np.array(theta_i)
    
    # Snell's law
    sin_theta_t = (n1 / n2) * np.sin(theta_i)
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    cos_theta_i = np.cos(theta_i)
    
    if polarization.upper() == 'TE':
        # TE polarization (s-polarized)
        r = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
        t = (2 * n1 * cos_theta_i) / (n1 * cos_theta_i + n2 * cos_theta_t)
    elif polarization.upper() == 'TM':
        # TM polarization (p-polarized)
        r = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
        t = (2 * n1 * cos_theta_i) / (n2 * cos_theta_i + n1 * cos_theta_t)
    else:
        raise ValueError("Polarization must be 'TE' or 'TM'")
        
    return complex(r), complex(t)


def transfer_matrix(thickness: float, n_eff: complex, wavelength: float) -> np.ndarray:
    """Calculate transfer matrix for a layer."""
    
    k0 = 2 * np.pi / wavelength
    beta = k0 * n_eff
    
    # Transfer matrix
    phi = beta * thickness
    
    T = np.array([
        [np.cos(phi), -1j * np.sin(phi) / n_eff],
        [-1j * n_eff * np.sin(phi), np.cos(phi)]
    ], dtype=complex)
    
    return T


def cavity_resonances(cavity_length: float, n_eff: complex, wavelength_range: np.ndarray,
                     mirror_r1: float = 1.0, mirror_r2: float = 1.0) -> np.ndarray:
    """Find cavity resonance wavelengths."""
    
    resonances = []
    
    for wavelength in wavelength_range:
        # Round-trip phase
        k0 = 2 * np.pi / wavelength
        beta = k0 * n_eff
        round_trip_phase = 2 * beta * cavity_length
        
        # Cavity condition: round_trip_phase = m * 2π
        phase_mod_2pi = np.angle(np.exp(1j * round_trip_phase))
        
        # Check if close to resonance
        if abs(phase_mod_2pi) < 0.1 or abs(phase_mod_2pi - 2*np.pi) < 0.1:
            # Calculate finesse and Q factor
            loss = np.imag(beta)
            finesse = np.pi / (2 * loss * cavity_length + 1e-12)
            
            if finesse > 1:  # Only significant resonances
                resonances.append(wavelength)
                
    return np.array(resonances)


def coupling_matrix(waveguides: list, coupling_lengths: np.ndarray,
                   coupling_coefficients: np.ndarray) -> np.ndarray:
    """Calculate coupling matrix for coupled waveguides."""
    
    n_guides = len(waveguides)
    C = np.zeros((n_guides, n_guides), dtype=complex)
    
    # Diagonal elements (self-coupling)
    for i, wg in enumerate(waveguides):
        if hasattr(wg, 'n_eff'):
            C[i, i] = wg.n_eff
        else:
            C[i, i] = 2.4  # Default effective index
            
    # Off-diagonal elements (coupling between guides)
    for i in range(n_guides):
        for j in range(i+1, n_guides):
            if i < len(coupling_coefficients) and j-1 < len(coupling_coefficients):
                coupling_idx = min(i, j-1)
                if coupling_idx < len(coupling_coefficients):
                    C[i, j] = coupling_coefficients[coupling_idx]
                    C[j, i] = coupling_coefficients[coupling_idx]
                    
    return C


def beam_propagation_method(initial_field: np.ndarray, z_step: float, n_steps: int,
                           refractive_index_profile: np.ndarray, 
                           wavelength: float, dx: float) -> np.ndarray:
    """Simple beam propagation method simulation."""
    
    k0 = 2 * np.pi / wavelength
    
    # Initialize field array
    nx = initial_field.shape[0]
    field_evolution = np.zeros((n_steps, nx), dtype=complex)
    field_evolution[0] = initial_field
    
    current_field = initial_field.copy()
    
    for step in range(1, n_steps):
        # Apply phase due to refractive index variation
        phase = k0 * refractive_index_profile * z_step
        current_field *= np.exp(1j * phase)
        
        # Apply diffraction (simplified using FFT)
        field_fft = np.fft.fft(current_field)
        k_x = np.fft.fftfreq(nx, dx) * 2 * np.pi
        
        # Diffraction phase
        diffraction_phase = -1j * k_x**2 * z_step / (2 * k0)
        field_fft *= np.exp(diffraction_phase)
        
        # Transform back
        current_field = np.fft.ifft(field_fft)
        
        field_evolution[step] = current_field
        
    return field_evolution


def sellmeier_equation(wavelength_um: float, material: str) -> float:
    """Calculate refractive index using Sellmeier equation."""
    
    # Material parameters (wavelength in micrometers)
    coefficients = {
        'silicon': {
            'B': [10.6684293, 0.0030434748, 1.54133408],
            'C': [0.301516485**2, 1.13475115**2, 1104**2]
        },
        'silica': {
            'B': [0.6961663, 0.4079426, 0.8974794],
            'C': [0.0684043**2, 0.1162414**2, 9.896161**2]
        },
        'silicon_nitride': {
            'B': [2.8939, 1.0, 0.0],
            'C': [0.13967**2, 1.0, 0.0]
        }
    }
    
    if material.lower() not in coefficients:
        raise ValueError(f"Unknown material: {material}")
        
    params = coefficients[material.lower()]
    B = params['B']
    C = params['C']
    
    # Sellmeier formula: n² = 1 + Σ(Bi * λ²) / (λ² - Ci)
    lambda_sq = wavelength_um**2
    n_squared = 1.0
    
    for i in range(len(B)):
        if C[i] != 0:  # Avoid division by zero
            n_squared += B[i] * lambda_sq / (lambda_sq - C[i])
    
    return np.sqrt(n_squared)


def effective_index_slab(wavelength: float, core_thickness: float,
                        n_core: float, n_cladding: float) -> float:
    """Calculate effective index of slab waveguide."""
    
    k0 = 2 * np.pi / wavelength
    
    # Normalized frequency
    V = k0 * core_thickness * np.sqrt(n_core**2 - n_cladding**2)
    
    # Approximate solution for fundamental mode
    if V < np.pi/2:
        # Below cutoff
        return n_cladding
    else:
        # Above cutoff - use approximation
        b = 1 - (np.pi / (2 * V))**2
        n_eff = n_cladding + (n_core - n_cladding) * b
        return n_eff