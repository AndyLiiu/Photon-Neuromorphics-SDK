"""
Advanced circuit-level simulation and analysis for photonic systems.

This module provides:
- S-parameter file parsing and interpolation
- Multi-mode waveguide analysis  
- Polarization-dependent loss modeling
- Dispersion effects modeling
- Network analysis tools
"""

import numpy as np
import torch
import scipy.interpolate
import scipy.signal
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

from ..core.exceptions import (
    SimulationError, ValidationError, DataIntegrityError,
    validate_parameter, check_array_validity, safe_execution,
    global_error_recovery
)


class SParameterParser:
    """Parse and interpolate S-parameter data from various file formats."""
    
    def __init__(self):
        self.frequency_unit = 'Hz'
        self.parameter_format = 'MA'  # Magnitude-Angle, DB-Angle, RI (Real-Imaginary)
        self.reference_impedance = 50.0  # Ohms
        self.data = {}
        self.frequencies = None
        
    @safe_execution(fallback_value=False)
    def load_touchstone_file(self, filepath: str) -> bool:
        """Load S-parameters from Touchstone (.s2p, .s4p, etc.) file."""
        try:
            filepath = validate_parameter("filepath", filepath, expected_type=str)
            
            if not os.path.exists(filepath):
                raise ValidationError(f"S-parameter file not found: {filepath}")
            
            # Determine number of ports from file extension
            file_ext = Path(filepath).suffix.lower()
            if file_ext.startswith('.s'):
                n_ports = int(file_ext[2:]) if file_ext[2:].isdigit() else 2
            else:
                raise ValidationError(f"Unsupported file format: {file_ext}")
                
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Parse header information
            data_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    self._parse_header_line(line)
                elif line and not line.startswith('!'):
                    data_lines.append(line)
            
            # Parse data
            self._parse_touchstone_data(data_lines, n_ports)
            
            return True
            
        except Exception as e:
            raise SimulationError(f"Failed to load Touchstone file: {e}")
    
    def _parse_header_line(self, line: str):
        """Parse header line from Touchstone file."""
        parts = line[1:].split()  # Remove '#' and split
        
        if len(parts) >= 5:
            self.frequency_unit = parts[0].upper()
            self.parameter_format = parts[1].upper()
            self.reference_impedance = float(parts[4])
    
    def _parse_touchstone_data(self, data_lines: List[str], n_ports: int):
        """Parse data section of Touchstone file."""
        try:
            frequencies = []
            s_matrices = []
            
            n_params = n_ports * n_ports
            values_per_param = 2  # Real/Imag or Mag/Phase
            
            i = 0
            while i < len(data_lines):
                # Parse frequency
                parts = data_lines[i].split()
                freq = float(parts[0])
                
                # Convert frequency to Hz
                freq_hz = self._convert_frequency_to_hz(freq)
                frequencies.append(freq_hz)
                
                # Parse S-parameters for this frequency
                s_matrix = np.zeros((n_ports, n_ports), dtype=complex)
                param_values = []
                
                # Collect all parameter values (may span multiple lines)
                remaining_values = n_params * values_per_param
                j = 1  # Start after frequency
                
                while remaining_values > 0 and i < len(data_lines):
                    if j >= len(parts):
                        i += 1
                        if i < len(data_lines):
                            parts = data_lines[i].split()
                            j = 0
                        continue
                    
                    param_values.append(float(parts[j]))
                    j += 1
                    remaining_values -= 1
                
                # Convert parameter values to complex S-matrix
                self._fill_s_matrix(s_matrix, param_values, n_ports)
                s_matrices.append(s_matrix)
                
                i += 1
            
            self.frequencies = np.array(frequencies)
            self.s_parameters = np.array(s_matrices)
            
        except Exception as e:
            raise SimulationError(f"Failed to parse S-parameter data: {e}")
    
    def _convert_frequency_to_hz(self, freq: float) -> float:
        """Convert frequency to Hz based on unit."""
        conversions = {
            'HZ': 1.0,
            'KHZ': 1e3,
            'MHZ': 1e6,
            'GHZ': 1e9,
            'THZ': 1e12
        }
        return freq * conversions.get(self.frequency_unit, 1.0)
    
    def _fill_s_matrix(self, s_matrix: np.ndarray, values: List[float], n_ports: int):
        """Fill S-parameter matrix from parsed values."""
        try:
            idx = 0
            for i in range(n_ports):
                for j in range(n_ports):
                    if self.parameter_format == 'MA':
                        # Magnitude and Angle (degrees)
                        magnitude = values[idx]
                        angle_deg = values[idx + 1]
                        s_matrix[i, j] = magnitude * np.exp(1j * np.deg2rad(angle_deg))
                    elif self.parameter_format == 'DB':
                        # dB and Angle (degrees)
                        magnitude_db = values[idx]
                        angle_deg = values[idx + 1]
                        magnitude = 10**(magnitude_db / 20)
                        s_matrix[i, j] = magnitude * np.exp(1j * np.deg2rad(angle_deg))
                    elif self.parameter_format == 'RI':
                        # Real and Imaginary
                        real_part = values[idx]
                        imag_part = values[idx + 1]
                        s_matrix[i, j] = complex(real_part, imag_part)
                    
                    idx += 2
                    
        except Exception as e:
            raise SimulationError(f"Failed to fill S-parameter matrix: {e}")
    
    @safe_execution(fallback_value=None)
    def interpolate_s_parameters(self, target_frequencies: np.ndarray, 
                               method: str = 'linear') -> np.ndarray:
        """Interpolate S-parameters to target frequencies."""
        try:
            if self.frequencies is None or not hasattr(self, 's_parameters'):
                raise SimulationError("No S-parameter data loaded")
            
            target_frequencies = check_array_validity(target_frequencies, "target_frequencies")
            method = validate_parameter("method", method, expected_type=str, 
                                      valid_values=['linear', 'cubic', 'spline'])
            
            n_ports = self.s_parameters.shape[1]
            n_freq_out = len(target_frequencies)
            
            interpolated_s = np.zeros((n_freq_out, n_ports, n_ports), dtype=complex)
            
            # Interpolate each S-parameter separately
            for i in range(n_ports):
                for j in range(n_ports):
                    s_ij = self.s_parameters[:, i, j]
                    
                    # Interpolate magnitude and phase separately for better stability
                    magnitude = np.abs(s_ij)
                    phase = np.unwrap(np.angle(s_ij))
                    
                    if method == 'linear':
                        mag_interp = np.interp(target_frequencies, self.frequencies, magnitude)
                        phase_interp = np.interp(target_frequencies, self.frequencies, phase)
                    else:
                        # Use scipy for more advanced interpolation
                        mag_func = scipy.interpolate.interp1d(
                            self.frequencies, magnitude, kind=method, 
                            bounds_error=False, fill_value='extrapolate'
                        )
                        phase_func = scipy.interpolate.interp1d(
                            self.frequencies, phase, kind=method,
                            bounds_error=False, fill_value='extrapolate'
                        )
                        
                        mag_interp = mag_func(target_frequencies)
                        phase_interp = phase_func(target_frequencies)
                    
                    interpolated_s[:, i, j] = mag_interp * np.exp(1j * phase_interp)
            
            return interpolated_s
            
        except Exception as e:
            raise SimulationError(f"S-parameter interpolation failed: {e}")


class MultiModeWaveguide:
    """Multi-mode waveguide analysis with mode coupling."""
    
    def __init__(self, width: float, height: float, material: str = "silicon",
                 wavelength: float = 1550e-9):
        try:
            self.width = validate_parameter("width", width, 
                                          expected_type=(int, float), valid_range=(100e-9, 10e-6))
            self.height = validate_parameter("height", height,
                                           expected_type=(int, float), valid_range=(100e-9, 10e-6))
            self.wavelength = validate_parameter("wavelength", wavelength,
                                               expected_type=(int, float), valid_range=(1e-6, 10e-6))
            
            # Material properties
            self.materials = {
                'silicon': {'n_core': 3.5, 'n_clad': 1.444},
                'silicon_nitride': {'n_core': 2.0, 'n_clad': 1.444},
                'silica': {'n_core': 1.468, 'n_clad': 1.444}
            }
            
            if material not in self.materials:
                raise ValidationError(f"Unknown material: {material}")
                
            self.material = material
            self.n_core = self.materials[material]['n_core']
            self.n_clad = self.materials[material]['n_clad']
            
            # Calculate supported modes
            self.modes = self._calculate_supported_modes()
            
        except ValidationError as e:
            raise SimulationError(f"Invalid waveguide parameters: {e}")
    
    def _calculate_supported_modes(self) -> List[Dict[str, Any]]:
        """Calculate supported waveguide modes."""
        try:
            k0 = 2 * np.pi / self.wavelength
            
            # Normalized frequency parameters
            V_x = k0 * self.width * np.sqrt(self.n_core**2 - self.n_clad**2) / 2
            V_y = k0 * self.height * np.sqrt(self.n_core**2 - self.n_clad**2) / 2
            
            modes = []
            mode_id = 0
            
            # Find TE and TM modes
            for m in range(10):  # Check first 10 modes in each direction
                for n in range(10):
                    # TE mode cutoff condition
                    V_mn_te = np.sqrt((m * np.pi / 2)**2 + (n * np.pi / 2)**2)
                    
                    if V_mn_te < V_x and V_mn_te < V_y:
                        # Calculate effective index
                        gamma_x = np.sqrt((k0 * self.n_core)**2 - (m * np.pi / self.width)**2)
                        gamma_y = np.sqrt((k0 * self.n_core)**2 - (n * np.pi / self.height)**2)
                        beta = np.sqrt(gamma_x**2 + gamma_y**2 - (k0 * self.n_clad)**2)
                        
                        if beta > 0:  # Guided mode
                            n_eff = beta / k0
                            
                            mode = {
                                'id': mode_id,
                                'type': 'TE',
                                'm': m,
                                'n': n,
                                'n_eff': n_eff,
                                'beta': beta,
                                'cutoff_frequency': V_mn_te * 3e8 / (2 * np.pi * np.sqrt(self.n_core**2 - self.n_clad**2))
                            }
                            modes.append(mode)
                            mode_id += 1
                    
                    # TM mode (similar calculation with different boundary conditions)
                    if m > 0 and n > 0:  # TM modes start from TM11
                        V_mn_tm = np.sqrt(((m * np.pi) / 2)**2 + ((n * np.pi) / 2)**2)
                        
                        if V_mn_tm < V_x and V_mn_tm < V_y:
                            gamma_x = np.sqrt((k0 * self.n_core)**2 - (m * np.pi / self.width)**2)
                            gamma_y = np.sqrt((k0 * self.n_core)**2 - (n * np.pi / self.height)**2)
                            beta = np.sqrt(gamma_x**2 + gamma_y**2 - (k0 * self.n_clad)**2)
                            
                            if beta > 0:
                                n_eff = beta / k0
                                
                                mode = {
                                    'id': mode_id,
                                    'type': 'TM',
                                    'm': m,
                                    'n': n,
                                    'n_eff': n_eff,
                                    'beta': beta,
                                    'cutoff_frequency': V_mn_tm * 3e8 / (2 * np.pi * np.sqrt(self.n_core**2 - self.n_clad**2))
                                }
                                modes.append(mode)
                                mode_id += 1
            
            # Sort modes by effective index (highest first)
            modes.sort(key=lambda x: x['n_eff'], reverse=True)
            
            return modes
            
        except Exception as e:
            raise SimulationError(f"Mode calculation failed: {e}")
    
    def get_mode_field_profile(self, mode_id: int, x_points: int = 50, 
                             y_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate mode field profile."""
        try:
            if mode_id >= len(self.modes):
                raise ValidationError(f"Mode ID {mode_id} out of range")
            
            mode = self.modes[mode_id]
            
            # Create coordinate grids
            x = np.linspace(-self.width/2, self.width/2, x_points)
            y = np.linspace(-self.height/2, self.height/2, y_points)
            X, Y = np.meshgrid(x, y)
            
            # Calculate field profile (simplified analytical approximation)
            m, n = mode['m'], mode['n']
            
            if mode['type'] == 'TE':
                # TE mode field pattern
                Ex = 0  # TE modes have no Ex component
                Ey = np.sin((m + 1) * np.pi * (X + self.width/2) / self.width) * \
                     np.cos(n * np.pi * (Y + self.height/2) / self.height)
                Ez = 0
                
            else:  # TM mode
                # TM mode field pattern  
                Ex = np.cos(m * np.pi * (X + self.width/2) / self.width) * \
                     np.sin((n + 1) * np.pi * (Y + self.height/2) / self.height)
                Ey = np.sin((m + 1) * np.pi * (X + self.width/2) / self.width) * \
                     np.cos(n * np.pi * (Y + self.height/2) / self.height)
                Ez = 0
            
            # Apply boundary conditions (zero field outside waveguide)
            mask = (np.abs(X) <= self.width/2) & (np.abs(Y) <= self.height/2)
            Ex = Ex * mask
            Ey = Ey * mask
            Ez = Ez * mask
            
            return X, Y, np.sqrt(Ex**2 + Ey**2 + Ez**2)
            
        except Exception as e:
            raise SimulationError(f"Mode field profile calculation failed: {e}")
    
    def calculate_mode_coupling(self, mode1_id: int, mode2_id: int, 
                              perturbation_strength: float = 1e-3) -> float:
        """Calculate coupling coefficient between two modes."""
        try:
            if mode1_id >= len(self.modes) or mode2_id >= len(self.modes):
                raise ValidationError("Mode IDs out of range")
            
            mode1 = self.modes[mode1_id]
            mode2 = self.modes[mode2_id]
            
            # Phase matching factor
            delta_beta = abs(mode1['beta'] - mode2['beta'])
            
            # Coupling strength (simplified model)
            # In practice, this would involve overlap integrals of mode fields
            coupling_coefficient = perturbation_strength * np.exp(-delta_beta * 1e-6)
            
            return coupling_coefficient
            
        except Exception as e:
            raise SimulationError(f"Mode coupling calculation failed: {e}")


class PolarizationModel:
    """Model polarization-dependent effects in photonic components."""
    
    def __init__(self):
        self.jones_matrices = {}
        
    def linear_polarizer(self, angle: float) -> np.ndarray:
        """Jones matrix for linear polarizer at angle theta."""
        angle = np.deg2rad(angle)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        return np.array([
            [cos_theta**2, cos_theta * sin_theta],
            [cos_theta * sin_theta, sin_theta**2]
        ])
    
    def wave_plate(self, retardance: float, angle: float = 0) -> np.ndarray:
        """Jones matrix for wave plate with given retardance and orientation."""
        angle = np.deg2rad(angle)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # Rotation matrices
        R_theta = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])
        
        R_minus_theta = np.array([
            [cos_theta, sin_theta],
            [-sin_theta, cos_theta]
        ])
        
        # Wave plate in principal axes
        W = np.array([
            [1, 0],
            [0, np.exp(1j * retardance)]
        ])
        
        # Rotated wave plate
        return R_minus_theta @ W @ R_theta
    
    def quarter_wave_plate(self, angle: float = 45) -> np.ndarray:
        """Quarter-wave plate Jones matrix."""
        return self.wave_plate(np.pi/2, angle)
    
    def half_wave_plate(self, angle: float = 22.5) -> np.ndarray:
        """Half-wave plate Jones matrix."""
        return self.wave_plate(np.pi, angle)
    
    def calculate_pdl(self, jones_matrix: np.ndarray) -> float:
        """Calculate Polarization Dependent Loss (PDL) in dB."""
        try:
            # Singular value decomposition to find transmission eigenvalues
            U, s, Vh = np.linalg.svd(jones_matrix)
            
            # PDL is the ratio of maximum to minimum transmission
            if len(s) >= 2 and s[1] > 1e-12:
                pdl_linear = s[0] / s[1]
                pdl_db = 10 * np.log10(pdl_linear)
            else:
                pdl_db = np.inf
            
            return pdl_db
            
        except Exception as e:
            raise SimulationError(f"PDL calculation failed: {e}")
    
    def birefringent_waveguide(self, length: float, delta_n: float, 
                              wavelength: float = 1550e-9) -> np.ndarray:
        """Jones matrix for birefringent waveguide section."""
        try:
            # Phase difference between TE and TM modes
            delta_phi = 2 * np.pi * length * delta_n / wavelength
            
            return np.array([
                [1, 0],
                [0, np.exp(1j * delta_phi)]
            ])
            
        except Exception as e:
            raise SimulationError(f"Birefringent waveguide calculation failed: {e}")


class DispersionModel:
    """Model chromatic dispersion effects in photonic components."""
    
    def __init__(self):
        self.material_dispersion = {
            'silicon': self._silicon_dispersion,
            'silica': self._silica_dispersion,
            'silicon_nitride': self._silicon_nitride_dispersion
        }
    
    def _silicon_dispersion(self, wavelength_um: float) -> float:
        """Silicon refractive index vs wavelength (Sellmeier equation)."""
        # Sellmeier coefficients for silicon
        B1, B2, B3 = 10.6684293, 0.0030434748, 1.54133408
        C1, C2, C3 = 0.301516485**2, 1.13475115**2, 1104**2
        
        lam2 = wavelength_um**2
        
        n_squared = 1 + (B1 * lam2) / (lam2 - C1) + \
                       (B2 * lam2) / (lam2 - C2) + \
                       (B3 * lam2) / (lam2 - C3)
        
        return np.sqrt(n_squared)
    
    def _silica_dispersion(self, wavelength_um: float) -> float:
        """Silica refractive index vs wavelength."""
        # Sellmeier coefficients for fused silica
        B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
        C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2
        
        lam2 = wavelength_um**2
        
        n_squared = 1 + (B1 * lam2) / (lam2 - C1) + \
                       (B2 * lam2) / (lam2 - C2) + \
                       (B3 * lam2) / (lam2 - C3)
        
        return np.sqrt(n_squared)
    
    def _silicon_nitride_dispersion(self, wavelength_um: float) -> float:
        """Silicon nitride refractive index vs wavelength."""
        # Simplified model for Si3N4
        return 2.0 - 0.1 / wavelength_um**2
    
    def calculate_group_delay(self, material: str, length: float, 
                            wavelengths: np.ndarray) -> np.ndarray:
        """Calculate group delay for given material and length."""
        try:
            if material not in self.material_dispersion:
                raise ValidationError(f"Unknown material: {material}")
            
            wavelengths_um = wavelengths * 1e6  # Convert to micrometers
            
            # Calculate refractive index vs wavelength
            n_values = np.array([self.material_dispersion[material](wl) 
                               for wl in wavelengths_um])
            
            # Calculate group index ng = n - λ(dn/dλ)
            dn_dlam = np.gradient(n_values, wavelengths_um)
            n_group = n_values - wavelengths_um * dn_dlam
            
            # Group delay = L * ng / c
            c = 3e8  # Speed of light
            group_delay = length * n_group / c
            
            return group_delay
            
        except Exception as e:
            raise SimulationError(f"Group delay calculation failed: {e}")
    
    def calculate_dispersion_parameter(self, material: str, wavelength: float) -> float:
        """Calculate dispersion parameter D = -λ/c * d²n/dλ²."""
        try:
            if material not in self.material_dispersion:
                raise ValidationError(f"Unknown material: {material}")
            
            wavelength_um = wavelength * 1e6
            delta_wl = 0.001  # Small wavelength step in μm
            
            # Numerical differentiation for d²n/dλ²
            wl_points = np.array([wavelength_um - delta_wl, wavelength_um, wavelength_um + delta_wl])
            n_points = np.array([self.material_dispersion[material](wl) for wl in wl_points])
            
            # Second derivative
            d2n_dl2 = (n_points[2] - 2*n_points[1] + n_points[0]) / (delta_wl**2)
            
            # Dispersion parameter
            c = 3e8
            D = -(wavelength * 1e6) / c * d2n_dl2 * 1e-6  # ps/(nm·km)
            
            return D
            
        except Exception as e:
            raise SimulationError(f"Dispersion parameter calculation failed: {e}")
    
    def apply_dispersion_to_pulse(self, pulse_spectrum: np.ndarray, 
                                frequencies: np.ndarray, material: str, 
                                length: float) -> np.ndarray:
        """Apply dispersion to pulse spectrum."""
        try:
            wavelengths = 3e8 / frequencies
            
            # Calculate group delay vs frequency
            group_delays = self.calculate_group_delay(material, length, wavelengths)
            
            # Apply phase shift due to group delay
            phase_shift = -2 * np.pi * frequencies * group_delays
            dispersed_spectrum = pulse_spectrum * np.exp(1j * phase_shift)
            
            return dispersed_spectrum
            
        except Exception as e:
            raise SimulationError(f"Pulse dispersion calculation failed: {e}")


class NetworkAnalyzer:
    """Network analysis tools for photonic circuits."""
    
    def __init__(self):
        self.components = {}
        self.connections = []
        
    def add_component(self, name: str, s_parameters: np.ndarray, 
                     frequencies: np.ndarray = None):
        """Add component with S-parameters to network."""
        try:
            if s_parameters.ndim != 3:
                raise ValidationError("S-parameters must be 3D array (freq, port, port)")
                
            self.components[name] = {
                's_parameters': s_parameters,
                'frequencies': frequencies,
                'n_ports': s_parameters.shape[1]
            }
            
        except Exception as e:
            raise SimulationError(f"Failed to add component: {e}")
    
    def connect_components(self, comp1: str, port1: int, comp2: str, port2: int):
        """Connect two components."""
        if comp1 not in self.components or comp2 not in self.components:
            raise ValidationError("Components not found in network")
            
        self.connections.append((comp1, port1, comp2, port2))
    
    def calculate_network_response(self, input_component: str, input_port: int,
                                 output_component: str, output_port: int) -> np.ndarray:
        """Calculate network transfer function between input and output."""
        # This would implement full network analysis using nodal analysis
        # or scattering parameter cascading rules
        # For now, return placeholder
        
        if input_component in self.components:
            freqs = self.components[input_component]['frequencies']
            if freqs is not None:
                return np.ones(len(freqs), dtype=complex)
        
        return np.array([1.0 + 0j])
    
    def calculate_group_delay_network(self) -> np.ndarray:
        """Calculate group delay of entire network."""
        # Implementation would analyze phase response and compute group delay
        return np.array([0.0])  # Placeholder
    
    def stability_analysis(self) -> Dict[str, Any]:
        """Perform stability analysis of the network."""
        stability_results = {
            'stable': True,
            'stability_factors': [],
            'potential_oscillations': []
        }
        
        return stability_results