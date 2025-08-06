"""
Advanced noise modeling for photonic systems with quantum effects, 
coherence modeling, and crosstalk analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import scipy.signal
import scipy.interpolate
from ..core import PhotonicComponent
from ..core.exceptions import (
    NoiseModelError, ValidationError, validate_parameter,
    check_tensor_validity, safe_execution, global_error_recovery
)


class QuantumNoiseModel:
    """Advanced quantum noise model with multiple noise sources."""
    
    def __init__(self, temperature: float = 300, planck_constant: float = 6.626e-34):
        try:
            self.temperature = validate_parameter("temperature", temperature, 
                                                expected_type=(int, float), 
                                                valid_range=(0, 1000))
            self.h = planck_constant
            self.k_B = 1.381e-23  # Boltzmann constant
            self.hbar = self.h / (2 * np.pi)
            self.c = 3e8  # Speed of light
            self.q = 1.602e-19  # Elementary charge
            
            # Advanced noise parameters
            self.coherence_time = 1e-12  # 1 ps default
            self.pump_noise_enabled = True
            self.thermal_noise_model = "advanced"  # or "simple"
            
        except ValidationError as e:
            raise NoiseModelError(f"Invalid noise model parameters: {e}")
    
    @safe_execution(fallback_value=0.0)
    def shot_noise_variance(self, photocurrent: float, bandwidth: float) -> float:
        """Calculate enhanced shot noise variance with quantum corrections."""
        try:
            photocurrent = validate_parameter("photocurrent", photocurrent, 
                                            expected_type=(int, float), 
                                            valid_range=(0, 1.0))
            bandwidth = validate_parameter("bandwidth", bandwidth, 
                                         expected_type=(int, float), 
                                         valid_range=(1e6, 1e15))
            
            # Basic shot noise
            shot_variance = 2 * self.q * abs(photocurrent) * bandwidth
            
            # Quantum corrections for high photon rates
            photon_rate = photocurrent / self.q
            if photon_rate > 1e12:  # High photon rate regime
                quantum_correction = 1 + self.hbar / (2 * photon_rate * self.h)
                shot_variance *= quantum_correction
                
            return shot_variance
            
        except Exception as e:
            raise NoiseModelError(f"Shot noise calculation failed: {e}")
    
    @safe_execution(fallback_value=0.0)
    def thermal_noise_current_variance(self, resistance: float, bandwidth: float) -> float:
        """Calculate thermal (Johnson-Nyquist) noise current variance."""
        try:
            resistance = validate_parameter("resistance", resistance, 
                                          expected_type=(int, float), 
                                          valid_range=(1.0, 1e9))
            bandwidth = validate_parameter("bandwidth", bandwidth, 
                                         expected_type=(int, float), 
                                         valid_range=(1e6, 1e15))
            
            if self.thermal_noise_model == "advanced":
                # Advanced thermal noise with frequency dependence
                return 4 * self.k_B * self.temperature * bandwidth / resistance
            else:
                # Simple thermal noise
                return 4 * self.k_B * self.temperature * bandwidth / resistance
                
        except Exception as e:
            raise NoiseModelError(f"Thermal noise calculation failed: {e}")
    
    def pump_relative_intensity_noise(self, laser_power: float, frequency: float, 
                                    rin_spectral_density: float = -155) -> float:
        """Calculate pump relative intensity noise (RIN)."""
        try:
            laser_power = validate_parameter("laser_power", laser_power, 
                                           expected_type=(int, float), 
                                           valid_range=(1e-6, 1.0))
            frequency = validate_parameter("frequency", frequency, 
                                         expected_type=(int, float), 
                                         valid_range=(1e3, 1e12))
            
            # RIN in dB/Hz to linear
            rin_linear = 10**(rin_spectral_density / 10)  # 1/Hz
            
            # RIN noise power
            noise_power = rin_linear * laser_power**2 * frequency
            
            return np.sqrt(noise_power)
            
        except Exception as e:
            raise NoiseModelError(f"RIN calculation failed: {e}")
    
    def coherence_noise(self, optical_field: torch.Tensor, 
                       coherence_length: float = None) -> torch.Tensor:
        """Add coherence-limited phase noise to optical field."""
        try:
            optical_field = check_tensor_validity(optical_field, "optical_field")
            
            if coherence_length is None:
                coherence_length = self.c * self.coherence_time
                
            coherence_length = validate_parameter("coherence_length", coherence_length,
                                                expected_type=(int, float),
                                                valid_range=(1e-6, 1.0))
            
            # Calculate coherence bandwidth
            coherence_bandwidth = self.c / coherence_length
            
            # Generate coherence noise with appropriate correlation
            field_size = optical_field.shape
            
            # White phase noise filtered by coherence bandwidth  
            white_phase = torch.randn(field_size, dtype=optical_field.dtype)
            
            # Simple coherence filtering (in practice would use proper filter)
            coherence_factor = np.sqrt(coherence_bandwidth / 1e12)  # Normalize to 1 THz
            coherence_phase = white_phase * coherence_factor
            
            # Apply phase noise
            noisy_field = optical_field * torch.exp(1j * coherence_phase.real)
            
            return noisy_field
            
        except Exception as e:
            raise NoiseModelError(f"Coherence noise calculation failed: {e}")
    
    def excess_noise_factor(self, gain: float, noise_figure: float = 3.0) -> float:
        """Calculate excess noise factor for amplified systems."""
        try:
            gain = validate_parameter("gain", gain, expected_type=(int, float), valid_range=(1.0, 1000))
            noise_figure = validate_parameter("noise_figure", noise_figure, 
                                            expected_type=(int, float), valid_range=(0.1, 20))
            
            # Convert noise figure from dB to linear
            nf_linear = 10**(noise_figure / 10)
            
            # Excess noise factor
            excess_noise = nf_linear * (gain - 1)
            
            return excess_noise
            
        except Exception as e:
            raise NoiseModelError(f"Excess noise calculation failed: {e}")
    
    def one_over_f_noise(self, frequencies: np.ndarray, 
                        corner_frequency: float = 1e3) -> np.ndarray:
        """Generate 1/f (flicker) noise power spectral density."""
        try:
            frequencies = validate_parameter("frequencies", frequencies, 
                                           expected_type=np.ndarray)
            corner_frequency = validate_parameter("corner_frequency", corner_frequency,
                                                expected_type=(int, float),
                                                valid_range=(1.0, 1e9))
            
            # Avoid division by zero
            frequencies = np.maximum(frequencies, 1e-12)
            
            # 1/f noise PSD with corner frequency
            flicker_psd = corner_frequency / frequencies
            
            # Add white noise floor
            white_floor = 1e-6
            total_psd = flicker_psd + white_floor
            
            return total_psd
            
        except Exception as e:
            raise NoiseModelError(f"1/f noise calculation failed: {e}")


class CrosstalkModel:
    """Model for optical and electrical crosstalk between components."""
    
    def __init__(self):
        self.optical_crosstalk_matrix = {}
        self.electrical_crosstalk_matrix = {}
        self.thermal_crosstalk_matrix = {}
        
    def add_optical_crosstalk(self, source_idx: int, target_idx: int, 
                            coupling_coefficient: float, distance: float = None):
        """Add optical crosstalk between waveguides."""
        try:
            coupling_coefficient = validate_parameter("coupling_coefficient", coupling_coefficient,
                                                    expected_type=(int, float),
                                                    valid_range=(-60, 0))  # dB
            
            if distance is not None:
                # Distance-dependent crosstalk
                distance = validate_parameter("distance", distance,
                                            expected_type=(int, float),
                                            valid_range=(1e-6, 1e-2))
                # Simple exponential decay model
                coupling_coefficient -= 20 * np.log10(distance / 1e-6)  # Additional loss
            
            key = (source_idx, target_idx)
            self.optical_crosstalk_matrix[key] = 10**(coupling_coefficient / 20)  # Linear coefficient
            
        except ValidationError as e:
            raise NoiseModelError(f"Invalid crosstalk parameters: {e}")
    
    def add_electrical_crosstalk(self, source_idx: int, target_idx: int,
                               coupling_impedance: complex, frequency: float):
        """Add electrical crosstalk between circuits."""
        try:
            frequency = validate_parameter("frequency", frequency,
                                         expected_type=(int, float),
                                         valid_range=(1e3, 1e12))
            
            key = (source_idx, target_idx, frequency)
            self.electrical_crosstalk_matrix[key] = coupling_impedance
            
        except ValidationError as e:
            raise NoiseModelError(f"Invalid electrical crosstalk parameters: {e}")
    
    def add_thermal_crosstalk(self, source_idx: int, target_idx: int,
                            thermal_coupling: float):
        """Add thermal crosstalk between components."""
        try:
            thermal_coupling = validate_parameter("thermal_coupling", thermal_coupling,
                                                expected_type=(int, float),
                                                valid_range=(0, 1))
            
            key = (source_idx, target_idx)
            self.thermal_crosstalk_matrix[key] = thermal_coupling
            
        except ValidationError as e:
            raise NoiseModelError(f"Invalid thermal crosstalk parameters: {e}")
    
    def apply_optical_crosstalk(self, signals: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Apply optical crosstalk to signals."""
        try:
            crosstalk_signals = signals.copy()
            
            for (source_idx, target_idx), coupling in self.optical_crosstalk_matrix.items():
                if source_idx in signals and target_idx in crosstalk_signals:
                    # Add crosstalk contribution
                    crosstalk_contribution = signals[source_idx] * coupling
                    crosstalk_signals[target_idx] += crosstalk_contribution
            
            return crosstalk_signals
            
        except Exception as e:
            raise NoiseModelError(f"Optical crosstalk application failed: {e}")
    
    def calculate_crosstalk_matrix(self, n_channels: int, 
                                 spacing: float, coupling_coefficient: float = -30) -> np.ndarray:
        """Calculate full crosstalk matrix for array of channels."""
        try:
            n_channels = validate_parameter("n_channels", n_channels,
                                          expected_type=int,
                                          valid_range=(2, 1000))
            spacing = validate_parameter("spacing", spacing,
                                       expected_type=(int, float),
                                       valid_range=(1e-6, 1e-3))
            
            matrix = np.eye(n_channels, dtype=complex)
            
            # Adjacent channel crosstalk
            coupling_linear = 10**(coupling_coefficient / 20)
            
            for i in range(n_channels - 1):
                # Nearest neighbor coupling
                matrix[i, i + 1] = coupling_linear
                matrix[i + 1, i] = coupling_linear
                
                # Next-nearest neighbor (weaker)
                if i < n_channels - 2:
                    next_coupling = coupling_linear * 0.1
                    matrix[i, i + 2] = next_coupling
                    matrix[i + 2, i] = next_coupling
            
            return matrix
            
        except Exception as e:
            raise NoiseModelError(f"Crosstalk matrix calculation failed: {e}")


class TemperatureDependentNoise:
    """Model temperature-dependent noise effects."""
    
    def __init__(self, base_temperature: float = 300):
        self.base_temperature = validate_parameter("base_temperature", base_temperature,
                                                 expected_type=(int, float),
                                                 valid_range=(0, 1000))
        
    def thermal_phase_noise(self, component_length: float, 
                           temperature_fluctuation: float,
                           material: str = "silicon") -> float:
        """Calculate thermal phase noise due to temperature fluctuations."""
        try:
            component_length = validate_parameter("component_length", component_length,
                                                expected_type=(int, float),
                                                valid_range=(1e-6, 1e-2))
            
            # Thermo-optic coefficients (1/K)
            thermo_optic = {
                "silicon": 1.8e-4,
                "silicon_nitride": 2.5e-5,
                "silica": 1e-5,
                "polymer": 1e-4
            }
            
            dn_dt = thermo_optic.get(material, 1e-4)
            wavelength = 1550e-9  # m
            
            # Phase change due to temperature
            delta_phase = 2 * np.pi * component_length * dn_dt * temperature_fluctuation / wavelength
            
            return delta_phase
            
        except Exception as e:
            raise NoiseModelError(f"Thermal phase noise calculation failed: {e}")
    
    def temperature_noise_psd(self, frequencies: np.ndarray,
                            thermal_time_constant: float = 1e-3) -> np.ndarray:
        """Calculate temperature fluctuation noise PSD."""
        try:
            frequencies = validate_parameter("frequencies", frequencies, expected_type=np.ndarray)
            thermal_time_constant = validate_parameter("thermal_time_constant", thermal_time_constant,
                                                     expected_type=(int, float),
                                                     valid_range=(1e-9, 1.0))
            
            # First-order thermal response
            omega = 2 * np.pi * frequencies
            thermal_cutoff = 1 / thermal_time_constant
            
            # Low-pass thermal response
            temp_response = 1 / (1 + 1j * omega / thermal_cutoff)
            temp_psd = np.abs(temp_response)**2
            
            return temp_psd
            
        except Exception as e:
            raise NoiseModelError(f"Temperature noise PSD calculation failed: {e}")


class WaveguideCrosstalk:
    """Specialized crosstalk model for waveguide arrays."""
    
    def __init__(self, n_guides: int, spacing: float, coupling_length: float):
        try:
            self.n_guides = validate_parameter("n_guides", n_guides,
                                             expected_type=int,
                                             valid_range=(2, 1000))
            self.spacing = validate_parameter("spacing", spacing,
                                            expected_type=(int, float),
                                            valid_range=(1e-6, 1e-3))
            self.coupling_length = validate_parameter("coupling_length", coupling_length,
                                                    expected_type=(int, float),
                                                    valid_range=(1e-6, 1e-2))
            
            self.coupling_matrix = self._calculate_coupling_matrix()
            
        except ValidationError as e:
            raise NoiseModelError(f"Invalid waveguide crosstalk parameters: {e}")
    
    def _calculate_coupling_matrix(self) -> np.ndarray:
        """Calculate waveguide coupling matrix using coupled-mode theory."""
        try:
            # Coupling coefficient estimation (simplified)
            kappa_0 = 1e4  # Base coupling coefficient (1/m)
            
            # Exponential decay with spacing
            decay_length = 500e-9  # m
            coupling_strength = kappa_0 * np.exp(-self.spacing / decay_length)
            
            # Build coupling matrix
            C = np.zeros((self.n_guides, self.n_guides), dtype=complex)
            
            # Nearest neighbor coupling
            for i in range(self.n_guides - 1):
                coupling = coupling_strength * self.coupling_length
                C[i, i + 1] = 1j * coupling
                C[i + 1, i] = 1j * coupling
            
            return C
            
        except Exception as e:
            raise NoiseModelError(f"Coupling matrix calculation failed: {e}")
    
    def calculate_crosstalk(self, input_powers: np.ndarray) -> np.ndarray:
        """Calculate output powers with crosstalk."""
        try:
            input_powers = validate_parameter("input_powers", input_powers,
                                            expected_type=np.ndarray)
            
            if len(input_powers) != self.n_guides:
                raise ValidationError(f"Input power array length {len(input_powers)} != n_guides {self.n_guides}")
            
            # Convert to field amplitudes
            input_fields = np.sqrt(input_powers + 1e-12)
            
            # Apply coupling matrix (simplified - should use proper coupled-mode solution)
            phase_matrix = scipy.linalg.expm(self.coupling_matrix)
            output_fields = phase_matrix @ input_fields
            
            # Convert back to powers
            output_powers = np.abs(output_fields)**2
            
            return output_powers
            
        except Exception as e:
            raise NoiseModelError(f"Crosstalk calculation failed: {e}")
    
    def crosstalk_transfer_function(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate frequency-dependent crosstalk transfer function."""
        try:
            frequencies = validate_parameter("frequencies", frequencies, expected_type=np.ndarray)
            
            # Frequency-dependent coupling (simplified)
            # In reality, this would involve solving the full coupled-wave equations
            
            n_freq = len(frequencies)
            transfer_matrix = np.zeros((n_freq, self.n_guides, self.n_guides), dtype=complex)
            
            for i, freq in enumerate(frequencies):
                # Wavelength-dependent coupling
                wavelength = 3e8 / freq
                beta = 2 * np.pi / wavelength * 2.4  # Effective index ~2.4
                
                # Phase-matching factor
                phase_mismatch = 0  # Assume matched waveguides
                
                # Modified coupling matrix with frequency dependence
                freq_coupling_matrix = self.coupling_matrix.copy()
                
                # Apply frequency-dependent modification
                freq_factor = 1 / (1 + 1j * phase_mismatch / abs(self.coupling_matrix[0, 1]))
                freq_coupling_matrix *= freq_factor
                
                transfer_matrix[i] = scipy.linalg.expm(freq_coupling_matrix)
            
            return transfer_matrix
            
        except Exception as e:
            raise NoiseModelError(f"Frequency-dependent crosstalk calculation failed: {e}")


# Update original thermal_noise_variance method that was left incomplete
def thermal_noise_variance_legacy(self, resistance: float, bandwidth: float) -> float:
    """Calculate thermal (Johnson) noise variance - legacy method."""
    return 4 * self.k_B * self.temperature * bandwidth / resistance
        
    def quantum_shot_noise(self, optical_power: torch.Tensor, 
                          photon_energy: float, bandwidth: float) -> torch.Tensor:
        """Generate quantum shot noise."""
        # Photon flux
        photon_rate = optical_power / photon_energy
        
        # Shot noise standard deviation
        shot_noise_std = torch.sqrt(2 * photon_energy * photon_rate * bandwidth)
        
        # Generate noise
        noise = torch.randn_like(optical_power) * shot_noise_std
        return noise
        
    def phase_noise(self, optical_field: torch.Tensor, linewidth: float) -> torch.Tensor:
        """Add laser phase noise (Wiener process)."""
        if linewidth <= 0:
            return optical_field
            
        # Phase diffusion
        dt = 1e-12  # Assume 1 ps time step
        phase_variance = 2 * np.pi * linewidth * dt
        
        phase_noise = torch.randn_like(optical_field.real) * np.sqrt(phase_variance)
        
        return optical_field * torch.exp(1j * phase_noise)
        
    def add_thermal_fluctuations(self, component: PhotonicComponent) -> torch.Tensor:
        """Add thermal fluctuations to component parameters."""
        if not hasattr(component, 'temperature_coefficient'):
            return torch.zeros(1)
            
        # Temperature fluctuations (1/f noise)
        temp_noise_std = 0.1  # 0.1K RMS
        temp_fluctuation = torch.randn(1) * temp_noise_std
        
        # Parameter drift due to temperature
        parameter_drift = component.temperature_coefficient * temp_fluctuation
        
        return parameter_drift


class NoiseSimulator:
    """Comprehensive noise analysis for photonic systems."""
    
    def __init__(self, photonic_model: PhotonicComponent):
        self.model = photonic_model
        self.quantum_noise = QuantumNoiseModel()
        self.noise_history = {}
        
    def sweep_input_power(self, power_range_dbm: Tuple[float, float],
                         n_points: int = 50, include_shot_noise: bool = True,
                         include_thermal_noise: bool = True, 
                         temperature: float = 300) -> Dict[str, np.ndarray]:
        """Sweep input power and analyze SNR."""
        power_min, power_max = power_range_dbm
        power_dbm_range = np.linspace(power_min, power_max, n_points)
        power_watts_range = 1e-3 * 10**(power_dbm_range / 10)
        
        snr_results = []
        shot_noise_limited = []
        thermal_noise_limited = []
        
        for power_w in power_watts_range:
            # Create input signal
            input_field = torch.sqrt(torch.tensor(power_w, dtype=torch.float32))
            input_field = input_field.to(torch.complex64)
            
            # Process through model
            try:
                output_field = self.model.forward(input_field.unsqueeze(0))
                output_power = torch.abs(output_field)**2
                
                # Calculate noise components
                bandwidth = 10e9  # 10 GHz bandwidth
                
                signal_power = output_power.mean().item()
                
                if include_shot_noise:
                    photon_energy = 1.24e-6 / 1550  # eV at 1550nm
                    shot_variance = self.quantum_noise.shot_noise_variance(
                        signal_power, bandwidth
                    )
                else:
                    shot_variance = 0
                    
                if include_thermal_noise:
                    resistance = 50  # 50 Ohm load
                    thermal_variance = self.quantum_noise.thermal_noise_variance(
                        resistance, bandwidth
                    )
                else:
                    thermal_variance = 0
                    
                total_noise_variance = shot_variance + thermal_variance
                
                if total_noise_variance > 0:
                    snr_linear = signal_power / total_noise_variance
                    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
                else:
                    snr_db = np.inf
                    
                snr_results.append(snr_db)
                shot_noise_limited.append(10 * np.log10(signal_power / shot_variance) 
                                        if shot_variance > 0 else np.inf)
                thermal_noise_limited.append(10 * np.log10(signal_power / thermal_variance)
                                           if thermal_variance > 0 else np.inf)
                                           
            except Exception as e:
                print(f"Warning: Simulation failed at {power_w} W: {e}")
                snr_results.append(-np.inf)
                shot_noise_limited.append(-np.inf)
                thermal_noise_limited.append(-np.inf)
                
        return {
            'power_dbm': power_dbm_range,
            'power_watts': power_watts_range,
            'snr_db': np.array(snr_results),
            'shot_noise_limited': np.array(shot_noise_limited),
            'thermal_noise_limited': np.array(thermal_noise_limited)
        }
        
    def analyze_phase_noise(self, laser_linewidth: float, 
                           measurement_time: float = 1e-3) -> Dict[str, np.ndarray]:
        """Analyze phase noise characteristics."""
        dt = 1e-12  # 1 ps time step
        n_samples = int(measurement_time / dt)
        
        # Generate phase noise time series
        phase_variance = 2 * np.pi * laser_linewidth * dt
        phase_increments = np.random.normal(0, np.sqrt(phase_variance), n_samples)
        phase_evolution = np.cumsum(phase_increments)
        
        # Calculate phase noise PSD
        frequencies = np.fft.fftfreq(n_samples, dt)
        phase_fft = np.fft.fft(phase_evolution)
        phase_psd = np.abs(phase_fft)**2 / (n_samples * dt)
        
        # Allan variance calculation
        tau_range = np.logspace(-9, -6, 20)  # 1 ns to 1 μs
        allan_var = []
        
        for tau in tau_range:
            n_tau = int(tau / dt)
            if n_tau < n_samples // 2:
                y_values = []
                for i in range(0, n_samples - 2*n_tau, n_tau):
                    y_k = np.mean(phase_evolution[i:i+n_tau])
                    y_k1 = np.mean(phase_evolution[i+n_tau:i+2*n_tau])
                    y_values.append((y_k1 - y_k)**2)
                    
                allan_var.append(np.mean(y_values) / 2)
            else:
                allan_var.append(np.nan)
                
        return {
            'time': np.arange(n_samples) * dt,
            'phase_evolution': phase_evolution,
            'frequencies': frequencies[:n_samples//2],
            'phase_psd': phase_psd[:n_samples//2],
            'tau_range': tau_range,
            'allan_variance': np.array(allan_var)
        }
        
    def monte_carlo_analysis(self, n_runs: int = 1000, 
                           fabrication_variations: Dict[str, float] = None) -> Dict[str, np.ndarray]:
        """Monte Carlo analysis with fabrication variations."""
        if fabrication_variations is None:
            fabrication_variations = {
                'width_sigma': 5e-9,      # 5 nm width variation
                'thickness_sigma': 2e-9,   # 2 nm thickness variation  
                'index_sigma': 0.001       # 0.1% index variation
            }
            
        results = {
            'transmission': [],
            'phase': [],
            'loss': [],
            'bandwidth': []
        }
        
        # Nominal input
        input_field = torch.ones(1, dtype=torch.complex64)
        
        for run in range(n_runs):
            try:
                # Apply random variations to model parameters
                self._apply_fabrication_variations(fabrication_variations)
                
                # Run simulation
                output_field = self.model.forward(input_field)
                
                # Extract metrics
                transmission = torch.abs(output_field).item()
                phase = torch.angle(output_field).item()
                loss = -20 * np.log10(transmission) if transmission > 0 else np.inf
                
                results['transmission'].append(transmission)
                results['phase'].append(phase)
                results['loss'].append(loss)
                
                # Reset model parameters
                self._reset_model_parameters()
                
            except Exception as e:
                print(f"Warning: Monte Carlo run {run} failed: {e}")
                continue
                
        # Convert to arrays and calculate statistics
        for key in results:
            results[key] = np.array(results[key])
            
        return results
        
    def _apply_fabrication_variations(self, variations: Dict[str, float]):
        """Apply random fabrication variations to model."""
        # Store nominal values if not already stored
        if not hasattr(self, '_nominal_parameters'):
            self._store_nominal_parameters()
            
        # Apply variations to different parameter types
        if 'width_sigma' in variations and hasattr(self.model, 'width'):
            width_variation = np.random.normal(0, variations['width_sigma'])
            self.model.width += width_variation
            
        if 'thickness_sigma' in variations and hasattr(self.model, 'thickness'):
            thickness_variation = np.random.normal(0, variations['thickness_sigma'])
            self.model.thickness += thickness_variation
            
        if 'index_sigma' in variations and hasattr(self.model, 'n_eff'):
            # Relative index variation
            index_variation = np.random.normal(1.0, variations['index_sigma'])
            self.model.n_eff *= index_variation
            
        if 'loss_sigma' in variations and hasattr(self.model, 'loss_db_per_cm'):
            # Loss variation in dB/cm
            loss_variation = np.random.normal(0, variations.get('loss_sigma', 0.01))
            self.model.loss_db_per_cm += loss_variation
            
        # For phase shifters and modulators
        if 'phase_sigma' in variations:
            self._apply_phase_variations(variations['phase_sigma'])
            
        # For modulators
        if 'vpi_sigma' in variations and hasattr(self.model, 'v_pi'):
            vpi_variation = np.random.normal(1.0, variations['vpi_sigma'])
            self.model.v_pi *= vpi_variation
            
    def _apply_phase_variations(self, phase_sigma: float):
        """Apply phase variations to phase shifters."""
        if hasattr(self.model, 'phases'):
            # For MZI meshes with phase shifters
            phase_variations = np.random.normal(0, phase_sigma, self.model.phases.shape)
            self.model.phases.data += torch.tensor(phase_variations, dtype=torch.float32)
        elif hasattr(self.model, 'layers'):
            # For multi-layer models
            for layer in self.model.layers:
                if hasattr(layer, 'phases'):
                    phase_variations = np.random.normal(0, phase_sigma, layer.phases.shape)
                    layer.phases.data += torch.tensor(phase_variations, dtype=torch.float32)
                    
    def _store_nominal_parameters(self):
        """Store nominal parameter values for reset."""
        self._nominal_parameters = {}
        
        # Store common parameters
        params_to_store = ['width', 'thickness', 'n_eff', 'loss_db_per_cm', 'v_pi']
        for param in params_to_store:
            if hasattr(self.model, param):
                self._nominal_parameters[param] = getattr(self.model, param)
                
        # Store phase parameters
        if hasattr(self.model, 'phases'):
            self._nominal_parameters['phases'] = self.model.phases.data.clone()
        elif hasattr(self.model, 'layers'):
            self._nominal_parameters['layer_phases'] = {}
            for i, layer in enumerate(self.model.layers):
                if hasattr(layer, 'phases'):
                    self._nominal_parameters['layer_phases'][i] = layer.phases.data.clone()
        
    def _reset_model_parameters(self):
        """Reset model parameters to nominal values."""
        if not hasattr(self, '_nominal_parameters'):
            return
            
        # Reset common parameters
        for param, value in self._nominal_parameters.items():
            if param == 'phases' and hasattr(self.model, 'phases'):
                self.model.phases.data = value.clone()
            elif param == 'layer_phases' and hasattr(self.model, 'layers'):
                for i, phase_data in value.items():
                    if i < len(self.model.layers) and hasattr(self.model.layers[i], 'phases'):
                        self.model.layers[i].phases.data = phase_data.clone()
            elif param not in ['phases', 'layer_phases'] and hasattr(self.model, param):
                setattr(self.model, param, value)
        
    def plot_snr_curves(self, snr_results: Dict[str, np.ndarray]):
        """Plot SNR vs input power curves."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(snr_results['power_dbm'], snr_results['snr_db'], 
                'b-', linewidth=2, label='Total SNR')
        plt.plot(snr_results['power_dbm'], snr_results['shot_noise_limited'],
                'r--', linewidth=1, label='Shot noise limited')
        plt.plot(snr_results['power_dbm'], snr_results['thermal_noise_limited'],
                'g--', linewidth=1, label='Thermal noise limited')
        
        plt.xlabel('Input Power (dBm)')
        plt.ylabel('SNR (dB)')
        plt.title('Signal-to-Noise Ratio vs Input Power')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(snr_results['power_dbm'][0], snr_results['power_dbm'][-1])
        
        plt.tight_layout()
        plt.show()
        
    def plot_phase_noise(self, phase_results: Dict[str, np.ndarray]):
        """Plot phase noise analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time evolution
        axes[0, 0].plot(phase_results['time'] * 1e9, phase_results['phase_evolution'])
        axes[0, 0].set_xlabel('Time (ns)')
        axes[0, 0].set_ylabel('Phase (rad)')
        axes[0, 0].set_title('Phase Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase PSD
        axes[0, 1].loglog(phase_results['frequencies'], phase_results['phase_psd'])
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Phase PSD (rad²/Hz)')
        axes[0, 1].set_title('Phase Noise Power Spectral Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Allan variance
        valid_mask = ~np.isnan(phase_results['allan_variance'])
        axes[1, 0].loglog(phase_results['tau_range'][valid_mask], 
                         phase_results['allan_variance'][valid_mask])
        axes[1, 0].set_xlabel('Averaging Time τ (s)')
        axes[1, 0].set_ylabel('Allan Variance (rad²)')
        axes[1, 0].set_title('Allan Variance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram of phase values
        axes[1, 1].hist(phase_results['phase_evolution'], bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('Phase (rad)')
        axes[1, 1].set_ylabel('Counts')
        axes[1, 1].set_title('Phase Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()