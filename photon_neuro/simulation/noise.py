"""
Noise modeling for photonic systems.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from ..core import PhotonicComponent


class QuantumNoiseModel:
    """Quantum noise model for photonic devices."""
    
    def __init__(self, temperature: float = 300, planck_constant: float = 6.626e-34):
        self.temperature = temperature  # Kelvin
        self.h = planck_constant
        self.k_B = 1.381e-23  # Boltzmann constant
        self.hbar = self.h / (2 * np.pi)
        
    def shot_noise_variance(self, photocurrent: float, bandwidth: float) -> float:
        """Calculate shot noise variance."""
        q = 1.602e-19  # Elementary charge
        return 2 * q * abs(photocurrent) * bandwidth
        
    def thermal_noise_variance(self, resistance: float, bandwidth: float) -> float:
        """Calculate thermal (Johnson) noise variance."""
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
        # This would modify model parameters based on variations
        # Implementation depends on specific model structure
        pass
        
    def _reset_model_parameters(self):
        """Reset model parameters to nominal values."""
        # Reset any modified parameters
        pass
        
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