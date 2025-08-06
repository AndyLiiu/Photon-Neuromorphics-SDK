"""
Photodetector implementations.
"""

import torch
import numpy as np
from typing import Tuple, List
from .components import DetectorBase
from .registry import register_component


@register_component
class Photodetector(DetectorBase):
    """Single photodetector element."""
    
    def __init__(self, responsivity: float = 1.0, dark_current: float = 1e-9,
                 bandwidth: float = 10e9, area: float = 100e-12):
        super().__init__(responsivity, dark_current, bandwidth)
        self.area = area  # Active area in m²
        self.capacitance = self._calculate_capacitance()
        self.thermal_noise_current = self._calculate_thermal_noise()
        
    def _calculate_capacitance(self) -> float:
        """Calculate junction capacitance."""
        # Simplified model: C = ε₀εᵣA/d
        epsilon_0 = 8.854e-12  # F/m
        epsilon_r = 11.9       # Silicon relative permittivity
        depletion_width = 1e-6 # 1 μm
        return epsilon_0 * epsilon_r * self.area / depletion_width
        
    def _calculate_thermal_noise(self) -> float:
        """Calculate thermal noise current."""
        k_B = 1.381e-23  # Boltzmann constant
        T = 300          # Temperature in K
        R_load = 50      # Load resistance in Ω
        return np.sqrt(4 * k_B * T * self.bandwidth / R_load)
        
    def calculate_snr(self, optical_power: float) -> float:
        """Calculate signal-to-noise ratio."""
        signal_current = self.responsivity * optical_power
        
        # Noise sources
        shot_noise = np.sqrt(2 * 1.602e-19 * signal_current * self.bandwidth)
        thermal_noise = self.thermal_noise_current
        dark_noise = np.sqrt(2 * 1.602e-19 * self.dark_current * self.bandwidth)
        
        total_noise = np.sqrt(shot_noise**2 + thermal_noise**2 + dark_noise**2)
        return 20 * np.log10(signal_current / total_noise)
        
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Convert optical field to photocurrent."""
        optical_power = torch.abs(input_field)**2
        photocurrent = self.responsivity * optical_power
        
        # Add noise in training mode
        if self.training and optical_power.numel() > 0:
            signal_power = optical_power.mean().item()
            
            # Shot noise
            shot_noise_std = np.sqrt(2 * 1.602e-19 * self.responsivity * 
                                   signal_power * self.bandwidth)
            
            # Total noise
            total_noise_std = np.sqrt(shot_noise_std**2 + 
                                    self.thermal_noise_current**2)
            
            noise = torch.randn_like(photocurrent) * total_noise_std
            photocurrent = photocurrent + noise
            
        return photocurrent + self.dark_current


@register_component
class PhotodetectorArray(DetectorBase):
    """Array of photodetectors for parallel detection."""
    
    def __init__(self, n_detectors: int, spacing: float = 50e-6,
                 responsivity: float = 1.0, dark_current: float = 1e-9,
                 bandwidth: float = 10e9):
        super().__init__(responsivity, dark_current, bandwidth)
        self.n_detectors = n_detectors
        self.spacing = spacing
        self.detectors = [
            Photodetector(responsivity, dark_current, bandwidth)
            for _ in range(n_detectors)
        ]
        
    def forward(self, input_fields: torch.Tensor) -> torch.Tensor:
        """Process array of optical inputs."""
        if input_fields.shape[0] != self.n_detectors:
            raise ValueError(f"Expected {self.n_detectors} inputs, got {input_fields.shape[0]}")
            
        outputs = []
        for i, detector in enumerate(self.detectors):
            output = detector.forward(input_fields[i])
            outputs.append(output)
            
        return torch.stack(outputs)
        
    def measure_crosstalk(self) -> torch.Tensor:
        """Measure electrical crosstalk between detectors."""
        crosstalk_matrix = torch.eye(self.n_detectors)
        
        # Add nearest-neighbor crosstalk (simplified)
        crosstalk_level = -30  # dB
        crosstalk_linear = 10**(crosstalk_level/20)
        
        for i in range(self.n_detectors - 1):
            crosstalk_matrix[i, i+1] = crosstalk_linear
            crosstalk_matrix[i+1, i] = crosstalk_linear
            
        return crosstalk_matrix
        
    def calibrate_responsivity(self, reference_powers: List[float], 
                             measured_currents: List[float]) -> None:
        """Calibrate detector responsivity from measurements."""
        if len(reference_powers) != len(measured_currents):
            raise ValueError("Reference powers and currents must have same length")
            
        # Linear fit to get responsivity
        powers = np.array(reference_powers)
        currents = np.array(measured_currents)
        
        responsivity = np.polyfit(powers, currents, 1)[0]
        
        # Update all detectors
        for detector in self.detectors:
            detector.responsivity = responsivity
            
    def to_netlist(self) -> dict:
        return {
            "type": "photodetector_array",
            "n_detectors": self.n_detectors,
            "spacing": self.spacing,
            "responsivity": self.responsivity,
            "dark_current": self.dark_current,
            "bandwidth": self.bandwidth
        }