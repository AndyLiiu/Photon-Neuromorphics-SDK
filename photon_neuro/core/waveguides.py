"""
Specific waveguide implementations.
"""

import torch
import numpy as np
from .components import WaveguideBase
from .registry import register_component


@register_component
class SiliconWaveguide(WaveguideBase):
    """Silicon strip waveguide."""
    
    def __init__(self, length: float, width: float = 450e-9, height: float = 220e-9):
        super().__init__(length, width, "silicon")
        self.height = height
        self.loss_db_per_cm = 0.1  # Low loss for silicon
        self.n_eff = self._calculate_silicon_neff()
        
    def _calculate_silicon_neff(self) -> float:
        """Calculate effective index using approximate formula."""
        # Simplified model - in practice would use mode solver
        wavelength = 1550e-9
        n_si = 3.48
        n_sio2 = 1.44
        
        # Effective index approximation for strip waveguide
        v_number = 2 * np.pi / wavelength * self.width * np.sqrt(n_si**2 - n_sio2**2)
        
        if v_number < 1:
            # Single mode region
            neff = n_sio2 + (n_si - n_sio2) * (2/np.pi) * np.arctan(v_number)
        else:
            # Multi-mode region - use effective index method
            neff = 2.4  # Typical value
            
        return neff
        
    def calculate_dispersion(self, wavelength_range: np.ndarray) -> np.ndarray:
        """Calculate group velocity dispersion."""
        # Sellmeier equation for silicon
        wavelengths_um = wavelength_range * 1e6
        n_si = np.sqrt(1 + 10.6684293 * wavelengths_um**2 / 
                      (wavelengths_um**2 - 0.301516485**2) +
                      0.0030434748 * wavelengths_um**2 /
                      (wavelengths_um**2 - 1.13475115**2) +
                      1.54133408 * wavelengths_um**2 /
                      (wavelengths_um**2 - 1104**2))
        return n_si


@register_component  
class NitridWaveguide(WaveguideBase):
    """Silicon nitride waveguide."""
    
    def __init__(self, length: float, width: float = 1.2e-6, height: float = 400e-9):
        super().__init__(length, width, "silicon_nitride")
        self.height = height
        self.loss_db_per_cm = 0.01  # Ultra-low loss
        self.n_eff = 1.9
        
    def calculate_nonlinear_coefficient(self) -> float:
        """Calculate nonlinear coefficient γ."""
        # n2 for Si3N4
        n2 = 2.4e-19  # m2/W
        wavelength = 1550e-9
        
        # Effective area (simplified)
        A_eff = self.width * self.height * 1.5  # Mode overlap factor
        
        gamma = 2 * np.pi * n2 / (wavelength * A_eff)
        return gamma
        
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Forward propagation with nonlinear effects."""
        # Linear propagation
        output = super().forward(input_field)
        
        # Add Kerr nonlinearity for high powers
        intensity = torch.abs(input_field)**2
        if intensity.max() > 1e-3:  # Only for high power
            gamma = self.calculate_nonlinear_coefficient()
            nonlinear_phase = gamma * intensity * self.length
            output = output * torch.exp(1j * nonlinear_phase)
            
        return output