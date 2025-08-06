"""
Optical source implementations.
"""

import torch
import numpy as np
from typing import Optional
from .components import PhotonicComponent
from .registry import register_component


@register_component
class LaserSource(PhotonicComponent):
    """Continuous wave laser source."""
    
    def __init__(self, wavelength: float = 1550e-9, power_dbm: float = 0,
                 linewidth: float = 1e6, rin: float = -160):
        super().__init__()
        self.wavelength = wavelength
        self.power_dbm = power_dbm
        self.linewidth = linewidth  # Hz
        self.rin = rin  # dB/Hz (relative intensity noise)
        self.frequency = 3e8 / wavelength
        
    @property
    def power_watts(self) -> float:
        """Get optical power in watts."""
        return 1e-3 * 10**(self.power_dbm / 10)
        
    def forward(self, modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate laser output field."""
        # Base field amplitude
        field_amplitude = np.sqrt(self.power_watts)
        
        if modulation is not None:
            # Apply modulation
            output_field = field_amplitude * modulation
        else:
            # CW operation
            output_field = torch.ones(1, dtype=torch.complex64) * field_amplitude
            
        # Add phase noise from finite linewidth
        if self.training and self.linewidth > 0:
            # Wiener process for phase noise
            phase_noise_std = np.sqrt(2 * np.pi * self.linewidth)
            phase_noise = torch.randn_like(output_field.real) * phase_noise_std
            output_field = output_field * torch.exp(1j * phase_noise)
            
        # Add intensity noise
        if self.training and self.rin < 0:
            rin_linear = 10**(self.rin / 10)
            intensity_noise_std = np.sqrt(rin_linear * self.power_watts)
            intensity_noise = torch.randn_like(output_field.real) * intensity_noise_std
            amplitude_noise = intensity_noise / (2 * field_amplitude + 1e-12)
            output_field = output_field * (1 + amplitude_noise)
            
        return output_field
        
    def calculate_power_consumption(self) -> float:
        """Calculate electrical power consumption."""
        # Typical wall-plug efficiency for DFB laser
        efficiency = 0.15  # 15%
        return self.power_watts / efficiency
        
    def tune_wavelength(self, wavelength: float):
        """Tune laser wavelength."""
        self.wavelength = wavelength
        self.frequency = 3e8 / wavelength
        
    def set_power(self, power_dbm: float):
        """Set optical output power."""
        self.power_dbm = power_dbm
        
    def get_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Get optical spectrum (Lorentzian lineshape)."""
        delta_f = frequencies - self.frequency
        spectrum = self.power_watts * (self.linewidth / 2) / \
                  (delta_f**2 + (self.linewidth / 2)**2)
        return spectrum
        
    def to_netlist(self) -> dict:
        return {
            "type": "laser_source",
            "wavelength": self.wavelength,
            "power_dbm": self.power_dbm,
            "linewidth": self.linewidth,
            "rin": self.rin
        }


@register_component
class LEDSource(PhotonicComponent):
    """Light-emitting diode source."""
    
    def __init__(self, center_wavelength: float = 1550e-9, 
                 spectral_width: float = 50e-9, power_dbm: float = -10):
        super().__init__()
        self.center_wavelength = center_wavelength
        self.spectral_width = spectral_width
        self.power_dbm = power_dbm
        self.center_frequency = 3e8 / center_wavelength
        
    @property
    def power_watts(self) -> float:
        """Get optical power in watts."""
        return 1e-3 * 10**(self.power_dbm / 10)
        
    def forward(self, drive_current: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate LED output field."""
        if drive_current is not None:
            # Current-dependent output
            # Simplified L-I curve
            threshold_current = 10e-3  # 10 mA
            slope_efficiency = 0.1     # W/A above threshold
            
            optical_power = torch.clamp(
                slope_efficiency * (drive_current - threshold_current), 
                min=0
            )
        else:
            optical_power = torch.ones(1) * self.power_watts
            
        # LED has random phase (incoherent)
        random_phase = 2 * np.pi * torch.rand_like(optical_power)
        field_amplitude = torch.sqrt(optical_power)
        
        return field_amplitude * torch.exp(1j * random_phase)
        
    def get_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Get LED spectrum (Gaussian approximation)."""
        sigma_f = 3e8 * self.spectral_width / (2 * self.center_wavelength**2)
        delta_f = frequencies - self.center_frequency
        
        spectrum = self.power_watts * np.exp(-0.5 * (delta_f / sigma_f)**2)
        return spectrum / np.sqrt(2 * np.pi * sigma_f**2)
        
    def calculate_power_consumption(self) -> float:
        """Calculate electrical power consumption."""
        # Typical wall-plug efficiency for LED
        efficiency = 0.20  # 20%
        return self.power_watts / efficiency
        
    def to_netlist(self) -> dict:
        return {
            "type": "led_source",
            "center_wavelength": self.center_wavelength,
            "spectral_width": self.spectral_width,
            "power_dbm": self.power_dbm
        }