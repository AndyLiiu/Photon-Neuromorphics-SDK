"""
Base classes for photonic components.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import json


class PhotonicComponent(nn.Module, ABC):
    """Base class for all photonic components."""
    
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self._parameters = {}
        self._losses_db = {}
        self._noise_sources = {}
        
    @abstractmethod
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Forward propagation of optical field."""
        pass
        
    @abstractmethod
    def to_netlist(self) -> Dict[str, Any]:
        """Convert component to netlist representation."""
        pass
        
    def add_loss(self, loss_db: float, wavelength: float = 1550e-9):
        """Add insertion loss."""
        self._losses_db[wavelength] = loss_db
        
    def add_noise_source(self, noise_type: str, parameters: Dict[str, float]):
        """Add noise source to component."""
        self._noise_sources[noise_type] = parameters
        
    def get_loss_linear(self, wavelength: float = 1550e-9) -> float:
        """Get linear loss coefficient."""
        loss_db = self._losses_db.get(wavelength, 0.0)
        return 10**(-loss_db/20)
        
    def calculate_power_consumption(self) -> float:
        """Calculate electrical power consumption in watts."""
        return 0.0  # Override in subclasses
        
    def get_s_parameters(self, frequencies: np.ndarray) -> np.ndarray:
        """Get S-parameters for component."""
        return np.eye(2)  # Default to unity transmission


class WaveguideBase(PhotonicComponent):
    """Base class for optical waveguides."""
    
    def __init__(self, length: float, width: float = 450e-9, 
                 material: str = "silicon", name: str = None):
        super().__init__(name)
        self.length = length
        self.width = width
        self.material = material
        self.n_eff = self._calculate_effective_index()
        self.loss_db_per_cm = 0.1  # Default loss
        
    def _calculate_effective_index(self) -> float:
        """Calculate effective refractive index."""
        if self.material == "silicon":
            return 2.4  # Approximate for TE mode at 1550nm
        elif self.material == "silicon_nitride":
            return 1.9
        else:
            return 1.5
            
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Propagate field through waveguide."""
        # Apply phase shift due to propagation
        beta = 2 * np.pi * self.n_eff / 1550e-9
        phase = beta * self.length
        
        # Apply loss
        loss_linear = 10**(-self.loss_db_per_cm * self.length * 100 / 20)
        
        return input_field * loss_linear * torch.exp(1j * phase)
        
    def to_netlist(self) -> Dict[str, Any]:
        return {
            "type": "waveguide",
            "material": self.material,
            "length": self.length,
            "width": self.width,
            "loss_db_per_cm": self.loss_db_per_cm
        }


class ModulatorBase(PhotonicComponent):
    """Base class for optical modulators."""
    
    def __init__(self, modulation_type: str = "phase", name: str = None):
        super().__init__(name)
        self.modulation_type = modulation_type
        self.drive_voltage = 0.0
        self.v_pi = 1.0  # Voltage for pi phase shift
        
    def set_drive_voltage(self, voltage: float):
        """Set the drive voltage."""
        self.drive_voltage = voltage
        
    def get_modulation_response(self) -> float:
        """Get modulation response for current drive voltage."""
        if self.modulation_type == "phase":
            return np.pi * self.drive_voltage / self.v_pi
        elif self.modulation_type == "amplitude":
            return np.exp(-self.drive_voltage / self.v_pi)
        else:
            return 1.0
            
    @abstractmethod
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        pass


class DetectorBase(PhotonicComponent):
    """Base class for photodetectors."""
    
    def __init__(self, responsivity: float = 1.0, dark_current: float = 1e-9,
                 bandwidth: float = 1e9, name: str = None):
        super().__init__(name)
        self.responsivity = responsivity  # A/W
        self.dark_current = dark_current  # A
        self.bandwidth = bandwidth  # Hz
        self.noise_equivalent_power = 1e-12  # W/√Hz
        
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Convert optical power to electrical current."""
        optical_power = torch.abs(input_field)**2
        photocurrent = self.responsivity * optical_power
        
        # Add shot noise and thermal noise (simplified)
        if self.training:
            noise_std = np.sqrt(2 * 1.602e-19 * photocurrent.mean() * self.bandwidth)
            noise = torch.randn_like(photocurrent) * noise_std
            photocurrent = photocurrent + noise
            
        return photocurrent + self.dark_current
        
    def to_netlist(self) -> Dict[str, Any]:
        return {
            "type": "photodetector",
            "responsivity": self.responsivity,
            "dark_current": self.dark_current,
            "bandwidth": self.bandwidth
        }