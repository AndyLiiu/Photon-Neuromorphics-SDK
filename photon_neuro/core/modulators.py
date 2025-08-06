"""
Optical modulator implementations.
"""

import torch
import numpy as np
from .components import ModulatorBase
from .registry import register_component


@register_component
class MachZehnderModulator(ModulatorBase):
    """Mach-Zehnder interferometer based modulator."""
    
    def __init__(self, length: float = 2e-3, v_pi: float = 1.5):
        super().__init__("amplitude")
        self.length = length
        self.v_pi = v_pi
        self.extinction_ratio = 30  # dB
        self.insertion_loss = 3     # dB
        
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """MZI modulation."""
        # Split input equally
        field_1 = input_field / np.sqrt(2)
        field_2 = input_field / np.sqrt(2)
        
        # Apply phase shift to one arm
        phase_shift = np.pi * self.drive_voltage / self.v_pi
        field_2 = field_2 * torch.exp(1j * phase_shift)
        
        # Recombine with interference
        output = (field_1 + field_2) / np.sqrt(2)
        
        # Apply insertion loss
        loss_linear = 10**(-self.insertion_loss/20)
        return output * loss_linear
        
    def calculate_power_consumption(self) -> float:
        """Calculate RF power consumption."""
        # Simplified model: P = CV²f where C is capacitance
        capacitance = 100e-15 * self.length / 1e-3  # 100 fF/mm
        frequency = 1e9  # 1 GHz modulation
        return capacitance * self.drive_voltage**2 * frequency
        
    def to_netlist(self) -> dict:
        return {
            "type": "mzi_modulator",
            "length": self.length,
            "v_pi": self.v_pi,
            "extinction_ratio": self.extinction_ratio,
            "insertion_loss": self.insertion_loss
        }


@register_component
class MicroringModulator(ModulatorBase):
    """Microring resonator based modulator."""
    
    def __init__(self, radius: float = 5e-6, quality_factor: float = 10000,
                 coupling_coefficient: float = 0.1):
        super().__init__("amplitude") 
        self.radius = radius
        self.q_factor = quality_factor
        self.coupling = coupling_coefficient
        self.v_pi = 0.5  # Lower voltage for microring
        
    def calculate_resonance_wavelength(self) -> float:
        """Calculate resonance wavelength."""
        n_eff = 2.4  # Silicon effective index
        return 2 * np.pi * self.radius * n_eff
        
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Microring transmission."""
        # Voltage-induced resonance shift
        wavelength = 1550e-9
        resonance_shift = -50e-9 * self.drive_voltage / self.v_pi  # pm/V
        detuning = 2 * np.pi * 3e8 / wavelength**2 * resonance_shift
        
        # Ring transmission (simplified)
        t = np.sqrt(1 - self.coupling)
        r = np.sqrt(self.coupling)
        
        # Round-trip phase
        beta = 2 * np.pi / wavelength * 2.4  # Simplified
        phi = beta * 2 * np.pi * self.radius + detuning
        
        # Add loss
        a = np.exp(-np.pi / self.q_factor)
        
        # Transfer function
        transmission = (t - a * np.exp(1j * phi)) / (1 - a * t * np.exp(1j * phi))
        
        return input_field * transmission
        
    def to_netlist(self) -> dict:
        return {
            "type": "microring_modulator", 
            "radius": self.radius,
            "quality_factor": self.q_factor,
            "coupling": self.coupling,
            "v_pi": self.v_pi
        }


@register_component
class PhaseShifter(ModulatorBase):
    """Thermo-optic or electro-optic phase shifter."""
    
    def __init__(self, length: float = 100e-6, efficiency: float = 10e-6,
                 shifter_type: str = "thermal"):
        super().__init__("phase")
        self.length = length
        self.efficiency = efficiency  # rad/V or rad/mW
        self.shifter_type = shifter_type
        self.response_time = 1e-6 if shifter_type == "thermal" else 1e-12
        
    def forward(self, input_field: torch.Tensor) -> torch.Tensor:
        """Apply phase shift."""
        if self.shifter_type == "thermal":
            # Thermo-optic: drive_voltage represents power in mW
            phase_shift = self.efficiency * self.drive_voltage
        else:
            # Electro-optic: direct voltage control
            phase_shift = self.efficiency * self.drive_voltage
            
        return input_field * torch.exp(1j * phase_shift)
        
    def calculate_power_consumption(self) -> float:
        """Calculate power consumption."""
        if self.shifter_type == "thermal":
            return self.drive_voltage * 1e-3  # Convert mW to W
        else:
            # Electro-optic has negligible static power
            return 1e-6  # 1 μW
            
    def to_netlist(self) -> dict:
        return {
            "type": "phase_shifter",
            "shifter_type": self.shifter_type,
            "length": self.length,
            "efficiency": self.efficiency,
            "response_time": self.response_time
        }