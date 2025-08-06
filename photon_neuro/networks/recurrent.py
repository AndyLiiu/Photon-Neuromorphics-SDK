"""
Recurrent photonic neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from ..core import PhotonicComponent, MicroringModulator
from ..core.registry import register_component


@register_component  
class MicroringArray(PhotonicComponent):
    """Array of microring resonators for weight encoding."""
    
    def __init__(self, n_rings: int, free_spectral_range: float = 20e9,
                 quality_factor: float = 10000, tuning: str = "thermal"):
        super().__init__()
        self.n_rings = n_rings
        self.fsr = free_spectral_range
        self.q_factor = quality_factor
        self.tuning = tuning
        
        # Create array of microring modulators
        self.rings = nn.ModuleList([
            MicroringModulator(
                radius=np.random.uniform(4e-6, 6e-6),  # Radius variation
                quality_factor=quality_factor,
                coupling_coefficient=0.1
            ) for _ in range(n_rings)
        ])
        
        # Weight encoding parameters
        self.weights = nn.Parameter(torch.randn(n_rings) * 0.1)
        self.resonance_map = nn.Parameter(torch.linspace(0, 1, n_rings))
        
    def encode_weights(self, weights: np.ndarray, encoding: str = "wavelength_division") -> np.ndarray:
        """Map neural network weights to microring resonances."""
        if encoding == "wavelength_division":
            # Map weights to wavelength channels
            base_wavelength = 1550e-9
            channel_spacing = 0.8e-9  # 100 GHz spacing
            
            resonances = []
            for i, weight in enumerate(weights.flatten()):
                wavelength = base_wavelength + i * channel_spacing
                resonances.append(wavelength)
                
                # Set ring resonance to match wavelength
                if i < len(self.rings):
                    # Thermal tuning to shift resonance
                    ring = self.rings[i]
                    target_shift = (wavelength - base_wavelength) / base_wavelength
                    tuning_power = target_shift * 10e-3  # mW
                    ring.set_drive_voltage(tuning_power)
                    
            return np.array(resonances)
            
        elif encoding == "amplitude_modulation":
            # Map weights to ring coupling strengths
            for i, weight in enumerate(weights.flatten()):
                if i < len(self.rings):
                    # Normalize weight to coupling range [0, 1]
                    normalized_weight = (weight + 1) / 2  # Assume weights in [-1, 1]
                    coupling = np.clip(normalized_weight, 0.01, 0.99)
                    self.rings[i].coupling = coupling
                    
        return weights
        
    def add_variations(self, radius_sigma: float = 5e-9, coupling_sigma: float = 0.02):
        """Add fabrication variations to rings."""
        with torch.no_grad():
            for ring in self.rings:
                # Radius variation
                radius_variation = np.random.normal(0, radius_sigma)
                ring.radius += radius_variation
                
                # Coupling variation  
                coupling_variation = np.random.normal(0, coupling_sigma)
                ring.coupling = np.clip(ring.coupling + coupling_variation, 0.01, 0.99)
                
    def forward(self, input_fields: torch.Tensor) -> torch.Tensor:
        """Process optical signals through microring array."""
        if input_fields.shape[-1] != self.n_rings:
            # Broadcast input to all rings if needed
            input_fields = input_fields.unsqueeze(-1).expand(-1, self.n_rings)
            
        outputs = []
        for i, ring in enumerate(self.rings):
            # Apply weight via ring transmission
            ring_output = ring.forward(input_fields[..., i])
            outputs.append(ring_output)
            
        return torch.stack(outputs, dim=-1)
        
    def calibrate_resonances(self, target_wavelengths: List[float]):
        """Calibrate ring resonances to target wavelengths."""
        for i, (ring, target_wl) in enumerate(zip(self.rings, target_wavelengths)):
            # Calculate required tuning
            current_resonance = ring.calculate_resonance_wavelength()
            wavelength_shift = target_wl - current_resonance
            
            # Convert to thermal power (simplified)
            thermal_efficiency = 10e-9  # nm/mW
            required_power = wavelength_shift / thermal_efficiency
            ring.set_drive_voltage(required_power)
            
    def to_netlist(self) -> dict:
        return {
            "type": "microring_array",
            "n_rings": self.n_rings,
            "fsr": self.fsr,
            "q_factor": self.q_factor,
            "tuning": self.tuning,
            "rings": [ring.to_netlist() for ring in self.rings]
        }


@register_component
class PhotonicReservoir(PhotonicComponent):
    """Photonic reservoir computing network."""
    
    def __init__(self, reservoir_size: int = 100, spectral_radius: float = 0.95,
                 input_scaling: float = 0.1, leak_rate: float = 0.3):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling  
        self.leak_rate = leak_rate
        
        # Reservoir connectivity via microring couplers
        self.reservoir_rings = MicroringArray(
            n_rings=reservoir_size,
            quality_factor=5000,
            tuning="thermal"
        )
        
        # Random reservoir connections
        connectivity = 0.1  # 10% connectivity
        reservoir_weights = np.random.randn(reservoir_size, reservoir_size) * connectivity
        mask = np.random.random((reservoir_size, reservoir_size)) > connectivity
        reservoir_weights[mask] = 0
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(reservoir_weights)
        current_radius = np.max(np.abs(eigenvalues))
        reservoir_weights = reservoir_weights * (spectral_radius / current_radius)
        
        self.reservoir_weights = nn.Parameter(
            torch.tensor(reservoir_weights, dtype=torch.float32), 
            requires_grad=False
        )
        
        # Input weights
        self.input_weights = nn.Parameter(
            torch.randn(reservoir_size, 1) * input_scaling,
            requires_grad=False
        )
        
        # Reservoir state
        self.register_buffer('state', torch.zeros(reservoir_size))
        
    def forward(self, inputs: torch.Tensor, return_states: bool = False) -> torch.Tensor:
        """Process input sequence through photonic reservoir."""
        batch_size, seq_len, input_dim = inputs.shape
        
        # Initialize states
        states = torch.zeros(batch_size, seq_len, self.reservoir_size, device=inputs.device)
        current_state = torch.zeros(batch_size, self.reservoir_size, device=inputs.device)
        
        # Process sequence
        for t in range(seq_len):
            # Input injection (optical)
            input_contrib = torch.matmul(inputs[:, t:t+1], self.input_weights.T)
            
            # Reservoir recurrence (via optical coupling)
            reservoir_contrib = torch.matmul(current_state, self.reservoir_weights.T)
            
            # Leaky integration with nonlinearity
            pre_activation = input_contrib.squeeze(1) + reservoir_contrib
            
            # Photonic nonlinearity (saturable absorption)
            optical_power = torch.abs(pre_activation)**2
            saturation_power = 1.0
            nonlinear_response = optical_power / (1 + optical_power / saturation_power)
            
            # Convert back to field amplitude
            activation = torch.sqrt(nonlinear_response) * torch.sign(pre_activation)
            
            # Leaky integration
            current_state = (1 - self.leak_rate) * current_state + self.leak_rate * activation
            states[:, t] = current_state
            
        if return_states:
            return states
        else:
            return current_state
            
    def train_readout(self, reservoir_states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Train linear readout layer."""
        # Ridge regression for readout weights
        batch_size, seq_len, reservoir_size = reservoir_states.shape
        
        # Reshape for batch processing
        X = reservoir_states.view(-1, reservoir_size)
        y = targets.view(-1, targets.shape[-1])
        
        # Ridge regression with regularization
        ridge_param = 1e-6
        XTX = torch.matmul(X.T, X)
        XTy = torch.matmul(X.T, y)
        
        # Add ridge regularization
        I = torch.eye(reservoir_size, device=X.device)
        readout_weights = torch.solve(XTy, XTX + ridge_param * I)[0]
        
        return readout_weights
        
    def reset_state(self):
        """Reset reservoir state."""
        self.state.zero_()
        
    def calculate_memory_capacity(self, max_delay: int = 50) -> float:
        """Calculate linear memory capacity of reservoir."""
        # Generate test signals
        seq_len = 1000
        test_input = torch.randn(1, seq_len, 1)
        
        # Get reservoir states
        states = self.forward(test_input, return_states=True)
        
        total_capacity = 0
        for delay in range(1, max_delay + 1):
            if delay >= seq_len:
                break
                
            # Target is delayed input
            target = test_input[:, :-delay, :]
            reservoir_states = states[:, delay:, :]
            
            # Calculate correlation
            X = reservoir_states.view(-1, self.reservoir_size)
            y = target.view(-1, 1)
            
            if X.shape[0] > 0:
                correlation = torch.corrcoef(torch.cat([X.T, y.T]))[:-1, -1]
                capacity = torch.sum(correlation**2).item()
                total_capacity += capacity
                
        return total_capacity
        
    def to_netlist(self) -> dict:
        return {
            "type": "photonic_reservoir",
            "reservoir_size": self.reservoir_size,
            "spectral_radius": self.spectral_radius,
            "input_scaling": self.input_scaling,
            "leak_rate": self.leak_rate
        }