"""
Feedforward photonic neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from ..core import PhotonicComponent, PhaseShifter
from ..core.registry import register_component


@register_component
class MZIMesh(PhotonicComponent):
    """Universal Mach-Zehnder interferometer mesh."""
    
    def __init__(self, size: Tuple[int, int] = (8, 8), 
                 topology: str = "rectangular",
                 phase_encoding: str = "differential"):
        super().__init__()
        self.size = size
        self.topology = topology
        self.phase_encoding = phase_encoding
        
        # Create phase shifter arrays
        n_inputs, n_outputs = size
        if topology == "rectangular":
            self.n_phases = n_inputs * n_outputs
        else:
            self.n_phases = n_inputs * (n_inputs + 1) // 2
            
        # Initialize phase shifters
        self.phase_shifters = nn.ModuleList([
            PhaseShifter(length=50e-6, shifter_type="thermal")
            for _ in range(self.n_phases)
        ])
        
        # Initialize phases
        self.phases = nn.Parameter(torch.zeros(self.n_phases))
        
    def decompose(self, target_unitary: torch.Tensor) -> torch.Tensor:
        """Decompose unitary matrix into phase shifter settings."""
        if target_unitary.shape != self.size:
            raise ValueError(f"Unitary shape {target_unitary.shape} doesn't match mesh size {self.size}")
            
        # Simplified decomposition (Clements et al.)
        phases = torch.zeros(self.n_phases)
        
        if self.topology == "rectangular":
            # Rectangular mesh decomposition
            U = target_unitary.clone()
            phase_idx = 0
            
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    if i != j:
                        # Extract phase from matrix element
                        angle = torch.angle(U[i, j])
                        phases[phase_idx] = angle
                        phase_idx += 1
                        
        return phases
        
    def set_phases(self, phases: torch.Tensor):
        """Set phase shifter values."""
        if len(phases) != self.n_phases:
            raise ValueError(f"Expected {self.n_phases} phases, got {len(phases)}")
        self.phases.data = phases
        
        # Update individual phase shifters
        for i, phase_shifter in enumerate(self.phase_shifters):
            phase_shifter.drive_voltage = phases[i].item() / phase_shifter.efficiency
            
    def forward(self, input_fields: torch.Tensor) -> torch.Tensor:
        """Forward pass through MZI mesh."""
        batch_size = input_fields.shape[0]
        n_inputs = input_fields.shape[1]
        
        if n_inputs != self.size[0]:
            raise ValueError(f"Expected {self.size[0]} inputs, got {n_inputs}")
            
        # Apply unitary transformation
        unitary_matrix = self.compute_unitary_matrix()
        output = torch.matmul(input_fields, unitary_matrix.T)
        
        return output
        
    def compute_unitary_matrix(self) -> torch.Tensor:
        """Compute the implemented unitary matrix."""
        # Simplified: reconstruct from phases
        matrix = torch.eye(self.size[0], dtype=torch.complex64)
        
        # Apply phase shifts according to topology
        phase_idx = 0
        for i in range(self.size[0]):
            for j in range(i+1, self.size[0]):
                # Beam splitter + phase shifter
                theta = self.phases[phase_idx]
                phi = self.phases[phase_idx + 1] if phase_idx + 1 < self.n_phases else 0
                
                # 2x2 unitary for MZI
                cos_theta = torch.cos(theta / 2)
                sin_theta = torch.sin(theta / 2)
                exp_phi = torch.exp(1j * phi)
                
                mzi_matrix = torch.tensor([
                    [cos_theta, -sin_theta * exp_phi],
                    [sin_theta * exp_phi.conj(), cos_theta]
                ], dtype=torch.complex64)
                
                # Apply to full matrix (simplified)
                matrix[i, i] *= cos_theta
                matrix[j, j] *= cos_theta
                matrix[i, j] = -sin_theta * exp_phi
                matrix[j, i] = sin_theta * exp_phi.conj()
                
                phase_idx += 2
                
        return matrix
        
    def measure_unitary(self) -> torch.Tensor:
        """Measure the implemented unitary matrix."""
        return self.compute_unitary_matrix()
        
    def calculate_power_consumption(self) -> float:
        """Calculate total power consumption."""
        total_power = 0
        for phase_shifter in self.phase_shifters:
            total_power += phase_shifter.calculate_power_consumption()
        return total_power
        
    def to_netlist(self) -> dict:
        return {
            "type": "mzi_mesh",
            "size": self.size,
            "topology": self.topology,
            "phase_encoding": self.phase_encoding,
            "phases": self.phases.detach().numpy().tolist()
        }


@register_component
class PhotonicMLP(PhotonicComponent):
    """Multi-layer photonic perceptron."""
    
    def __init__(self, layer_sizes: List[int], activation: str = "relu"):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # Create MZI meshes for each layer
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            mesh = MZIMesh(size=(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(mesh)
            
        # Nonlinear activation (simplified)
        if activation == "relu":
            self.activation_fn = self._photonic_relu
        else:
            self.activation_fn = lambda x: x
            
    def _photonic_relu(self, x: torch.Tensor) -> torch.Tensor:
        """Photonic ReLU using saturable absorption."""
        # Simplified model - in practice would use nonlinear materials
        intensity = torch.abs(x)**2
        threshold = 0.1
        
        # Saturable absorption model
        absorption = 1 / (1 + intensity / threshold)
        output_intensity = intensity * absorption
        
        # Maintain phase information
        phase = torch.angle(x)
        return torch.sqrt(output_intensity) * torch.exp(1j * phase)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through photonic MLP."""
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer(current)
            
            # Apply activation except on last layer
            if i < len(self.layers) - 1:
                current = self.activation_fn(current)
                
        return current
        
    def train_optical_backprop(self, inputs: torch.Tensor, targets: torch.Tensor,
                             learning_rate: float = 0.01):
        """Train using optical backpropagation."""
        # Forward pass
        outputs = self.forward(inputs)
        loss = torch.mean(torch.abs(outputs - targets)**2)
        
        # Optical backpropagation (simplified)
        # In practice would use phase-conjugate reference beams
        loss.backward()
        
        # Update phases using gradients
        with torch.no_grad():
            for layer in self.layers:
                if layer.phases.grad is not None:
                    layer.phases -= learning_rate * layer.phases.grad
                    layer.phases.grad.zero_()
                    layer.set_phases(layer.phases)
                    
        return loss.item()
        
    def to_netlist(self) -> dict:
        return {
            "type": "photonic_mlp",
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "layers": [layer.to_netlist() for layer in self.layers]
        }