"""
Feedforward photonic neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from ..core import PhotonicComponent
# Import will be handled dynamically
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
        
        # Initialize phase parameters
        self.phases = nn.Parameter(torch.zeros(self.n_phases))
        self.phase_variance = nn.Parameter(torch.ones(self.n_phases) * 0.1)
            
        # Initialize phase shifters - use simplified model to avoid import issues
        self.thermal_coefficients = nn.Parameter(torch.ones(self.n_phases) * 2.0e-4)
        self.drive_voltages = nn.Parameter(torch.zeros(self.n_phases))
        
        # Initialize phases
        self.phases = nn.Parameter(torch.zeros(self.n_phases))
        
    def decompose(self, target_unitary: torch.Tensor) -> torch.Tensor:
        """Decompose unitary matrix into phase shifter settings using Clements decomposition."""
        if target_unitary.shape[0] != target_unitary.shape[1]:
            raise ValueError("Input must be a square matrix")
            
        n = target_unitary.shape[0]
        U = target_unitary.clone().to(torch.complex64)
        
        # Storage for MZI parameters
        theta_phases = []  # Beam splitter angles
        phi_phases = []   # Phase shifter angles
        
        # Clements decomposition algorithm
        # Process columns from left to right
        for col in range(n - 1):
            # Process rows from bottom to top in each column
            for row in range(n - 1, col, -1):
                # Use Givens rotation to zero out U[row, col]
                if torch.abs(U[row, col]) > 1e-10:  # Only if element is significant
                    # Calculate rotation parameters
                    a = U[row - 1, col]
                    b = U[row, col]
                    
                    if torch.abs(a) < 1e-10:
                        theta = torch.pi / 2
                        phi = torch.angle(b)
                    elif torch.abs(b) < 1e-10:
                        theta = 0.0
                        phi = torch.angle(a)
                    else:
                        # Calculate beam splitter angle
                        r = torch.abs(b) / torch.abs(a)
                        theta = torch.atan(r)
                        
                        # Calculate phase difference
                        phi = torch.angle(b) - torch.angle(a)
                        
                    theta_phases.append(theta.item())
                    phi_phases.append(phi.item())
                    
                    # Apply Givens rotation
                    cos_theta = torch.cos(theta)
                    sin_theta = torch.sin(theta)
                    exp_phi = torch.exp(1j * phi)
                    
                    # Create rotation matrix
                    rotation = torch.eye(n, dtype=torch.complex64)
                    rotation[row - 1, row - 1] = cos_theta
                    rotation[row - 1, row] = -sin_theta * exp_phi.conj()
                    rotation[row, row - 1] = sin_theta
                    rotation[row, row] = cos_theta * exp_phi.conj()
                    
                    # Apply rotation: U = R * U
                    U = torch.matmul(rotation, U)
                else:
                    # Add identity transformation
                    theta_phases.append(0.0)
                    phi_phases.append(0.0)
        
        # Add final phase shifters (diagonal elements)
        final_phases = []
        for i in range(n):
            final_phases.append(torch.angle(U[i, i]).item())
            
        # Combine all phases into single tensor
        all_phases = torch.tensor(theta_phases + phi_phases + final_phases)
        
        # Pad or truncate to match expected number of phases
        if len(all_phases) < self.n_phases:
            padded_phases = torch.zeros(self.n_phases)
            padded_phases[:len(all_phases)] = all_phases
            all_phases = padded_phases
        elif len(all_phases) > self.n_phases:
            all_phases = all_phases[:self.n_phases]
            
        return all_phases
        
    def set_phases(self, phases: torch.Tensor):
        """Set phase shifter values."""
        if len(phases) != self.n_phases:
            raise ValueError(f"Expected {self.n_phases} phases, got {len(phases)}")
        self.phases.data = phases
        
        # Update thermal coefficients for phase control
        for i in range(len(phases)):
            if i < len(self.drive_voltages):
                self.drive_voltages.data[i] = phases[i] / self.thermal_coefficients[i]
            
    def forward(self, input_fields: torch.Tensor) -> torch.Tensor:
        """Forward pass through MZI mesh."""
        # Handle different input dimensions
        if input_fields.dim() == 1:
            input_fields = input_fields.unsqueeze(0)  # Add batch dimension
            
        batch_size = input_fields.shape[0]
        n_inputs = input_fields.shape[1] if input_fields.dim() > 1 else len(input_fields)
        
        if n_inputs != self.size[0]:
            raise ValueError(f"Expected {self.size[0]} inputs, got {n_inputs}")
            
        # Apply unitary transformation
        unitary_matrix = self.compute_unitary_matrix()
        output = torch.matmul(input_fields, unitary_matrix.T)
        
        return output
        
    def compute_unitary_matrix(self) -> torch.Tensor:
        """Compute the implemented unitary matrix from current phase settings."""
        n = self.size[0]
        U = torch.eye(n, dtype=torch.complex64)
        
        if self.topology == "rectangular":
            # Reconstruct using rectangular mesh structure
            phase_idx = 0
            
            # Apply layers of MZIs
            for layer in range(n - 1):
                for mzi in range(n - 1 - layer):
                    row = n - 1 - mzi
                    
                    if phase_idx < self.n_phases - 1:
                        # Get beam splitter and phase shifter angles
                        theta = self.phases[phase_idx]
                        phi = self.phases[phase_idx + 1] if phase_idx + 1 < self.n_phases else 0
                        
                        # Create MZI transformation matrix
                        cos_theta = torch.cos(theta)
                        sin_theta = torch.sin(theta)
                        exp_phi = torch.exp(1j * phi)
                        
                        # MZI matrix for rows (row-1, row)
                        mzi_transform = torch.eye(n, dtype=torch.complex64)
                        mzi_transform[row - 1, row - 1] = cos_theta
                        mzi_transform[row - 1, row] = -sin_theta * exp_phi.conj()
                        mzi_transform[row, row - 1] = sin_theta
                        mzi_transform[row, row] = cos_theta * exp_phi.conj()
                        
                        # Apply transformation: U = mzi_transform * U
                        U = torch.matmul(mzi_transform, U)
                        
                        phase_idx += 2
                        
            # Apply final phase shifters (diagonal phases)
            if phase_idx < self.n_phases:
                remaining_phases = self.n_phases - phase_idx
                final_phases = self.phases[phase_idx:phase_idx + min(remaining_phases, n)]
                
                # Create diagonal phase matrix
                phase_matrix = torch.eye(n, dtype=torch.complex64)
                for i in range(min(len(final_phases), n)):
                    phase_matrix[i, i] = torch.exp(1j * final_phases[i])
                    
                U = torch.matmul(phase_matrix, U)
                
        else:  # triangular topology
            # Simplified triangular mesh implementation
            phase_idx = 0
            for layer in range(n):
                for i in range(layer):
                    if phase_idx < self.n_phases:
                        # Apply phase shift
                        phase = self.phases[phase_idx]
                        phase_matrix = torch.eye(n, dtype=torch.complex64)
                        phase_matrix[i, i] = torch.exp(1j * phase)
                        U = torch.matmul(phase_matrix, U)
                        phase_idx += 1
                        
        return U
        
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
                             learning_rate: float = 0.01, method: str = "adjoint"):
        """Train using optical backpropagation with phase-conjugate beams."""
        batch_size = inputs.shape[0]
        
        # Forward pass - store intermediate activations
        activations = []
        current = inputs
        
        for i, layer in enumerate(self.layers):
            current = layer(current)
            activations.append(current.clone())
            
            # Apply activation except on last layer
            if i < len(self.layers) - 1:
                current = self.activation_fn(current)
                
        outputs = current
        
        # Calculate loss
        loss = torch.mean(torch.abs(outputs - targets)**2)
        
        if method == "adjoint":
            return self._train_adjoint_method(inputs, targets, activations, learning_rate)
        elif method == "direct_feedback":
            return self._train_direct_feedback(inputs, targets, outputs, learning_rate)
        else:
            # Standard backpropagation for comparison
            return self._train_standard_backprop(loss, learning_rate)
            
    def _train_adjoint_method(self, inputs: torch.Tensor, targets: torch.Tensor, 
                            activations: list, learning_rate: float) -> float:
        """Train using adjoint method with phase-conjugate reference beams."""
        batch_size = inputs.shape[0]
        
        # Calculate output error
        output_error = activations[-1] - targets
        
        # Adjoint propagation (time-reversed, phase-conjugated)
        adjoint_fields = []
        current_adjoint = torch.conj(output_error)  # Phase conjugate
        
        # Propagate adjoint fields backward through layers
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]
            
            # Get layer's unitary matrix
            U = layer.compute_unitary_matrix()
            
            # Adjoint propagation: field travels backward with conjugate transpose
            U_adjoint = torch.conj(U.T)
            current_adjoint = torch.matmul(current_adjoint, U_adjoint)
            
            adjoint_fields.insert(0, current_adjoint.clone())
            
        # Calculate gradients using optical interference
        total_loss = 0
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                input_field = inputs
            else:
                input_field = activations[layer_idx - 1]
                
            # Calculate gradient via optical interference pattern
            gradient = self._calculate_optical_gradient(
                input_field, adjoint_fields[layer_idx], layer
            )
            
            # Update phases
            with torch.no_grad():
                layer.phases -= learning_rate * gradient
                layer.set_phases(layer.phases)
                
        # Calculate final loss
        with torch.no_grad():
            final_output = self.forward(inputs)
            total_loss = torch.mean(torch.abs(final_output - targets)**2).item()
            
        return total_loss
        
    def _calculate_optical_gradient(self, input_field: torch.Tensor, 
                                  adjoint_field: torch.Tensor, 
                                  layer: 'MZIMesh') -> torch.Tensor:
        """Calculate gradients using optical interference patterns."""
        batch_size = input_field.shape[0]
        n_phases = layer.n_phases
        gradients = torch.zeros(n_phases)
        
        # For each phase shifter, calculate sensitivity
        for phase_idx in range(n_phases):
            # Small phase perturbation for finite difference
            delta_phase = 1e-4
            
            # Perturb phase
            original_phase = layer.phases[phase_idx].item()
            layer.phases.data[phase_idx] = original_phase + delta_phase
            
            # Calculate perturbed output
            U_perturbed = layer.compute_unitary_matrix()
            output_perturbed = torch.matmul(input_field, U_perturbed.T)
            
            # Restore original phase
            layer.phases.data[phase_idx] = original_phase
            
            # Calculate gradient via interference
            # Simplified: use real part of interference pattern
            interference = torch.sum(torch.real(
                torch.conj(adjoint_field) * output_perturbed
            ))
            
            # Finite difference gradient
            gradients[phase_idx] = interference / delta_phase / batch_size
            
        return gradients
        
    def _train_direct_feedback(self, inputs: torch.Tensor, targets: torch.Tensor,
                             outputs: torch.Tensor, learning_rate: float) -> float:
        """Train using direct feedback alignment for photonic networks."""
        # Random feedback weights (fixed during training)
        if not hasattr(self, '_feedback_weights'):
            self._feedback_weights = []
            for layer in self.layers:
                n_out = layer.size[1] if hasattr(layer, 'size') else outputs.shape[-1]
                n_in = layer.size[0] if hasattr(layer, 'size') else inputs.shape[-1]
                feedback_weight = torch.randn(n_out, n_in) * 0.1
                self._feedback_weights.append(feedback_weight)
                
        # Calculate output error
        error = outputs - targets
        
        # Direct feedback to each layer
        for layer_idx, layer in enumerate(self.layers):
            # Random feedback matrix
            feedback_matrix = self._feedback_weights[layer_idx]
            
            # Calculate local error signal
            local_error = torch.matmul(error, feedback_matrix)
            
            # Convert to phase updates (simplified)
            if hasattr(layer, 'phases'):
                # Project error onto phase space
                phase_gradient = torch.mean(torch.real(local_error), dim=0)
                
                # Ensure gradient size matches phases
                if len(phase_gradient) > layer.n_phases:
                    phase_gradient = phase_gradient[:layer.n_phases]
                elif len(phase_gradient) < layer.n_phases:
                    padded_grad = torch.zeros(layer.n_phases)
                    padded_grad[:len(phase_gradient)] = phase_gradient
                    phase_gradient = padded_grad
                    
                # Update phases
                with torch.no_grad():
                    layer.phases -= learning_rate * phase_gradient
                    layer.set_phases(layer.phases)
                    
        # Calculate loss
        final_output = self.forward(inputs)
        loss = torch.mean(torch.abs(final_output - targets)**2).item()
        return loss
        
    def _train_standard_backprop(self, loss: torch.Tensor, learning_rate: float) -> float:
        """Standard backpropagation for comparison."""
        loss.backward()
        
        # Update phases using gradients
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, 'phases') and layer.phases.grad is not None:
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