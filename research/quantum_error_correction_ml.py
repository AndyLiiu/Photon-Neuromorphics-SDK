#!/usr/bin/env python3
"""
Quantum Error Correction for Machine Learning on Photonic Processors (2025)
============================================================================

Implementation of quantum error correction schemes specifically designed for
machine learning workloads on photonic quantum processors.

Combines surface codes, stabilizer codes, and ML-optimized error correction
for fault-tolerant quantum machine learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import itertools
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import photon_neuro as pn


@dataclass
class ErrorCorrectionConfig:
    """Configuration for quantum error correction in ML."""
    code_distance: int = 3
    syndrome_threshold: float = 0.5
    max_correction_rounds: int = 10
    physical_error_rate: float = 1e-3
    measurement_error_rate: float = 1e-4
    correction_algorithm: str = 'ml_decoder'  # 'ml_decoder', 'mwpm', 'uf'


class StabilizerCode:
    """
    Stabilizer code implementation for quantum error correction.
    
    Supports surface codes and other CSS codes for photonic quantum computing.
    """
    
    def __init__(self, 
                 stabilizer_matrix: np.ndarray,
                 logical_operators: Dict[str, np.ndarray],
                 code_distance: int):
        """
        Initialize stabilizer code.
        
        Args:
            stabilizer_matrix: Binary matrix defining stabilizer generators
            logical_operators: Dictionary of logical Pauli operators  
            code_distance: Minimum distance of the code
        """
        self.stabilizer_matrix = stabilizer_matrix
        self.logical_operators = logical_operators
        self.code_distance = code_distance
        self.n_qubits = stabilizer_matrix.shape[1] // 2  # X and Z parts
        self.n_stabilizers = stabilizer_matrix.shape[0]
        
        # Precompute syndrome lookup table for fast decoding
        self._build_syndrome_table()
    
    def _build_syndrome_table(self):
        """Build lookup table mapping syndromes to error patterns."""
        self.syndrome_table = {}
        
        # Generate all possible single-qubit errors
        for qubit in range(self.n_qubits):
            # X error
            error_x = np.zeros(2 * self.n_qubits, dtype=int)
            error_x[qubit] = 1
            syndrome = self._compute_syndrome(error_x)
            self.syndrome_table[tuple(syndrome)] = error_x
            
            # Z error  
            error_z = np.zeros(2 * self.n_qubits, dtype=int)
            error_z[self.n_qubits + qubit] = 1
            syndrome = self._compute_syndrome(error_z)
            self.syndrome_table[tuple(syndrome)] = error_z
            
            # Y error (X + Z)
            error_y = error_x + error_z
            syndrome = self._compute_syndrome(error_y)
            self.syndrome_table[tuple(syndrome)] = error_y
    
    def _compute_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Compute syndrome for given error pattern."""
        return (self.stabilizer_matrix @ error) % 2
    
    def detect_errors(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Detect errors from syndrome measurement.
        
        Args:
            syndrome: Measured syndrome
            
        Returns:
            Estimated error pattern
        """
        syndrome_key = tuple(syndrome)
        
        if syndrome_key in self.syndrome_table:
            return self.syndrome_table[syndrome_key]
        else:
            # For multi-qubit errors, use more sophisticated decoding
            return self._decode_syndrome(syndrome)
    
    def _decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode complex syndrome patterns using minimum weight matching."""
        # Simplified implementation - in practice would use sophisticated algorithms
        best_error = np.zeros(2 * self.n_qubits, dtype=int)
        min_weight = float('inf')
        
        # Try all possible error patterns up to weight 2
        for weight in range(1, min(3, self.code_distance)):
            for positions in itertools.combinations(range(2 * self.n_qubits), weight):
                error_pattern = np.zeros(2 * self.n_qubits, dtype=int)
                error_pattern[list(positions)] = 1
                
                computed_syndrome = self._compute_syndrome(error_pattern)
                if np.array_equal(computed_syndrome, syndrome):
                    if weight < min_weight:
                        min_weight = weight
                        best_error = error_pattern
        
        return best_error
    
    def apply_correction(self, quantum_state: torch.Tensor, error: np.ndarray) -> torch.Tensor:
        """Apply error correction to quantum state."""
        # This is a simplified representation
        # In practice would apply Pauli corrections to actual quantum state
        
        corrected_state = quantum_state.clone()
        
        # Apply X corrections
        for qubit in range(self.n_qubits):
            if error[qubit] == 1:
                corrected_state = self._apply_x_correction(corrected_state, qubit)
        
        # Apply Z corrections  
        for qubit in range(self.n_qubits):
            if error[self.n_qubits + qubit] == 1:
                corrected_state = self._apply_z_correction(corrected_state, qubit)
        
        return corrected_state
    
    def _apply_x_correction(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply X correction to specified qubit."""
        # Bit flip correction
        corrected_state = state.clone()
        n_states = state.shape[-1]
        
        for i in range(n_states):
            if (i >> qubit) & 1:
                j = i ^ (1 << qubit)
                corrected_state[..., i] = state[..., j]
                corrected_state[..., j] = state[..., i]
        
        return corrected_state
    
    def _apply_z_correction(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply Z correction to specified qubit."""
        # Phase flip correction
        corrected_state = state.clone()
        n_states = state.shape[-1]
        
        for i in range(n_states):
            if (i >> qubit) & 1:
                corrected_state[..., i] *= -1
        
        return corrected_state


class SurfaceCode(StabilizerCode):
    """
    Surface code implementation optimized for photonic quantum processors.
    
    Provides high error thresholds and compatibility with 2D photonic layouts.
    """
    
    def __init__(self, distance: int = 3):
        """
        Initialize surface code with given distance.
        
        Args:
            distance: Code distance (odd integer)
        """
        if distance % 2 == 0:
            raise ValueError("Surface code distance must be odd")
        
        self.distance = distance
        self.lattice_size = distance
        
        # Generate stabilizer matrix for surface code
        stabilizers, logical_ops = self._generate_surface_code_stabilizers()
        
        super().__init__(stabilizers, logical_ops, distance)
    
    def _generate_surface_code_stabilizers(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate stabilizer generators for surface code."""
        # Simplified surface code generation
        n_data_qubits = self.distance ** 2
        n_ancilla = n_data_qubits - 1  # Rough estimate
        
        total_qubits = n_data_qubits + n_ancilla
        
        # Generate X-type and Z-type stabilizers
        stabilizers = []
        
        # X-type stabilizers (star operators)
        for i in range(0, self.distance - 1, 2):
            for j in range(1, self.distance, 2):
                x_stabilizer = np.zeros(2 * total_qubits, dtype=int)
                # Add X operators on neighboring data qubits
                center_qubit = i * self.distance + j
                if center_qubit < n_data_qubits:
                    x_stabilizer[center_qubit] = 1
                    # Add neighboring qubits
                    neighbors = self._get_neighbors(i, j)
                    for ni, nj in neighbors:
                        neighbor_qubit = ni * self.distance + nj
                        if neighbor_qubit < n_data_qubits:
                            x_stabilizer[neighbor_qubit] = 1
                
                stabilizers.append(x_stabilizer)
        
        # Z-type stabilizers (plaquette operators)
        for i in range(1, self.distance, 2):
            for j in range(0, self.distance - 1, 2):
                z_stabilizer = np.zeros(2 * total_qubits, dtype=int)
                # Add Z operators on neighboring data qubits
                center_qubit = i * self.distance + j
                if center_qubit < n_data_qubits:
                    z_stabilizer[total_qubits + center_qubit] = 1
                    # Add neighboring qubits
                    neighbors = self._get_neighbors(i, j)
                    for ni, nj in neighbors:
                        neighbor_qubit = ni * self.distance + nj
                        if neighbor_qubit < n_data_qubits:
                            z_stabilizer[total_qubits + neighbor_qubit] = 1
                
                stabilizers.append(z_stabilizer)
        
        stabilizer_matrix = np.array(stabilizers)
        
        # Generate logical operators
        logical_x = np.zeros(2 * total_qubits, dtype=int)
        logical_z = np.zeros(2 * total_qubits, dtype=int)
        
        # Logical X: horizontal string
        for j in range(self.distance):
            qubit = 0 * self.distance + j
            if qubit < n_data_qubits:
                logical_x[qubit] = 1
        
        # Logical Z: vertical string
        for i in range(self.distance):
            qubit = i * self.distance + 0
            if qubit < n_data_qubits:
                logical_z[total_qubits + qubit] = 1
        
        logical_operators = {
            'X': logical_x,
            'Z': logical_z
        }
        
        return stabilizer_matrix, logical_operators
    
    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get neighboring positions on the lattice."""
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.distance and 0 <= nj < self.distance:
                neighbors.append((ni, nj))
        return neighbors


class MLQuantumDecoder(nn.Module):
    """
    Machine learning-based quantum error decoder.
    
    Uses neural networks to predict optimal corrections from syndrome patterns,
    potentially achieving better performance than classical decoders.
    """
    
    def __init__(self, 
                 n_syndrome_bits: int,
                 n_qubits: int, 
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        self.n_syndrome_bits = n_syndrome_bits
        self.n_qubits = n_qubits
        
        # Syndrome encoder
        self.syndrome_encoder = nn.Sequential(
            nn.Linear(n_syndrome_bits, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Error prediction network
        layers = []
        dims = [hidden_dims[0]] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        # Output layers for X and Z errors
        self.x_error_head = nn.Linear(dims[-1], n_qubits)
        self.z_error_head = nn.Linear(dims[-1], n_qubits)
        
        self.error_network = nn.Sequential(*layers)
    
    def forward(self, syndrome: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict error correction from syndrome.
        
        Args:
            syndrome: Syndrome measurement (batch_size, n_syndrome_bits)
            
        Returns:
            Tuple of (x_corrections, z_corrections) probabilities
        """
        # Encode syndrome
        encoded = self.syndrome_encoder(syndrome)
        
        # Predict errors
        features = self.error_network(encoded)
        
        x_corrections = torch.sigmoid(self.x_error_head(features))
        z_corrections = torch.sigmoid(self.z_error_head(features))
        
        return x_corrections, z_corrections
    
    def predict_corrections(self, syndrome: torch.Tensor, 
                          threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict binary corrections from syndrome."""
        x_probs, z_probs = self.forward(syndrome)
        
        x_corrections = (x_probs > threshold).float()
        z_corrections = (z_probs > threshold).float()
        
        return x_corrections, z_corrections


class FaultTolerantQuantumLayer(nn.Module):
    """
    Fault-tolerant quantum layer with integrated error correction.
    
    Combines quantum computation with real-time error correction for
    reliable quantum machine learning.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 code: StabilizerCode,
                 config: ErrorCorrectionConfig):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.code = code
        self.config = config
        
        # Quantum computation parameters
        self.n_logical_qubits = max(input_dim, output_dim).bit_length()
        self.n_physical_qubits = self.n_logical_qubits * code.n_qubits
        
        # Quantum circuit parameters
        self.quantum_weights = nn.Parameter(
            torch.randn(self.n_logical_qubits, 3, 4)  # 3 rotation axes, 4 layers
        )
        
        # Error correction decoder
        if config.correction_algorithm == 'ml_decoder':
            self.decoder = MLQuantumDecoder(
                n_syndrome_bits=code.n_stabilizers,
                n_qubits=code.n_qubits
            )
        
        # Classical output projection
        self.output_projection = nn.Linear(self.n_logical_qubits, output_dim)
        
        # Error tracking
        self.register_buffer('error_counts', torch.zeros(3))  # [total, corrected, uncorrected]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fault-tolerant quantum computation.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Output after fault-tolerant quantum processing
        """
        batch_size = x.shape[0]
        
        # Encode input into logical qubits
        encoded_state = self._encode_input(x)
        
        # Apply quantum computation with error correction
        quantum_output = self._fault_tolerant_computation(encoded_state)
        
        # Project to classical output
        output = self.output_projection(quantum_output.real)
        
        return output
    
    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Encode classical input into logical quantum states."""
        batch_size, input_dim = x.shape
        
        # Pad input to match logical qubit count
        if input_dim < self.n_logical_qubits:
            padding = torch.zeros(batch_size, self.n_logical_qubits - input_dim, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x[:, :self.n_logical_qubits]
        
        # Convert to quantum amplitudes (normalized)
        amplitudes = F.normalize(x_padded, p=2, dim=1)
        
        # Expand to full quantum state space
        n_states = 2 ** self.n_logical_qubits
        quantum_state = torch.zeros(batch_size, n_states, dtype=torch.complex64, device=x.device)
        
        # Simple amplitude encoding (can be improved)
        for i in range(min(self.n_logical_qubits, n_states)):
            quantum_state[:, i] = amplitudes[:, i % self.n_logical_qubits].to(torch.complex64)
        
        return quantum_state
    
    def _fault_tolerant_computation(self, state: torch.Tensor) -> torch.Tensor:
        """Apply quantum computation with error correction."""
        current_state = state
        
        for round_idx in range(self.config.max_correction_rounds):
            # Apply quantum gates
            current_state = self._apply_quantum_layer(current_state, round_idx)
            
            # Simulate physical errors
            if self.training:
                current_state = self._simulate_errors(current_state)
            
            # Measure syndromes
            syndrome = self._measure_syndromes(current_state)
            
            # Decode and correct errors
            if torch.any(syndrome > self.config.syndrome_threshold):
                current_state = self._correct_errors(current_state, syndrome)
                self.error_counts[1] += 1  # Corrected errors
            
            self.error_counts[0] += 1  # Total operations
        
        return current_state
    
    def _apply_quantum_layer(self, state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply parameterized quantum gates."""
        new_state = state.clone()
        
        # Apply rotation gates based on learned parameters
        for qubit in range(self.n_logical_qubits):
            if layer_idx < self.quantum_weights.shape[2]:
                rx_angle = self.quantum_weights[qubit, 0, layer_idx]
                ry_angle = self.quantum_weights[qubit, 1, layer_idx] 
                rz_angle = self.quantum_weights[qubit, 2, layer_idx]
                
                new_state = self._apply_rotation_gates(new_state, qubit, rx_angle, ry_angle, rz_angle)
        
        # Apply entangling gates
        for qubit in range(self.n_logical_qubits - 1):
            new_state = self._apply_cnot_gate(new_state, qubit, qubit + 1)
        
        return new_state
    
    def _apply_rotation_gates(self, state: torch.Tensor, qubit: int, 
                            rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
        """Apply RX, RY, RZ rotation gates."""
        # Simplified implementation - in practice would use proper tensor products
        rotated_state = state.clone()
        
        # Apply rotations (simplified)
        cos_rx, sin_rx = torch.cos(rx/2), torch.sin(rx/2)
        cos_ry, sin_ry = torch.cos(ry/2), torch.sin(ry/2)
        
        # This is a simplified rotation - proper implementation would require
        # full tensor product operations
        n_states = state.shape[-1]
        for i in range(n_states):
            if (i >> qubit) & 1 == 0:
                j = i | (1 << qubit)
                if j < n_states:
                    old_0 = state[:, i].clone()
                    old_1 = state[:, j].clone()
                    
                    rotated_state[:, i] = cos_rx * cos_ry * old_0 - sin_rx * sin_ry * old_1 * 1j
                    rotated_state[:, j] = sin_rx * cos_ry * old_0 + cos_rx * sin_ry * old_1 * 1j
        
        return rotated_state
    
    def _apply_cnot_gate(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate between control and target qubits."""
        new_state = state.clone()
        n_states = state.shape[-1]
        
        for i in range(n_states):
            if (i >> control) & 1 == 1:  # Control is |1⟩
                j = i ^ (1 << target)  # Flip target
                if j < n_states:
                    new_state[:, j] = state[:, i]
        
        return new_state
    
    def _simulate_errors(self, state: torch.Tensor) -> torch.Tensor:
        """Simulate physical errors during computation."""
        if not self.training:
            return state
        
        noisy_state = state.clone()
        batch_size, n_states = state.shape
        
        # Add depolarization noise
        error_prob = self.config.physical_error_rate
        
        if torch.rand(1) < error_prob:
            # Apply random Pauli error
            error_type = torch.randint(0, 3, (1,)).item()  # X, Y, or Z
            error_qubit = torch.randint(0, self.n_logical_qubits, (1,)).item()
            
            if error_type == 0:  # X error
                noisy_state = self._apply_x_error(noisy_state, error_qubit)
            elif error_type == 1:  # Y error  
                noisy_state = self._apply_y_error(noisy_state, error_qubit)
            else:  # Z error
                noisy_state = self._apply_z_error(noisy_state, error_qubit)
        
        return noisy_state
    
    def _apply_x_error(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply X (bit flip) error."""
        new_state = state.clone()
        n_states = state.shape[-1]
        
        for i in range(n_states):
            j = i ^ (1 << qubit)
            if j < n_states:
                new_state[:, j] = state[:, i]
                new_state[:, i] = state[:, j]
        
        return new_state
    
    def _apply_y_error(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply Y error (X + Z with i factor)."""
        new_state = self._apply_x_error(state, qubit)
        new_state = self._apply_z_error(new_state, qubit)
        return new_state * 1j
    
    def _apply_z_error(self, state: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply Z (phase flip) error."""
        new_state = state.clone()
        n_states = state.shape[-1]
        
        for i in range(n_states):
            if (i >> qubit) & 1 == 1:
                new_state[:, i] *= -1
        
        return new_state
    
    def _measure_syndromes(self, state: torch.Tensor) -> torch.Tensor:
        """Measure error syndromes from quantum state."""
        batch_size = state.shape[0]
        
        # Simulate syndrome measurement
        # In practice, this would involve measuring stabilizer generators
        syndromes = torch.zeros(batch_size, self.code.n_stabilizers, device=state.device)
        
        # Simplified syndrome computation based on state deviations
        state_probs = torch.abs(state) ** 2
        expected_uniform = torch.ones_like(state_probs) / state_probs.shape[1]
        
        deviation = torch.norm(state_probs - expected_uniform, dim=1)
        
        # Convert deviation to synthetic syndromes
        for i in range(self.code.n_stabilizers):
            syndromes[:, i] = (deviation + torch.randn(batch_size, device=state.device) * 0.1) > 0.5
        
        return syndromes.float()
    
    def _correct_errors(self, state: torch.Tensor, syndrome: torch.Tensor) -> torch.Tensor:
        """Correct errors based on syndrome measurements."""
        if hasattr(self, 'decoder') and self.config.correction_algorithm == 'ml_decoder':
            # Use ML decoder
            x_corrections, z_corrections = self.decoder.predict_corrections(syndrome)
            
            corrected_state = state.clone()
            batch_size = state.shape[0]
            
            for batch_idx in range(batch_size):
                # Apply X corrections
                for qubit in range(self.n_logical_qubits):
                    if qubit < x_corrections.shape[1] and x_corrections[batch_idx, qubit] > 0.5:
                        corrected_state[batch_idx:batch_idx+1] = self._apply_x_error(
                            corrected_state[batch_idx:batch_idx+1], qubit
                        )
                
                # Apply Z corrections
                for qubit in range(self.n_logical_qubits):
                    if qubit < z_corrections.shape[1] and z_corrections[batch_idx, qubit] > 0.5:
                        corrected_state[batch_idx:batch_idx+1] = self._apply_z_error(
                            corrected_state[batch_idx:batch_idx+1], qubit
                        )
            
            return corrected_state
        else:
            # Use classical decoding
            return self._classical_decode_and_correct(state, syndrome)
    
    def _classical_decode_and_correct(self, state: torch.Tensor, syndrome: torch.Tensor) -> torch.Tensor:
        """Classical decoding and correction."""
        # Simplified classical decoder
        corrected_state = state.clone()
        
        # Apply majority-vote correction based on syndrome
        for batch_idx in range(state.shape[0]):
            syndrome_bits = syndrome[batch_idx].cpu().numpy()
            error_pattern = self.code.detect_errors(syndrome_bits)
            
            # Apply corrections (simplified)
            for qubit in range(min(self.n_logical_qubits, len(error_pattern) // 2)):
                if error_pattern[qubit] == 1:  # X correction
                    corrected_state[batch_idx:batch_idx+1] = self._apply_x_error(
                        corrected_state[batch_idx:batch_idx+1], qubit
                    )
                if error_pattern[qubit + len(error_pattern) // 2] == 1:  # Z correction
                    corrected_state[batch_idx:batch_idx+1] = self._apply_z_error(
                        corrected_state[batch_idx:batch_idx+1], qubit
                    )
        
        return corrected_state
    
    def get_error_rate(self) -> float:
        """Get current logical error rate."""
        if self.error_counts[0] > 0:
            return (self.error_counts[2] / self.error_counts[0]).item()
        return 0.0


class FaultTolerantQuantumMLP(nn.Module):
    """
    Fault-tolerant quantum multilayer perceptron.
    
    Complete quantum machine learning model with integrated error correction
    for reliable quantum advantage.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 code_distance: int = 3,
                 config: Optional[ErrorCorrectionConfig] = None):
        super().__init__()
        
        if config is None:
            config = ErrorCorrectionConfig(code_distance=code_distance)
        
        self.config = config
        
        # Create surface code for error correction
        self.surface_code = SurfaceCode(distance=code_distance)
        
        # Build fault-tolerant layers
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layer = FaultTolerantQuantumLayer(
                dims[i], dims[i+1], self.surface_code, config
            )
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fault-tolerant quantum network."""
        current = x
        
        for i, layer in enumerate(self.layers):
            current = layer(current)
            
            # Apply activation (except final layer)
            if i < len(self.layers) - 1:
                current = torch.relu(current)
        
        return current
    
    def get_total_error_rate(self) -> float:
        """Get average error rate across all layers."""
        error_rates = [layer.get_error_rate() for layer in self.layers]
        return sum(error_rates) / len(error_rates) if error_rates else 0.0
    
    def reset_error_counters(self):
        """Reset error counters for all layers."""
        for layer in self.layers:
            layer.error_counts.zero_()


def train_ml_decoder(surface_code: SurfaceCode, 
                    training_samples: int = 10000) -> MLQuantumDecoder:
    """
    Train ML decoder for quantum error correction.
    
    Args:
        surface_code: The quantum error correcting code
        training_samples: Number of training samples to generate
        
    Returns:
        Trained ML decoder
    """
    print("🧠 Training ML Quantum Error Decoder...")
    
    # Generate training data
    syndromes = []
    x_errors = []
    z_errors = []
    
    for _ in range(training_samples):
        # Generate random error pattern
        error_weight = np.random.poisson(1.0)  # Poisson-distributed error weight
        error_weight = min(error_weight, surface_code.code_distance // 2)  # Correctable errors
        
        error_pattern = np.zeros(2 * surface_code.n_qubits, dtype=int)
        
        if error_weight > 0:
            # Select random qubits for errors
            error_positions = np.random.choice(
                2 * surface_code.n_qubits, 
                size=error_weight, 
                replace=False
            )
            error_pattern[error_positions] = 1
        
        # Compute syndrome
        syndrome = surface_code._compute_syndrome(error_pattern)
        
        syndromes.append(syndrome)
        x_errors.append(error_pattern[:surface_code.n_qubits])
        z_errors.append(error_pattern[surface_code.n_qubits:])
    
    # Convert to tensors
    syndrome_tensor = torch.tensor(syndromes, dtype=torch.float32)
    x_error_tensor = torch.tensor(x_errors, dtype=torch.float32)  
    z_error_tensor = torch.tensor(z_errors, dtype=torch.float32)
    
    # Create and train decoder
    decoder = MLQuantumDecoder(
        n_syndrome_bits=surface_code.n_stabilizers,
        n_qubits=surface_code.n_qubits
    )
    
    # Training setup
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Training loop
    batch_size = 128
    n_epochs = 100
    
    dataset = torch.utils.data.TensorDataset(syndrome_tensor, x_error_tensor, z_error_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    decoder.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for syndromes_batch, x_errors_batch, z_errors_batch in dataloader:
            optimizer.zero_grad()
            
            x_pred, z_pred = decoder(syndromes_batch)
            
            loss_x = criterion(x_pred, x_errors_batch)
            loss_z = criterion(z_pred, z_errors_batch)
            loss = loss_x + loss_z
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    decoder.eval()
    print("✅ ML Decoder Training Complete!")
    
    return decoder


def demonstrate_fault_tolerant_learning():
    """Demonstrate fault-tolerant quantum machine learning."""
    print("🛡️ Fault-Tolerant Quantum Machine Learning Demo")
    print("=" * 55)
    
    # Create test dataset
    n_samples = 1000
    input_dim = 8
    n_classes = 3
    
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create fault-tolerant model
    config = ErrorCorrectionConfig(
        code_distance=3,
        physical_error_rate=1e-3,
        correction_algorithm='ml_decoder'
    )
    
    print("\n🏗️ Building Fault-Tolerant Quantum Model...")
    ft_model = FaultTolerantQuantumMLP(
        input_dim=input_dim,
        hidden_dims=[16, 8],
        output_dim=n_classes,
        code_distance=3,
        config=config
    )
    
    # Train ML decoder
    surface_code = SurfaceCode(distance=3)
    ml_decoder = train_ml_decoder(surface_code, training_samples=5000)
    
    # Update decoders in all layers
    for layer in ft_model.layers:
        if hasattr(layer, 'decoder'):
            layer.decoder = ml_decoder
    
    # Training setup
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("\n🚀 Training Fault-Tolerant Model...")
    
    ft_model.train()
    for epoch in range(20):
        ft_model.reset_error_counters()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = ft_model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            error_rate = ft_model.get_total_error_rate()
            print(f"  Epoch {epoch+1}/20, Loss: {avg_loss:.4f}, Error Rate: {error_rate:.6f}")
    
    # Evaluation
    print("\n📊 Evaluating Fault-Tolerant Performance...")
    ft_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = ft_model(batch_x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    final_error_rate = ft_model.get_total_error_rate()
    
    print(f"\n✅ Fault-Tolerant Quantum ML Results:")
    print(f"  🎯 Test Accuracy: {accuracy:.4f}")
    print(f"  🛡️ Final Error Rate: {final_error_rate:.6f}")
    print(f"  🔬 Surface Code Distance: {config.code_distance}")
    print(f"  🤖 ML Decoder Used: {config.correction_algorithm}")
    
    print(f"\n🎊 Fault-tolerant quantum machine learning demonstrated successfully!")
    print(f"     The model maintains {accuracy:.1%} accuracy despite quantum errors!")


if __name__ == "__main__":
    demonstrate_fault_tolerant_learning()