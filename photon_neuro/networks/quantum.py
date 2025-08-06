"""
Quantum photonic neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from ..core import PhotonicComponent
from ..core.registry import register_component


@register_component
class QuantumPhotonic(PhotonicComponent):
    """Interface between classical and quantum photonic circuits."""
    
    def __init__(self, n_qubits: int, n_modes: int, encoding: str = "dual_rail"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_modes = n_modes
        self.encoding = encoding
        
        # Quantum state representation
        self.n_basis_states = 2**n_qubits
        self.register_buffer('quantum_state', 
                           torch.zeros(self.n_basis_states, dtype=torch.complex64))
        
        # Photonic mode operators
        self.creation_ops = self._initialize_creation_operators()
        self.annihilation_ops = self._initialize_annihilation_operators()
        
        # Gate implementations
        self.gate_library = self._build_gate_library()
        
    def _initialize_creation_operators(self) -> List[torch.Tensor]:
        """Initialize photonic creation operators."""
        operators = []
        max_photons = 5  # Truncated Fock space
        
        for mode in range(self.n_modes):
            # Creation operator for each mode
            op = torch.zeros((max_photons + 1, max_photons + 1), dtype=torch.complex64)
            for n in range(max_photons):
                op[n + 1, n] = np.sqrt(n + 1)
            operators.append(op)
            
        return operators
        
    def _initialize_annihilation_operators(self) -> List[torch.Tensor]:
        """Initialize photonic annihilation operators."""  
        operators = []
        max_photons = 5
        
        for mode in range(self.n_modes):
            # Annihilation operator for each mode
            op = torch.zeros((max_photons + 1, max_photons + 1), dtype=torch.complex64)
            for n in range(1, max_photons + 1):
                op[n - 1, n] = np.sqrt(n)
            operators.append(op)
            
        return operators
        
    def _build_gate_library(self) -> Dict[str, torch.Tensor]:
        """Build library of quantum gates."""
        gates = {}
        
        # Single-qubit gates
        gates['I'] = torch.eye(2, dtype=torch.complex64)
        gates['X'] = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        gates['Y'] = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        gates['Z'] = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        gates['H'] = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
        
        # Phase gates
        gates['S'] = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
        gates['T'] = torch.tensor([[1, 0], [0, np.exp(1j * np.pi/4)]], dtype=torch.complex64)
        
        # Two-qubit gates
        gates['CNOT'] = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)
        
        gates['CZ'] = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0], 
            [0, 0, 0, -1]
        ], dtype=torch.complex64)
        
        return gates
        
    def prepare_state(self, basis_state: str):
        """Prepare quantum state in computational basis."""
        if len(basis_state) != self.n_qubits:
            raise ValueError(f"Basis state must have {self.n_qubits} qubits")
            
        # Convert binary string to state index
        state_index = int(basis_state, 2)
        
        # Initialize to |0...0⟩ then flip to target state
        self.quantum_state.zero_()
        self.quantum_state[state_index] = 1.0
        
    def apply_gate(self, gate_name: str, qubits: List[int]):
        """Apply quantum gate to specified qubits."""
        if gate_name not in self.gate_library:
            raise ValueError(f"Gate {gate_name} not in library")
            
        gate = self.gate_library[gate_name]
        
        if len(qubits) == 1:
            # Single-qubit gate
            self._apply_single_qubit_gate(gate, qubits[0])
        elif len(qubits) == 2:
            # Two-qubit gate  
            self._apply_two_qubit_gate(gate, qubits[0], qubits[1])
        else:
            raise NotImplementedError("Only 1 and 2 qubit gates supported")
            
    def _apply_single_qubit_gate(self, gate: torch.Tensor, qubit: int):
        """Apply single-qubit gate."""
        new_state = torch.zeros_like(self.quantum_state)
        
        for i in range(self.n_basis_states):
            # Extract bit for target qubit
            qubit_bit = (i >> qubit) & 1
            
            for j in range(2):
                # Calculate new amplitude
                amplitude = gate[j, qubit_bit]
                
                # Flip bit if needed
                new_i = i
                if j != qubit_bit:
                    new_i = i ^ (1 << qubit)
                    
                new_state[new_i] += amplitude * self.quantum_state[i]
                
        self.quantum_state = new_state
        
    def _apply_two_qubit_gate(self, gate: torch.Tensor, control: int, target: int):
        """Apply two-qubit gate."""
        new_state = torch.zeros_like(self.quantum_state)
        
        for i in range(self.n_basis_states):
            # Extract bits for control and target qubits
            control_bit = (i >> control) & 1
            target_bit = (i >> target) & 1
            two_qubit_state = control_bit * 2 + target_bit
            
            for j in range(4):
                # Calculate new amplitude
                amplitude = gate[j, two_qubit_state]
                
                if amplitude != 0:
                    # Calculate new state index
                    new_control_bit = j >> 1
                    new_target_bit = j & 1
                    
                    new_i = i
                    if new_control_bit != control_bit:
                        new_i = new_i ^ (1 << control)
                    if new_target_bit != target_bit:
                        new_i = new_i ^ (1 << target)
                        
                    new_state[new_i] += amplitude * self.quantum_state[i]
                    
        self.quantum_state = new_state
        
    def measure(self, shots: int = 1000) -> Dict[str, int]:
        """Measure quantum state in computational basis."""
        # Calculate measurement probabilities
        probabilities = torch.abs(self.quantum_state)**2
        
        # Sample according to probabilities
        counts = {}
        for _ in range(shots):
            # Sample state
            state_idx = torch.multinomial(probabilities, 1).item()
            
            # Convert to binary string
            binary_string = format(state_idx, f'0{self.n_qubits}b')
            
            counts[binary_string] = counts.get(binary_string, 0) + 1
            
        return counts
        
    def forward(self, classical_input: torch.Tensor) -> torch.Tensor:
        """Interface classical input to quantum circuit."""
        batch_size = classical_input.shape[0]
        
        # Encode classical data into quantum states
        quantum_outputs = []
        
        for b in range(batch_size):
            # Reset quantum state
            self.prepare_state('0' * self.n_qubits)
            
            # Encode classical input (amplitude encoding)
            input_data = classical_input[b]
            
            # Normalize input
            norm = torch.sqrt(torch.sum(input_data**2))
            if norm > 0:
                normalized_input = input_data / norm
                
                # Encode into quantum amplitudes (simplified)
                for i, amplitude in enumerate(normalized_input):
                    if i < self.n_basis_states:
                        self.quantum_state[i] = amplitude
                        
            # Apply quantum processing (example circuit)
            self._example_quantum_circuit()
            
            # Measure expectation values
            quantum_output = self._calculate_expectation_values()
            quantum_outputs.append(quantum_output)
            
        return torch.stack(quantum_outputs)
        
    def _example_quantum_circuit(self):
        """Example quantum circuit for processing."""
        # Create superposition
        for q in range(self.n_qubits):
            self.apply_gate('H', [q])
            
        # Add entanglement
        for q in range(self.n_qubits - 1):
            self.apply_gate('CNOT', [q, q + 1])
            
        # Rotation gates (parametrized - would be trainable)
        for q in range(self.n_qubits):
            # Phase rotation (simplified)
            theta = 0.1  # Would be learnable parameter
            phase_gate = torch.tensor([
                [1, 0],
                [0, np.exp(1j * theta)]
            ], dtype=torch.complex64)
            
            self._apply_single_qubit_gate(phase_gate, q)
            
    def _calculate_expectation_values(self) -> torch.Tensor:
        """Calculate expectation values for measurement."""
        expectations = []
        
        # Pauli-Z expectation for each qubit
        for q in range(self.n_qubits):
            expectation = 0.0
            for i in range(self.n_basis_states):
                # Check if qubit q is |0⟩ or |1⟩
                qubit_bit = (i >> q) & 1
                sign = 1 if qubit_bit == 0 else -1
                
                probability = torch.abs(self.quantum_state[i])**2
                expectation += sign * probability
                
            expectations.append(expectation)
            
        return torch.tensor(expectations, dtype=torch.float32)
        
    def get_fidelity(self, target_state: torch.Tensor) -> float:
        """Calculate fidelity with target quantum state."""
        overlap = torch.sum(torch.conj(target_state) * self.quantum_state)
        fidelity = torch.abs(overlap)**2
        return fidelity.item()
        
    def to_netlist(self) -> dict:
        return {
            "type": "quantum_photonic",
            "n_qubits": self.n_qubits,
            "n_modes": self.n_modes,
            "encoding": self.encoding,
            "quantum_state": self.quantum_state.detach().numpy().tolist()
        }