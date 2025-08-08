#!/usr/bin/env python3
"""
Quantum-Photonic Interface Example
==================================

Demonstrates interfacing between quantum photonic circuits and classical
photonic neural networks for hybrid quantum-classical computing.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

import photon_neuro as pn


class QuantumPhotonicGate:
    """Base class for quantum photonic gates."""
    
    def __init__(self, n_modes: int, gate_type: str):
        self.n_modes = n_modes
        self.gate_type = gate_type
        self.unitary_matrix = torch.eye(n_modes, dtype=torch.complex128)
        self.fidelity = 1.0
        
    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """Apply quantum gate to photonic state."""
        return torch.matmul(self.unitary_matrix, state)


class BeamSplitter(QuantumPhotonicGate):
    """50:50 beam splitter for dual-rail encoding."""
    
    def __init__(self, mode1: int, mode2: int, reflectivity: float = 0.5):
        super().__init__(2, "beam_splitter")
        self.mode1 = mode1
        self.mode2 = mode2
        self.reflectivity = reflectivity
        
        # Beam splitter transformation matrix
        r = np.sqrt(reflectivity)
        t = np.sqrt(1 - reflectivity)
        
        self.unitary_matrix = torch.tensor([
            [t, 1j * r],
            [1j * r, t]
        ], dtype=torch.complex128)


class PhaseShifterQuantum(QuantumPhotonicGate):
    """Quantum phase shifter for single-mode phase modulation."""
    
    def __init__(self, phase: float):
        super().__init__(1, "phase_shifter")
        self.phase = phase
        
        self.unitary_matrix = torch.tensor(
            [[torch.exp(1j * phase)]], dtype=torch.complex128
        )
    
    def set_phase(self, phase: float):
        """Update phase shifter setting."""
        self.phase = phase
        self.unitary_matrix = torch.tensor(
            [[torch.exp(1j * phase)]], dtype=torch.complex128
        )


class QuantumPhotonicProcessor:
    """Quantum photonic circuit processor with dual-rail encoding."""
    
    def __init__(self, n_qubits: int, n_modes: Optional[int] = None):
        self.n_qubits = n_qubits
        self.n_modes = n_modes or 2 * n_qubits  # Dual-rail encoding
        self.encoding = "dual_rail"
        
        # Quantum state (photonic Fock states)
        self.state = torch.zeros(2**self.n_modes, dtype=torch.complex128)
        self.state[0] = 1.0  # Vacuum state initially
        
        # Gate sequence
        self.gate_sequence = []
        
        # Measurement outcomes
        self.measurement_history = []
        
        # Error tracking
        self.decoherence_rate = 0.001
        self.gate_error_rate = 0.01
    
    def prepare_state(self, classical_bits: str) -> None:
        """Prepare quantum state from classical bit string."""
        if len(classical_bits) != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} bits, got {len(classical_bits)}")
        
        # Initialize in computational basis
        self.state = torch.zeros(2**self.n_modes, dtype=torch.complex128)
        
        # Dual-rail encoding: |0⟩ → |10⟩, |1⟩ → |01⟩
        state_index = 0
        for i, bit in enumerate(classical_bits):
            if bit == '0':
                # Photon in first mode of qubit i
                state_index += (1 << (2*i))
            else:
                # Photon in second mode of qubit i  
                state_index += (1 << (2*i + 1))
        
        self.state[state_index] = 1.0
    
    def add_hadamard_gate(self, qubit: int) -> None:
        """Add Hadamard gate implemented with beam splitters."""
        # Hadamard in dual-rail: symmetric beam splitter
        bs = BeamSplitter(2*qubit, 2*qubit + 1, reflectivity=0.5)
        self.gate_sequence.append(("hadamard", qubit, bs))
    
    def add_cnot_gate(self, control: int, target: int) -> None:
        """Add CNOT gate implemented with conditional phase shifts."""
        # CNOT in dual-rail requires complex interferometric setup
        # Simplified model using conditional phase
        self.gate_sequence.append(("cnot", (control, target), None))
    
    def add_phase_gate(self, qubit: int, phase: float) -> None:
        """Add phase gate to specific qubit."""
        ps = PhaseShifterQuantum(phase)
        self.gate_sequence.append(("phase", qubit, ps))
    
    def add_rotation_gate(self, qubit: int, theta: float, phi: float) -> None:
        """Add arbitrary single-qubit rotation."""
        self.gate_sequence.append(("rotation", qubit, (theta, phi)))
    
    def apply_gates(self) -> None:
        """Apply all gates in the sequence to the quantum state."""
        for gate_type, params, gate_obj in self.gate_sequence:
            
            if gate_type == "hadamard":
                qubit = params
                self._apply_hadamard(qubit)
                
            elif gate_type == "cnot":
                control, target = params
                self._apply_cnot(control, target)
                
            elif gate_type == "phase":
                qubit = params
                phase = gate_obj.phase
                self._apply_phase(qubit, phase)
                
            elif gate_type == "rotation":
                qubit = params
                theta, phi = gate_obj
                self._apply_rotation(qubit, theta, phi)
            
            # Add gate error
            self._add_gate_error()
    
    def _apply_hadamard(self, qubit: int) -> None:
        """Apply Hadamard gate to qubit using beam splitter."""
        # Create superposition between dual-rail modes
        new_state = torch.zeros_like(self.state)
        
        for i in range(len(self.state)):
            if abs(self.state[i]) > 1e-12:
                # Check photon occupation in dual-rail modes
                mode0_occupied = (i >> (2*qubit)) & 1
                mode1_occupied = (i >> (2*qubit + 1)) & 1
                
                if mode0_occupied and not mode1_occupied:  # |10⟩
                    # |10⟩ → (|10⟩ + |01⟩)/√2
                    new_state[i] += self.state[i] / np.sqrt(2)  # |10⟩ component
                    flip_state = i ^ (3 << (2*qubit))  # Flip to |01⟩
                    new_state[flip_state] += self.state[i] / np.sqrt(2)
                    
                elif mode1_occupied and not mode0_occupied:  # |01⟩
                    # |01⟩ → (|10⟩ - |01⟩)/√2
                    flip_state = i ^ (3 << (2*qubit))  # Flip to |10⟩
                    new_state[flip_state] += self.state[i] / np.sqrt(2)
                    new_state[i] -= self.state[i] / np.sqrt(2)  # |01⟩ component
        
        self.state = new_state
    
    def _apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT gate using conditional beam splitter."""
        new_state = torch.zeros_like(self.state)
        
        for i in range(len(self.state)):
            if abs(self.state[i]) > 1e-12:
                # Check control qubit state
                control_mode0 = (i >> (2*control)) & 1
                control_mode1 = (i >> (2*control + 1)) & 1
                
                # If control is in |1⟩ state (mode1 occupied), flip target
                if control_mode1 and not control_mode0:
                    # Flip target qubit
                    target_flip = i ^ (3 << (2*target))
                    new_state[target_flip] += self.state[i]
                else:
                    # No flip
                    new_state[i] += self.state[i]
        
        self.state = new_state
    
    def _apply_phase(self, qubit: int, phase: float) -> None:
        """Apply phase gate to qubit."""
        for i in range(len(self.state)):
            # Apply phase only to |1⟩ component (mode1 occupied)
            mode1_occupied = (i >> (2*qubit + 1)) & 1
            mode0_occupied = (i >> (2*qubit)) & 1
            
            if mode1_occupied and not mode0_occupied:
                self.state[i] *= torch.exp(1j * phase)
    
    def _apply_rotation(self, qubit: int, theta: float, phi: float) -> None:
        """Apply arbitrary single-qubit rotation."""
        # Decompose rotation into basis operations
        # R(θ,φ) = RZ(φ) RY(θ) 
        self._apply_phase(qubit, phi)
        
        # Y-rotation requires more complex beam splitter network
        # Simplified implementation
        cos_half = torch.cos(theta / 2)
        sin_half = torch.sin(theta / 2)
        
        new_state = torch.zeros_like(self.state)
        
        for i in range(len(self.state)):
            if abs(self.state[i]) > 1e-12:
                mode0_occupied = (i >> (2*qubit)) & 1
                mode1_occupied = (i >> (2*qubit + 1)) & 1
                
                if mode0_occupied and not mode1_occupied:  # |0⟩
                    new_state[i] += cos_half * self.state[i]
                    flip_state = i ^ (3 << (2*qubit))
                    new_state[flip_state] += sin_half * self.state[i]
                    
                elif mode1_occupied and not mode0_occupied:  # |1⟩
                    flip_state = i ^ (3 << (2*qubit))
                    new_state[flip_state] -= sin_half * self.state[i]
                    new_state[i] += cos_half * self.state[i]
        
        self.state = new_state
    
    def _add_gate_error(self) -> None:
        """Add realistic gate error (decoherence)."""
        if np.random.random() < self.gate_error_rate:
            # Add small random phase error
            phase_error = np.random.normal(0, 0.1)
            self.state *= torch.exp(1j * phase_error)
        
        # Normalize state
        norm = torch.norm(self.state)
        if norm > 1e-12:
            self.state /= norm
    
    def measure_qubit(self, qubit: int) -> int:
        """Measure single qubit in computational basis."""
        # Calculate probabilities for |0⟩ and |1⟩
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(len(self.state)):
            prob_i = abs(self.state[i]) ** 2
            
            mode0_occupied = (i >> (2*qubit)) & 1
            mode1_occupied = (i >> (2*qubit + 1)) & 1
            
            if mode0_occupied and not mode1_occupied:
                prob_0 += prob_i
            elif mode1_occupied and not mode0_occupied:
                prob_1 += prob_i
        
        # Quantum measurement
        measurement = 0 if np.random.random() < prob_0 / (prob_0 + prob_1) else 1
        
        # State collapse
        self._collapse_state(qubit, measurement)
        
        self.measurement_history.append((qubit, measurement))
        return measurement
    
    def measure_all(self) -> List[int]:
        """Measure all qubits."""
        results = []
        for qubit in range(self.n_qubits):
            result = self.measure_qubit(qubit)
            results.append(result)
        return results
    
    def _collapse_state(self, qubit: int, measurement: int) -> None:
        """Collapse state after measurement."""
        new_state = torch.zeros_like(self.state)
        norm = 0.0
        
        for i in range(len(self.state)):
            mode0_occupied = (i >> (2*qubit)) & 1
            mode1_occupied = (i >> (2*qubit + 1)) & 1
            
            consistent = False
            if measurement == 0 and mode0_occupied and not mode1_occupied:
                consistent = True
            elif measurement == 1 and mode1_occupied and not mode0_occupied:
                consistent = True
            
            if consistent:
                new_state[i] = self.state[i]
                norm += abs(self.state[i]) ** 2
        
        # Renormalize
        if norm > 1e-12:
            new_state /= torch.sqrt(norm)
        
        self.state = new_state
    
    def get_state_fidelity(self, target_state: torch.Tensor) -> float:
        """Calculate fidelity with target state."""
        overlap = torch.abs(torch.dot(torch.conj(target_state), self.state)) ** 2
        return float(overlap)
    
    def reset(self) -> None:
        """Reset to vacuum state."""
        self.state = torch.zeros(2**self.n_modes, dtype=torch.complex128)
        self.state[0] = 1.0
        self.gate_sequence = []
        self.measurement_history = []


class HybridQuantumClassicalNetwork(nn.Module):
    """Hybrid network combining quantum photonic processing with classical layers."""
    
    def __init__(self, n_qubits: int = 4, classical_dim: int = 256):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.classical_dim = classical_dim
        
        # Quantum photonic processor
        self.quantum_processor = QuantumPhotonicProcessor(n_qubits)
        
        # Classical preprocessing
        self.classical_encoder = nn.Sequential(
            nn.Linear(classical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits)
        )
        
        # Quantum circuit parameters (trainable)
        self.quantum_params = nn.Parameter(torch.randn(n_qubits * 3))  # 3 params per qubit
        
        # Classical postprocessing
        self.classical_decoder = nn.Sequential(
            nn.Linear(2**n_qubits, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 10-class output
        )
        
        # Photonic-classical interface
        self.photonic_interface = pn.PhotonicQuantumInterface(
            n_qubits=n_qubits,
            encoding="dual_rail",
            measurement_type="computational_basis"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid quantum-classical network."""
        batch_size = x.shape[0]
        
        # Classical encoding
        encoded = torch.tanh(self.classical_encoder(x))  # [-1, 1] range
        
        # Quantum processing
        quantum_outputs = []
        
        for i in range(batch_size):
            # Prepare quantum state from classical data
            classical_bits = self._encode_to_bits(encoded[i])
            self.quantum_processor.prepare_state(classical_bits)
            
            # Apply parameterized quantum circuit
            self._apply_parameterized_circuit(self.quantum_params)
            
            # Measure quantum state
            measurement_probs = self._get_measurement_probabilities()
            quantum_outputs.append(measurement_probs)
            
            # Reset for next sample
            self.quantum_processor.reset()
        
        quantum_tensor = torch.stack(quantum_outputs)
        
        # Classical postprocessing
        output = self.classical_decoder(quantum_tensor)
        
        return output
    
    def _encode_to_bits(self, values: torch.Tensor) -> str:
        """Encode continuous values to bit string."""
        bits = []
        for val in values:
            bit = '1' if val.item() > 0 else '0'
            bits.append(bit)
        return ''.join(bits)
    
    def _apply_parameterized_circuit(self, params: torch.Tensor) -> None:
        """Apply parameterized quantum circuit."""
        param_idx = 0
        
        # Layer 1: Hadamard gates
        for qubit in range(self.n_qubits):
            self.quantum_processor.add_hadamard_gate(qubit)
        
        # Layer 2: Parameterized rotations
        for qubit in range(self.n_qubits):
            theta = params[param_idx].item()
            phi = params[param_idx + 1].item()
            self.quantum_processor.add_rotation_gate(qubit, theta, phi)
            param_idx += 2
        
        # Layer 3: Entangling gates
        for qubit in range(self.n_qubits - 1):
            self.quantum_processor.add_cnot_gate(qubit, qubit + 1)
        
        # Layer 4: Final rotations
        for qubit in range(self.n_qubits):
            if param_idx < len(params):
                phase = params[param_idx].item()
                self.quantum_processor.add_phase_gate(qubit, phase)
                param_idx += 1
        
        # Apply all gates
        self.quantum_processor.apply_gates()
    
    def _get_measurement_probabilities(self) -> torch.Tensor:
        """Get measurement probability distribution."""
        probabilities = torch.zeros(2**self.n_qubits)
        
        # Calculate probability for each computational basis state
        for i in range(2**self.n_qubits):
            # Convert index to bit string
            bit_string = format(i, f'0{self.n_qubits}b')
            
            # Calculate probability amplitude
            prob = 0.0
            for j, amplitude in enumerate(self.quantum_processor.state):
                if self._matches_computational_basis(j, bit_string):
                    prob += abs(amplitude) ** 2
            
            probabilities[i] = prob
        
        return probabilities
    
    def _matches_computational_basis(self, state_index: int, bit_string: str) -> bool:
        """Check if Fock state corresponds to computational basis state."""
        for qubit, bit in enumerate(bit_string):
            mode0_occupied = (state_index >> (2*qubit)) & 1
            mode1_occupied = (state_index >> (2*qubit + 1)) & 1
            
            if bit == '0' and not (mode0_occupied and not mode1_occupied):
                return False
            elif bit == '1' and not (mode1_occupied and not mode0_occupied):
                return False
        
        return True


def demonstrate_quantum_algorithms():
    """Demonstrate quantum algorithms on photonic processor."""
    print("🔬 Quantum Photonic Algorithm Demonstrations")
    print("=" * 50)
    
    # 1. Quantum Fourier Transform
    print("\n1. Quantum Fourier Transform (3 qubits)")
    qft_processor = QuantumPhotonicProcessor(n_qubits=3)
    qft_processor.prepare_state("101")  # Initial state |101⟩
    
    # QFT circuit
    for qubit in range(3):
        qft_processor.add_hadamard_gate(qubit)
        for control in range(qubit):
            angle = np.pi / (2**(qubit - control))
            qft_processor.add_cnot_gate(control, qubit)
            qft_processor.add_phase_gate(qubit, angle)
    
    # Apply QFT
    initial_state = qft_processor.state.clone()
    qft_processor.apply_gates()
    final_state = qft_processor.state
    
    print(f"   Initial state amplitudes: {torch.abs(initial_state[:8])}")
    print(f"   QFT output amplitudes: {torch.abs(final_state[:8])}")
    
    # 2. Grover's Algorithm (2 qubits)
    print("\n2. Grover's Algorithm (2 qubits, marked state |11⟩)")
    grover_processor = QuantumPhotonicProcessor(n_qubits=2)
    grover_processor.prepare_state("00")
    
    # Initialize superposition
    grover_processor.add_hadamard_gate(0)
    grover_processor.add_hadamard_gate(1)
    grover_processor.apply_gates()
    
    # Grover iterations (1 iteration optimal for 2 qubits)
    n_iterations = 1
    for _ in range(n_iterations):
        # Oracle: flip phase of |11⟩
        grover_processor.add_phase_gate(0, np.pi)
        grover_processor.add_phase_gate(1, np.pi)
        grover_processor.add_cnot_gate(0, 1)
        grover_processor.add_phase_gate(1, np.pi)
        grover_processor.add_cnot_gate(0, 1)
        
        # Diffusion operator
        grover_processor.add_hadamard_gate(0)
        grover_processor.add_hadamard_gate(1)
        grover_processor.add_phase_gate(0, np.pi)
        grover_processor.add_phase_gate(1, np.pi)
        grover_processor.add_cnot_gate(0, 1)
        grover_processor.add_phase_gate(1, np.pi)
        grover_processor.add_cnot_gate(0, 1)
        grover_processor.add_phase_gate(0, np.pi)
        grover_processor.add_phase_gate(1, np.pi)
        grover_processor.add_hadamard_gate(0)
        grover_processor.add_hadamard_gate(1)
        
        grover_processor.apply_gates()
    
    # Measure results multiple times
    measurement_counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    for _ in range(1000):
        grover_processor_copy = QuantumPhotonicProcessor(n_qubits=2)
        grover_processor_copy.state = grover_processor.state.clone()
        results = grover_processor_copy.measure_all()
        result_string = ''.join(map(str, results))
        measurement_counts[result_string] += 1
    
    print(f"   Measurement statistics (1000 runs): {measurement_counts}")
    expected_prob = measurement_counts['11'] / 1000
    print(f"   Probability of finding marked state |11⟩: {expected_prob:.3f}")
    
    # 3. Variational Quantum Eigensolver (VQE) demonstration
    print("\n3. Variational Quantum Eigensolver (VQE) - H₂ molecule")
    
    def pauli_expectation(processor: QuantumPhotonicProcessor, 
                         pauli_string: str) -> float:
        """Calculate expectation value of Pauli operator."""
        # Simplified VQE for demonstration
        # In practice, would implement full Pauli measurement
        measurements = []
        for _ in range(100):
            proc_copy = QuantumPhotonicProcessor(processor.n_qubits)
            proc_copy.state = processor.state.clone()
            result = proc_copy.measure_all()
            measurements.append(result)
        
        # Mock expectation value calculation
        return np.mean([(-1)**sum(result) for result in measurements])
    
    def h2_hamiltonian_energy(params: List[float]) -> float:
        """H₂ Hamiltonian energy calculation."""
        vqe_processor = QuantumPhotonicProcessor(n_qubits=2)
        vqe_processor.prepare_state("00")
        
        # Ansatz circuit
        vqe_processor.add_rotation_gate(0, params[0], 0)
        vqe_processor.add_rotation_gate(1, params[1], 0)
        vqe_processor.add_cnot_gate(0, 1)
        vqe_processor.add_rotation_gate(1, params[2], 0)
        
        vqe_processor.apply_gates()
        
        # H₂ Hamiltonian terms (simplified)
        h_coeff = [-1.052, 0.396, -0.396, -0.012]
        pauli_ops = ['II', 'ZI', 'IZ', 'ZZ']
        
        energy = 0.0
        for coeff, pauli in zip(h_coeff, pauli_ops):
            expectation = pauli_expectation(vqe_processor, pauli)
            energy += coeff * expectation
        
        return energy
    
    # Optimize VQE parameters
    initial_params = [0.1, 0.1, 0.1]
    result = minimize(h2_hamiltonian_energy, initial_params, method='Nelder-Mead')
    
    print(f"   Optimized energy: {result.fun:.4f} Ha")
    print(f"   Optimal parameters: {result.x}")
    print(f"   Classical H₂ ground state: -1.137 Ha (reference)")


def train_hybrid_model():
    """Train hybrid quantum-classical model."""
    print("\n🧠 Hybrid Quantum-Classical Neural Network Training")
    print("=" * 60)
    
    # Generate synthetic data
    n_samples = 500
    input_dim = 256
    
    X = torch.randn(n_samples, input_dim)
    # Create structured labels based on input statistics
    y = (torch.sum(X > 0, dim=1) % 10).long()
    
    # Create model
    model = HybridQuantumClassicalNetwork(n_qubits=4, classical_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    losses = []
    
    print("Training hybrid model...")
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward pass (batch processing)
        batch_size = 32
        epoch_loss = 0.0
        
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (n_samples // batch_size)
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X[:100])
        test_predictions = torch.argmax(test_outputs, dim=1)
        accuracy = (test_predictions == y[:100]).float().mean()
        
        print(f"   Final test accuracy: {accuracy:.3f}")
    
    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title('Hybrid Quantum-Classical Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('hybrid_training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function."""
    print("🌟 Quantum-Photonic Interface Demonstration")
    print("=" * 50)
    
    try:
        # Demonstrate quantum algorithms
        demonstrate_quantum_algorithms()
        
        # Train hybrid quantum-classical model
        train_hybrid_model()
        
        print(f"\n✅ Quantum-photonic demonstration completed successfully!")
        print(f"   - Implemented QFT, Grover's algorithm, and VQE")
        print(f"   - Demonstrated hybrid quantum-classical neural network")
        print(f"   - All algorithms use dual-rail photonic encoding")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()