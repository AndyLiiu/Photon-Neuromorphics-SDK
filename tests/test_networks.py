"""
Tests for photonic neural networks.
"""

import pytest
import torch
import numpy as np
from photon_neuro.networks import (
    PhotonicSNN, PhotonicLIFNeuron, MZIMesh, PhotonicMLP,
    MicroringArray, QuantumPhotonic
)


class TestPhotonicSNN:
    """Test spiking neural networks."""
    
    def test_snn_creation(self):
        """Test SNN creation with different topologies."""
        topology = [10, 20, 5]
        snn = PhotonicSNN(topology, timestep=1e-12)
        
        assert snn.topology == topology
        assert snn.timestep == 1e-12
        assert len(snn.layers) == len(topology) - 1
        
    def test_snn_forward(self):
        """Test SNN forward pass."""
        topology = [8, 4, 2]
        snn = PhotonicSNN(topology, timestep=1e-12)
        
        batch_size = 2
        input_spikes = torch.randn(batch_size, topology[0])
        
        outputs = snn.forward(input_spikes, n_timesteps=10)
        
        # Check output shape: [batch, neurons, time]
        expected_shape = (batch_size, topology[-1], 10)
        assert outputs.shape == expected_shape
        
    def test_snn_efficiency(self):
        """Test SNN efficiency calculation."""
        snn = PhotonicSNN([5, 5], timestep=1e-12)
        
        efficiency = snn.efficiency
        assert 0 <= efficiency <= 1
        
    def test_snn_latency(self):
        """Test SNN latency calculation."""
        snn = PhotonicSNN([3, 6, 3], timestep=1e-12)
        
        latency = snn.latency_ps
        assert latency > 0
        # Should be roughly proportional to number of layers
        assert latency >= len(snn.layers)
        
    def test_stdp_training(self):
        """Test STDP training."""
        snn = PhotonicSNN([4, 3], timestep=1e-12)
        
        input_spikes = torch.randn(1, 4)
        target_spikes = torch.randn(1, 3, 10)  # [batch, neurons, time]
        
        # Test STDP training step
        loss = snn.train_stdp(input_spikes, target_spikes, learning_rate=0.01)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestPhotonicLIFNeuron:
    """Test LIF neuron implementation."""
    
    def test_lif_neuron_creation(self):
        """Test LIF neuron creation."""
        neuron = PhotonicLIFNeuron(threshold=1.0, leak_rate=0.1)
        
        assert neuron.threshold == 1.0
        assert neuron.leak_rate == 0.1
        
    def test_lif_integration(self):
        """Test leaky integration."""
        neuron = PhotonicLIFNeuron(threshold=2.0, leak_rate=0.1)
        
        # Apply input current below threshold
        input_current = torch.tensor([0.5])
        dt = 1e-12
        
        spike = neuron.forward(input_current, dt)
        
        # Should not spike with low input
        assert torch.abs(spike) < 1e-6
        
    def test_lif_spiking(self):
        """Test spike generation."""
        neuron = PhotonicLIFNeuron(threshold=1.0, leak_rate=0.0)  # No leak for simplicity
        
        # Apply current above threshold
        input_current = torch.tensor([2.0])
        dt = 1e-12
        
        spike = neuron.forward(input_current, dt)
        
        # Should generate spike
        assert torch.abs(spike) > 0
        
    def test_refractory_period(self):
        """Test refractory period behavior."""
        neuron = PhotonicLIFNeuron(threshold=1.0, refractory_period=1e-9)
        
        # First spike
        input_current = torch.tensor([2.0])
        spike1 = neuron.forward(input_current, 1e-12)
        
        # Immediate second input (should be in refractory period)
        spike2 = neuron.forward(input_current, 1e-12)
        
        assert torch.abs(spike1) > 0  # First should spike
        assert torch.abs(spike2) < 1e-6  # Second should be blocked
        
    def test_neuron_reset(self):
        """Test neuron state reset."""
        neuron = PhotonicLIFNeuron()
        
        # Build up some membrane potential
        neuron.forward(torch.tensor([0.5]), 1e-12)
        assert neuron.membrane_potential.item() > 0
        
        # Reset neuron
        neuron.reset_state()
        assert neuron.membrane_potential.item() == 0


class TestMZIMesh:
    """Test MZI mesh networks."""
    
    def test_mzi_mesh_creation(self):
        """Test MZI mesh creation."""
        size = (4, 4)
        mesh = MZIMesh(size=size, topology="rectangular")
        
        assert mesh.size == size
        assert mesh.topology == "rectangular"
        assert len(mesh.phase_shifters) == mesh.n_phases
        
    def test_unitary_decomposition(self):
        """Test unitary matrix decomposition."""
        mesh = MZIMesh(size=(3, 3))
        
        # Create a simple unitary matrix
        target_unitary = torch.eye(3, dtype=torch.complex64)
        phases = mesh.decompose(target_unitary)
        
        assert len(phases) == mesh.n_phases
        assert torch.is_real(phases)
        
    def test_mzi_forward(self):
        """Test MZI mesh forward pass."""
        mesh = MZIMesh(size=(4, 4))
        
        batch_size = 2
        input_fields = torch.randn(batch_size, 4, dtype=torch.complex64)
        
        outputs = mesh.forward(input_fields)
        
        assert outputs.shape == (batch_size, 4)
        assert torch.is_complex(outputs)
        
    def test_phase_setting(self):
        """Test phase shifter setting."""
        mesh = MZIMesh(size=(2, 2))
        
        phases = torch.tensor([0.5, 1.0, 1.5])
        mesh.set_phases(phases)
        
        assert torch.allclose(mesh.phases, phases)
        
    def test_unitary_measurement(self):
        """Test unitary matrix measurement."""
        mesh = MZIMesh(size=(2, 2))
        
        unitary = mesh.measure_unitary()
        
        assert unitary.shape == (2, 2)
        assert torch.is_complex(unitary)
        
        # Check unitarity (U†U = I)
        identity = torch.matmul(unitary.conj().T, unitary)
        expected_identity = torch.eye(2, dtype=torch.complex64)
        
        # Allow some numerical error
        assert torch.allclose(identity, expected_identity, atol=1e-3)


class TestPhotonicMLP:
    """Test photonic multi-layer perceptron."""
    
    def test_mlp_creation(self):
        """Test MLP creation."""
        layer_sizes = [8, 16, 4]
        mlp = PhotonicMLP(layer_sizes, activation="relu")
        
        assert mlp.layer_sizes == layer_sizes
        assert mlp.activation == "relu"
        assert len(mlp.layers) == len(layer_sizes) - 1
        
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        layer_sizes = [6, 4, 2]
        mlp = PhotonicMLP(layer_sizes)
        
        batch_size = 3
        inputs = torch.randn(batch_size, layer_sizes[0], dtype=torch.complex64)
        
        outputs = mlp.forward(inputs)
        
        assert outputs.shape == (batch_size, layer_sizes[-1])
        assert torch.is_complex(outputs)
        
    def test_optical_backprop(self):
        """Test optical backpropagation training."""
        mlp = PhotonicMLP([4, 3, 2])
        
        inputs = torch.randn(2, 4, dtype=torch.complex64)
        targets = torch.randn(2, 2, dtype=torch.complex64)
        
        loss = mlp.train_optical_backprop(inputs, targets, learning_rate=0.01)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestMicroringArray:
    """Test microring resonator arrays."""
    
    def test_array_creation(self):
        """Test microring array creation."""
        n_rings = 16
        array = MicroringArray(n_rings=n_rings, quality_factor=10000)
        
        assert array.n_rings == n_rings
        assert array.q_factor == 10000
        assert len(array.rings) == n_rings
        
    def test_weight_encoding(self):
        """Test weight encoding to ring parameters."""
        array = MicroringArray(n_rings=9)
        
        weights = np.random.randn(3, 3)
        resonances = array.encode_weights(weights, encoding="wavelength_division")
        
        assert len(resonances) == weights.size
        
    def test_fabrication_variations(self):
        """Test adding fabrication variations."""
        array = MicroringArray(n_rings=8)
        
        # Store original parameters
        original_radii = [ring.radius for ring in array.rings]
        original_couplings = [ring.coupling for ring in array.rings]
        
        # Add variations
        array.add_variations(radius_sigma=5e-9, coupling_sigma=0.02)
        
        # Check that parameters have changed
        new_radii = [ring.radius for ring in array.rings]
        new_couplings = [ring.coupling for ring in array.rings]
        
        # At least some should be different (with high probability)
        assert not all(abs(o - n) < 1e-12 for o, n in zip(original_radii, new_radii))
        
    def test_array_forward(self):
        """Test array forward pass."""
        array = MicroringArray(n_rings=4)
        
        input_fields = torch.randn(2, 4, dtype=torch.complex64)
        outputs = array.forward(input_fields)
        
        assert outputs.shape == (2, 4)
        assert torch.is_complex(outputs)


class TestQuantumPhotonic:
    """Test quantum photonic systems."""
    
    def test_quantum_system_creation(self):
        """Test quantum photonic system creation."""
        qp = QuantumPhotonic(n_qubits=3, n_modes=6, encoding="dual_rail")
        
        assert qp.n_qubits == 3
        assert qp.n_modes == 6
        assert qp.encoding == "dual_rail"
        assert qp.n_basis_states == 2**3
        
    def test_state_preparation(self):
        """Test quantum state preparation."""
        qp = QuantumPhotonic(n_qubits=2, n_modes=4)
        
        qp.prepare_state("01")  # |01⟩ state
        
        # Check that state is prepared correctly
        assert torch.abs(qp.quantum_state[1] - 1.0) < 1e-6  # |01⟩ = index 1
        assert torch.abs(qp.quantum_state[0]) < 1e-6  # Other states should be zero
        
    def test_quantum_gates(self):
        """Test quantum gate operations."""
        qp = QuantumPhotonic(n_qubits=2, n_modes=4)
        
        # Start in |00⟩
        qp.prepare_state("00")
        
        # Apply Hadamard to first qubit: |00⟩ → (|00⟩ + |10⟩)/√2
        qp.apply_gate("H", [0])
        
        # Check superposition state
        expected_amp = 1/np.sqrt(2)
        assert abs(qp.quantum_state[0].item() - expected_amp) < 1e-5  # |00⟩
        assert abs(qp.quantum_state[2].item() - expected_amp) < 1e-5  # |10⟩
        
    def test_quantum_measurement(self):
        """Test quantum state measurement."""
        qp = QuantumPhotonic(n_qubits=2, n_modes=4)
        
        qp.prepare_state("11")
        counts = qp.measure(shots=100)
        
        # Should measure |11⟩ most of the time
        assert "11" in counts
        assert counts["11"] > 80  # Most measurements should be |11⟩
        
    def test_quantum_forward(self):
        """Test quantum circuit forward pass."""
        qp = QuantumPhotonic(n_qubits=2, n_modes=4)
        
        inputs = torch.randn(1, 4)  # Classical input
        outputs = qp.forward(inputs)
        
        assert outputs.shape == (1, 2)  # Expectation values for 2 qubits
        assert torch.is_real(outputs)
        
    def test_fidelity_calculation(self):
        """Test quantum state fidelity."""
        qp = QuantumPhotonic(n_qubits=2, n_modes=4)
        
        qp.prepare_state("00")
        target_state = torch.zeros(4, dtype=torch.complex64)
        target_state[0] = 1.0  # |00⟩ state
        
        fidelity = qp.get_fidelity(target_state)
        
        assert abs(fidelity - 1.0) < 1e-6  # Should have perfect fidelity


@pytest.fixture
def sample_networks():
    """Fixture providing sample networks for testing."""
    return {
        'snn': PhotonicSNN([4, 3], timestep=1e-12),
        'mzi': MZIMesh(size=(4, 4)),
        'mlp': PhotonicMLP([6, 4, 2]),
        'microring': MicroringArray(n_rings=8),
        'quantum': QuantumPhotonic(n_qubits=2, n_modes=4)
    }


class TestNetworkIntegration:
    """Test integration between different network types."""
    
    def test_network_netlists(self, sample_networks):
        """Test netlist generation for all networks."""
        for name, network in sample_networks.items():
            netlist = network.to_netlist()
            
            assert isinstance(netlist, dict)
            assert "type" in netlist
            assert isinstance(netlist["type"], str)
            
    def test_network_forward_shapes(self, sample_networks):
        """Test that all networks produce correct output shapes."""
        # Skip quantum network for this test as it has different interface
        test_networks = {k: v for k, v in sample_networks.items() if k != 'quantum'}
        
        for name, network in test_networks.items():
            if name == 'snn':
                inputs = torch.randn(1, 4)
                outputs = network.forward(inputs, n_timesteps=5)
                assert outputs.shape == (1, 3, 5)  # [batch, neurons, time]
            elif name == 'mzi':
                inputs = torch.randn(1, 4, dtype=torch.complex64)
                outputs = network.forward(inputs)
                assert outputs.shape == (1, 4)
            elif name == 'mlp':
                inputs = torch.randn(1, 6, dtype=torch.complex64)
                outputs = network.forward(inputs)
                assert outputs.shape == (1, 2)
            elif name == 'microring':
                inputs = torch.randn(1, 8, dtype=torch.complex64)
                outputs = network.forward(inputs)
                assert outputs.shape == (1, 8)


if __name__ == "__main__":
    pytest.main([__file__])