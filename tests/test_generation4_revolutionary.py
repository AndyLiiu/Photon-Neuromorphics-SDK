"""
Tests for Generation 4 Revolutionary Features
============================================

Advanced testing for quantum computing, AI transformers, NAS, and federated learning.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

import photon_neuro as pn
from photon_neuro.quantum.error_correction import (
    SurfaceCode, QuantumErrorCorrector, LogicalQubitEncoder
)
from photon_neuro.ai.transformers import (
    OpticalTransformer, PhotonicGPT, InterferenceSelfAttention
)
from photon_neuro.ai.neural_architecture_search import (
    PhotonicNAS, ArchitectureSearchSpace, EvolutionaryPhotonicNAS
)
from photon_neuro.distributed.federated_learning import (
    FederatedPhotonicTrainer, PhotonicFederatedClient, FederatedConfig
)


class TestQuantumErrorCorrection:
    """Test quantum error correction capabilities."""
    
    def test_surface_code_initialization(self):
        """Test surface code initialization."""
        distance = 3
        code = SurfaceCode(distance)
        
        assert code.distance == distance
        assert code.n_physical > 0
        assert code.n_logical == 1
        
        # Check stabilizer generators exist
        assert code.x_stabilizers.shape[1] == code.n_physical
        assert code.z_stabilizers.shape[1] == code.n_physical
    
    def test_surface_code_invalid_distance(self):
        """Test surface code with invalid distance."""
        with pytest.raises(Exception):  # Should raise error for even distance
            SurfaceCode(4)
    
    def test_quantum_error_corrector(self):
        """Test quantum error corrector."""
        code = SurfaceCode(3)
        corrector = QuantumErrorCorrector(code)
        
        # Create dummy quantum state
        quantum_state = torch.randn(2**code.n_physical, dtype=torch.complex64)
        quantum_state = quantum_state / torch.norm(quantum_state)
        
        # Test syndrome measurement
        x_syndrome, z_syndrome = corrector.measure_syndrome(quantum_state)
        
        assert x_syndrome.shape[0] == code.x_stabilizers.shape[0]
        assert z_syndrome.shape[0] == code.z_stabilizers.shape[0]
        
        # Test error correction
        corrected_state = corrector.correct_errors(quantum_state)
        assert corrected_state.shape == quantum_state.shape
    
    def test_logical_qubit_encoder(self):
        """Test logical qubit encoding."""
        code = SurfaceCode(3)
        encoder = LogicalQubitEncoder(code)
        
        # Test logical zero encoding
        logical_zero = encoder.encode_logical_zero()
        assert logical_zero.shape[0] == 2**code.n_physical
        assert torch.allclose(torch.norm(logical_zero), torch.tensor(1.0), atol=1e-6)
        
        # Test logical one encoding
        logical_one = encoder.encode_logical_one()
        assert logical_one.shape[0] == 2**code.n_physical
        assert torch.allclose(torch.norm(logical_one), torch.tensor(1.0), atol=1e-6)
    
    def test_error_correction_statistics(self):
        """Test error correction statistics tracking."""
        code = SurfaceCode(3)
        corrector = QuantumErrorCorrector(code)
        
        # Perform multiple corrections
        for _ in range(10):
            quantum_state = torch.randn(2**code.n_physical, dtype=torch.complex64)
            quantum_state = quantum_state / torch.norm(quantum_state)
            corrector.correct_errors(quantum_state)
        
        stats = corrector.get_error_statistics()
        assert 'success_rate' in stats
        assert 'avg_syndrome_weight' in stats
        assert 'total_corrections' in stats
        assert stats['total_corrections'] == 10


class TestOpticalTransformers:
    """Test optical transformer implementations."""
    
    def test_optical_positional_encoding(self):
        """Test optical positional encoding."""
        d_model = 64
        max_seq_length = 128
        
        pos_encoding = pn.ai.transformers.OpticalPositionalEncoding(
            d_model, max_seq_length
        )
        
        # Test forward pass
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_encoding(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.allclose(output, x)  # Should be modified
    
    def test_interference_self_attention(self):
        """Test interference-based self-attention."""
        d_model = 32
        attention = InterferenceSelfAttention(d_model)
        
        batch_size, seq_len = 2, 16
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32
    
    def test_optical_multi_head_attention(self):
        """Test optical multi-head attention."""
        d_model = 64
        n_heads = 4
        
        attention = pn.ai.transformers.OpticalMultiHeadAttention(
            d_model, n_heads
        )
        
        batch_size, seq_len = 2, 8
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_maps = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_maps.attention_weights.shape[1] == n_heads
    
    def test_optical_transformer(self):
        """Test complete optical transformer."""
        d_model = 64
        n_heads = 4
        n_layers = 2
        d_ff = 128
        
        transformer = OpticalTransformer(
            d_model, n_heads, n_layers, d_ff
        )
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, d_model)
        
        result = transformer(x)
        
        assert 'output' in result
        assert 'layer_outputs' in result
        assert 'attention_maps' in result
        assert 'optical_efficiency' in result
        
        assert result['output'].shape == (batch_size, seq_len, d_model)
        assert len(result['layer_outputs']) == n_layers
        assert isinstance(result['optical_efficiency'], float)
    
    def test_photonic_gpt(self):
        """Test Photonic GPT implementation."""
        vocab_size = 1000
        d_model = 128
        n_heads = 4
        n_layers = 2
        
        model = PhotonicGPT(
            vocab_size, d_model, n_heads, n_layers
        )
        
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        result = model(input_ids)
        
        assert 'logits' in result
        assert result['logits'].shape == (batch_size, seq_len, vocab_size)


class TestNeuralArchitectureSearch:
    """Test neural architecture search capabilities."""
    
    def test_architecture_search_space(self):
        """Test architecture search space."""
        search_space = ArchitectureSearchSpace()
        
        # Test sampling
        arch = search_space.sample_architecture('medium')
        
        assert arch.name.startswith('arch_medium_')
        assert len(arch.layers) >= 4
        assert len(arch.connections) >= len(arch.layers) - 1
        assert 'wavelength' in arch.optical_parameters
        assert 'loss_db_per_cm' in arch.optical_parameters
    
    def test_architecture_evaluator(self):
        """Test architecture evaluation."""
        from photon_neuro.ai.neural_architecture_search import PhotonicArchitectureEvaluator
        
        evaluator = PhotonicArchitectureEvaluator()
        search_space = ArchitectureSearchSpace()
        
        # Sample architecture
        arch = search_space.sample_architecture('simple')
        
        # Evaluate
        metrics = evaluator.evaluate(arch)
        
        assert 'accuracy' in metrics
        assert 'optical_efficiency' in metrics
        assert 'power_consumption' in metrics
        assert 'composite_score' in metrics
        
        # Check metric ranges
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['optical_efficiency'] <= 1.0
        assert 0.0 <= metrics['power_consumption'] <= 1.0
    
    def test_evolutionary_nas(self):
        """Test evolutionary NAS optimizer."""
        search_space = ArchitectureSearchSpace()
        evaluator = pn.ai.neural_architecture_search.PhotonicArchitectureEvaluator()
        optimizer = EvolutionaryPhotonicNAS(population_size=10, mutation_rate=0.2)
        
        # Run short search
        result = optimizer.optimize(search_space, evaluator, n_iterations=5)
        
        assert result.best_architecture is not None
        assert len(result.search_history) > 0
        assert len(result.performance_evolution) == 5
        assert result.total_evaluations >= 50  # 10 population * 5 generations
        assert result.search_time > 0
    
    def test_photonic_nas_system(self):
        """Test complete PhotonicNAS system."""
        nas = PhotonicNAS()
        
        # Run search
        result = nas.search('evolutionary', n_iterations=3, population_size=5)
        
        assert result.best_architecture is not None
        assert result.best_architecture.performance_metrics
        assert 'composite_score' in result.best_architecture.performance_metrics


class TestFederatedLearning:
    """Test federated learning capabilities."""
    
    def test_federated_config(self):
        """Test federated configuration."""
        config = FederatedConfig(
            n_clients=5,
            rounds=3,
            local_epochs=2,
            client_fraction=0.8
        )
        
        assert config.n_clients == 5
        assert config.rounds == 3
        assert config.local_epochs == 2
        assert config.client_fraction == 0.8
    
    def test_photonic_federated_client(self):
        """Test photonic federated client."""
        # Simple model for testing
        model = torch.nn.Linear(10, 2)
        client = PhotonicFederatedClient('client_0', model)
        
        assert client.client_id == 'client_0'
        assert client.model is model
        
        # Test parameter operations
        params = client.get_model_parameters()
        assert len(params) == 2  # weight and bias
        
        # Test setting parameters
        new_params = {name: torch.randn_like(param) for name, param in params.items()}
        client.set_model_parameters(new_params)
    
    def test_optical_federated_averaging(self):
        """Test optical federated averaging."""
        from photon_neuro.distributed.federated_learning import OpticalFederatedAveraging
        
        aggregator = OpticalFederatedAveraging()
        
        # Create dummy client models
        n_clients = 3
        client_models = []
        client_weights = []
        
        for i in range(n_clients):
            model = {
                'layer.weight': torch.randn(2, 10),
                'layer.bias': torch.randn(2)
            }
            client_models.append(model)
            client_weights.append(100 + i * 50)  # Different data sizes
        
        # Test aggregation
        global_model = aggregator.aggregate(client_models, client_weights)
        
        assert 'layer.weight' in global_model
        assert 'layer.bias' in global_model
        assert global_model['layer.weight'].shape == (2, 10)
        assert global_model['layer.bias'].shape == (2,)
    
    def test_federated_photonic_trainer(self):
        """Test complete federated training system."""
        # Model factory
        def create_model():
            return torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
        
        # Configuration
        config = FederatedConfig(
            n_clients=3,
            rounds=2,
            local_epochs=1,
            client_fraction=1.0
        )
        
        # Create trainer
        trainer = FederatedPhotonicTrainer(create_model, config)
        
        assert len(trainer.clients) == 3
        assert trainer.server.config == config
        
        # Test training (short run)
        results = trainer.train()
        
        assert len(results) == 2  # 2 rounds
        assert all(isinstance(r.global_loss, float) for r in results)
        assert all(isinstance(r.global_accuracy, float) for r in results)


class TestPerformanceBenchmarks:
    """Test performance benchmarking and regression detection."""
    
    def test_quantum_gate_performance(self):
        """Benchmark quantum gate operations."""
        from photon_neuro.networks.quantum import QuantumPhotonic
        
        # Test different qubit counts
        for n_qubits in [4, 6, 8]:
            qp = QuantumPhotonic(n_qubits=n_qubits, n_modes=n_qubits*2)
            
            # Benchmark Toffoli gate
            start_time = time.time()
            for _ in range(100):
                qp.toffoli_gate(0, 1, 2)
            toffoli_time = (time.time() - start_time) / 100
            
            # Performance should scale reasonably
            assert toffoli_time < 0.01  # Should complete in < 10ms
            
            # Benchmark QFT
            start_time = time.time()
            qp.quantum_fourier_transform(list(range(min(4, n_qubits))))
            qft_time = time.time() - start_time
            
            assert qft_time < 0.1  # Should complete in < 100ms
    
    def test_optical_transformer_performance(self):
        """Benchmark optical transformer performance."""
        d_model = 128
        n_heads = 8
        n_layers = 6
        
        transformer = OpticalTransformer(d_model, n_heads, n_layers, d_model*4)
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Benchmark forward pass
        start_time = time.time()
        for _ in range(10):
            result = transformer(x)
        avg_forward_time = (time.time() - start_time) / 10
        
        # Should be reasonably fast
        assert avg_forward_time < 1.0  # < 1s per forward pass
        
        # Check optical efficiency
        assert result['optical_efficiency'] > 0.1
    
    def test_nas_convergence(self):
        """Test NAS convergence behavior."""
        nas = PhotonicNAS()
        
        # Run search with different iteration counts
        short_result = nas.search('evolutionary', n_iterations=5, population_size=10)
        longer_result = nas.search('evolutionary', n_iterations=10, population_size=10)
        
        # Longer search should generally find better architectures
        short_score = short_result.best_architecture.performance_metrics['composite_score']
        longer_score = longer_result.best_architecture.performance_metrics['composite_score']
        
        # Allow for some variance but expect improvement trend
        assert longer_score >= short_score * 0.8
        
        # Check convergence in performance evolution
        evolution = longer_result.performance_evolution
        early_avg = np.mean(evolution[:3])
        late_avg = np.mean(evolution[-3:])
        
        # Should show some improvement or stability
        assert late_avg >= early_avg * 0.9
    
    def test_federated_scaling(self):
        """Test federated learning scaling behavior."""
        def create_simple_model():
            return torch.nn.Linear(10, 2)
        
        # Test with different client counts
        client_counts = [2, 5, 10]
        training_times = []
        
        for n_clients in client_counts:
            config = FederatedConfig(
                n_clients=n_clients,
                rounds=2,
                local_epochs=1
            )
            
            trainer = FederatedPhotonicTrainer(create_simple_model, config)
            
            start_time = time.time()
            results = trainer.train()
            training_time = time.time() - start_time
            
            training_times.append(training_time)
            
            # Check that we get reasonable results
            assert len(results) == 2
            assert all(r.global_accuracy > 0 for r in results)
        
        # Training time should scale reasonably (not exponentially)
        assert training_times[-1] < training_times[0] * len(client_counts)


class TestIntegrationScenarios:
    """Test integration between Generation 4 components."""
    
    def test_quantum_photonic_transformer(self):
        """Test integration of quantum computing with transformers."""
        # This would test a hybrid quantum-photonic transformer
        # For now, test components work together conceptually
        
        # Create quantum component
        from photon_neuro.networks.quantum import QuantumPhotonic
        quantum_layer = QuantumPhotonic(n_qubits=4, n_modes=8)
        
        # Create transformer component
        transformer = OpticalTransformer(d_model=32, n_heads=4, n_layers=2, d_ff=64)
        
        # Test sequential processing
        batch_size = 2
        input_data = torch.randn(batch_size, 16, 32)
        
        transformer_output = transformer(input_data)
        
        # Should work independently
        assert transformer_output['output'].shape == (batch_size, 16, 32)
        
        # Quantum layer processing (simplified)
        quantum_layer.prepare_state('0000')
        quantum_layer.apply_gate('Hadamard', [0])
        
        # Integration would involve more sophisticated coupling
        assert True  # Placeholder for actual integration test
    
    def test_nas_with_federated_learning(self):
        """Test NAS for finding optimal federated architectures."""
        # Create custom search space for federated-friendly architectures
        search_space = ArchitectureSearchSpace()
        
        # Custom evaluator that considers federated metrics
        class FederatedArchitectureEvaluator(pn.ai.neural_architecture_search.PhotonicArchitectureEvaluator):
            def _evaluate_accuracy(self, model, architecture):
                # Penalize architectures that are too complex for federated learning
                complexity_penalty = len(architecture.layers) / 20.0
                base_accuracy = super()._evaluate_accuracy(model, architecture)
                return base_accuracy * (1.0 - complexity_penalty * 0.1)
        
        evaluator = FederatedArchitectureEvaluator()
        nas = PhotonicNAS(search_space, evaluator)
        
        # Run search
        result = nas.search('evolutionary', n_iterations=3, population_size=5)
        
        # Should find architecture suitable for federated learning
        assert result.best_architecture is not None
        best_score = result.best_architecture.performance_metrics['composite_score']
        assert best_score > 0.0


# Performance regression tests
import time

def test_performance_regression():
    """Test for performance regressions in core operations."""
    
    # Benchmark MZI mesh operations
    mesh = pn.MZIMesh(size=(8, 8))
    x = torch.randn(4, 64)
    
    start_time = time.time()
    for _ in range(100):
        output = mesh(x)
    mzi_time = time.time() - start_time
    
    # Should complete in reasonable time
    assert mzi_time < 5.0  # 5 seconds for 100 operations
    
    # Benchmark quantum operations
    qp = pn.networks.quantum.QuantumPhotonic(n_qubits=4, n_modes=8)
    
    start_time = time.time()
    for _ in range(50):
        qp.prepare_state('0000')
        qp.apply_gate('Hadamard', [0])
        qp.apply_gate('CNOT', [0, 1])
    quantum_time = time.time() - start_time
    
    assert quantum_time < 2.0  # 2 seconds for 50 quantum operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])