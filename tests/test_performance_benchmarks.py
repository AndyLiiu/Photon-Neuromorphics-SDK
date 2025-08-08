"""
Comprehensive performance benchmarking test suite.
"""

import pytest
import numpy as np
import torch
import time
import psutil
import os
from typing import Dict, List, Tuple, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    import photon_neuro as pn
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    pytest.skip(f"PhotonNeuro import failed: {e}", allow_module_level=True)


class SystemProfiler:
    """System resource profiler for benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        self.start_time = None
        
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.perf_counter()
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        current_time = time.perf_counter()
        current_memory = self.process.memory_info().rss
        
        return {
            'elapsed_time_s': current_time - self.start_time if self.start_time else 0,
            'memory_usage_mb': current_memory / (1024 * 1024),
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'memory_delta_mb': (current_memory - self.initial_memory) / (1024 * 1024),
            'cpu_percent': self.process.cpu_percent()
        }


@pytest.fixture(scope="session")
def profiler():
    """Session-scoped system profiler."""
    return SystemProfiler()


class TestMZIPerformance:
    """MZI mesh performance benchmarking."""
    
    @pytest.mark.parametrize("size", [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)])
    def test_mzi_scaling_performance(self, size, profiler):
        """Test MZI mesh performance scaling with size."""
        profiler.start_profiling()
        
        rows, cols = size
        
        # Create MZI mesh
        mzi_mesh = pn.MZIMesh(size=size, topology='rectangular')
        
        # Generate test data
        batch_size = 32
        input_batch = torch.randn(batch_size, rows, dtype=torch.complex64)
        
        # Warmup
        for _ in range(5):
            _ = mzi_mesh(input_batch[0])
        
        # Benchmark forward passes
        n_runs = 100
        start_time = time.perf_counter()
        
        for i in range(n_runs):
            batch_idx = i % batch_size
            output = mzi_mesh(input_batch[batch_idx])
            profiler.update_peak_memory()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time_ms = total_time / n_runs * 1000
        throughput = n_runs / total_time
        operations_per_second = rows * cols * throughput
        
        metrics = profiler.get_metrics()
        
        # Performance assertions
        assert avg_time_ms > 0, "Invalid timing measurement"
        assert throughput > 0, "Invalid throughput calculation"
        
        # Size-dependent performance thresholds
        if rows <= 16:
            assert avg_time_ms < 10, f"MZI {size} too slow: {avg_time_ms:.2f}ms"
        elif rows <= 32:
            assert avg_time_ms < 50, f"MZI {size} too slow: {avg_time_ms:.2f}ms"
        else:  # 64x64
            assert avg_time_ms < 200, f"MZI {size} too slow: {avg_time_ms:.2f}ms"
        
        # Memory usage should be reasonable
        assert metrics['memory_delta_mb'] < 100, f"Excessive memory usage: {metrics['memory_delta_mb']:.1f}MB"
        
        print(f"MZI {size}: {avg_time_ms:.2f}ms, {operations_per_second:.0f} ops/s, {metrics['memory_delta_mb']:.1f}MB")
    
    def test_mzi_batch_processing(self, profiler):
        """Test MZI batch processing performance."""
        profiler.start_profiling()
        
        mzi_size = (16, 16)
        mzi_mesh = pn.MZIMesh(size=mzi_size, topology='rectangular')
        
        # Test different batch sizes
        batch_sizes = [1, 4, 16, 64, 256]
        performance_data = []
        
        for batch_size in batch_sizes:
            input_batch = torch.randn(batch_size, mzi_size[0], dtype=torch.complex64)
            
            # Warmup
            _ = mzi_mesh(input_batch)
            
            # Benchmark
            start_time = time.perf_counter()
            
            for _ in range(10):
                output_batch = mzi_mesh(input_batch)
                profiler.update_peak_memory()
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            time_per_sample = total_time / (10 * batch_size) * 1000  # ms per sample
            
            performance_data.append({
                'batch_size': batch_size,
                'time_per_sample_ms': time_per_sample,
                'total_time_s': total_time,
                'memory_mb': profiler.get_metrics()['memory_usage_mb']
            })
        
        # Check batch efficiency
        single_time = performance_data[0]['time_per_sample_ms']
        large_batch_time = performance_data[-1]['time_per_sample_ms']
        
        # Large batches should be more efficient per sample
        efficiency_improvement = single_time / large_batch_time
        assert efficiency_improvement > 1.5, f"Poor batch efficiency: {efficiency_improvement:.2f}x"
        
        print(f"Batch efficiency improvement: {efficiency_improvement:.2f}x")
    
    def test_mzi_memory_efficiency(self, profiler):
        """Test MZI memory usage patterns."""
        profiler.start_profiling()
        
        initial_memory = profiler.get_metrics()['memory_usage_mb']
        memory_usage = []
        
        sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
        
        for size in sizes:
            mzi_mesh = pn.MZIMesh(size=size, topology='rectangular')
            input_tensor = torch.randn(size[0], dtype=torch.complex64)
            
            # Force computation to allocate memory
            for _ in range(10):
                _ = mzi_mesh(input_tensor)
            
            current_memory = profiler.get_metrics()['memory_usage_mb']
            memory_usage.append({
                'size': size,
                'memory_mb': current_memory - initial_memory,
                'parameters': size[0] * size[1]
            })
        
        # Memory usage should scale reasonably with model size
        for i in range(1, len(memory_usage)):
            prev_params = memory_usage[i-1]['parameters']
            curr_params = memory_usage[i]['parameters']
            prev_memory = memory_usage[i-1]['memory_mb']
            curr_memory = memory_usage[i]['memory_mb']
            
            param_ratio = curr_params / prev_params
            memory_ratio = curr_memory / prev_memory if prev_memory > 0 else float('inf')
            
            # Memory scaling should be reasonable (not exponential)
            assert memory_ratio < param_ratio * 2, f"Memory scaling too steep: {memory_ratio:.2f}x"
        
        print(f"Memory usage scaling: {[m['memory_mb'] for m in memory_usage]}")


class TestMicroringPerformance:
    """Microring resonator performance benchmarking."""
    
    @pytest.mark.parametrize("n_rings", [4, 8, 16, 32, 64])
    def test_microring_array_performance(self, n_rings, profiler):
        """Test microring array performance scaling."""
        profiler.start_profiling()
        
        # Create microring array
        ring_array = pn.MicroringArray(
            n_rings=n_rings,
            free_spectral_range=20e9,
            quality_factor=10000
        )
        
        # Test parameters
        n_wavelengths = 100
        wavelengths = torch.linspace(1540e-9, 1560e-9, n_wavelengths)
        input_power = torch.ones(n_wavelengths) * 1e-3  # 1 mW
        
        # Warmup
        for _ in range(5):
            _ = ring_array.simulate_transmission(wavelengths, input_power)
        
        # Benchmark
        n_runs = 50
        start_time = time.perf_counter()
        
        for _ in range(n_runs):
            transmission = ring_array.simulate_transmission(wavelengths, input_power)
            profiler.update_peak_memory()
        
        end_time = time.perf_counter()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        avg_time_ms = total_time / n_runs * 1000
        operations_per_second = n_rings * n_wavelengths * n_runs / total_time
        
        metrics = profiler.get_metrics()
        
        # Performance thresholds
        if n_rings <= 16:
            assert avg_time_ms < 20, f"Microring {n_rings} too slow: {avg_time_ms:.2f}ms"
        elif n_rings <= 32:
            assert avg_time_ms < 50, f"Microring {n_rings} too slow: {avg_time_ms:.2f}ms"
        else:
            assert avg_time_ms < 100, f"Microring {n_rings} too slow: {avg_time_ms:.2f}ms"
        
        assert operations_per_second > 1000, f"Low throughput: {operations_per_second:.0f} ops/s"
        
        print(f"Microring {n_rings}: {avg_time_ms:.2f}ms, {operations_per_second:.0f} ops/s")
    
    def test_microring_wavelength_sweep(self, profiler):
        """Test microring performance with different wavelength resolutions."""
        profiler.start_profiling()
        
        ring_array = pn.MicroringArray(n_rings=16, free_spectral_range=20e9, quality_factor=10000)
        
        wavelength_counts = [50, 100, 200, 500, 1000]
        performance_data = []
        
        for n_wavelengths in wavelength_counts:
            wavelengths = torch.linspace(1540e-9, 1560e-9, n_wavelengths)
            input_power = torch.ones(n_wavelengths) * 1e-3
            
            # Benchmark
            start_time = time.perf_counter()
            
            for _ in range(20):
                _ = ring_array.simulate_transmission(wavelengths, input_power)
                profiler.update_peak_memory()
            
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            time_per_wavelength = total_time / (20 * n_wavelengths) * 1000  # ms per wavelength
            
            performance_data.append({
                'n_wavelengths': n_wavelengths,
                'time_per_wavelength_us': time_per_wavelength * 1000,
                'total_time_s': total_time
            })
        
        # Performance should scale linearly with wavelength count
        for data in performance_data:
            assert data['time_per_wavelength_us'] < 100, f"Too slow per wavelength: {data['time_per_wavelength_us']:.1f}μs"
        
        print(f"Wavelength sweep performance: {[d['time_per_wavelength_us'] for d in performance_data]} μs/wavelength")


class TestNetworkPerformance:
    """Photonic network performance benchmarking."""
    
    def test_spiking_network_performance(self, profiler):
        """Test photonic SNN performance."""
        profiler.start_profiling()
        
        # Create photonic SNN
        snn = pn.PhotonicSNN(
            topology=[784, 256, 128, 10],
            neuron_model='photonic_lif',
            synapse_type='microring',
            timestep=1e-12
        )
        
        # Generate spike input data
        batch_size = 16
        n_timesteps = 100
        spike_input = torch.rand(batch_size, n_timesteps, 784) < 0.1  # 10% spike probability
        
        # Warmup
        _ = snn(spike_input[:1])
        
        # Benchmark
        start_time = time.perf_counter()
        
        for i in range(batch_size):
            output = snn(spike_input[i:i+1])
            profiler.update_peak_memory()
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        time_per_sample = total_time / batch_size * 1000  # ms per sample
        
        metrics = profiler.get_metrics()
        
        # SNN should process samples in reasonable time
        assert time_per_sample < 100, f"SNN too slow: {time_per_sample:.2f}ms per sample"
        assert metrics['memory_delta_mb'] < 200, f"Excessive SNN memory: {metrics['memory_delta_mb']:.1f}MB"
        
        print(f"SNN performance: {time_per_sample:.2f}ms per sample, {metrics['memory_delta_mb']:.1f}MB")
    
    def test_quantum_interface_performance(self, profiler):
        """Test quantum-photonic interface performance."""
        profiler.start_profiling()
        
        # Create quantum interface
        quantum_interface = pn.QuantumPhotonic(
            n_qubits=4,
            n_modes=8,
            encoding='dual_rail'
        )
        
        # Prepare quantum states
        n_states = 100
        quantum_states = []
        
        for i in range(n_states):
            state_string = format(i % 16, '04b')  # 4-bit states
            quantum_interface.prepare_state(state_string)
            
            # Apply quantum gates
            quantum_interface.apply_gate('Hadamard', qubits=[0])
            quantum_interface.apply_gate('CNOT', qubits=[0, 1])
            
            profiler.update_peak_memory()
        
        metrics = profiler.get_metrics()
        total_time = metrics['elapsed_time_s']
        
        # Quantum operations should complete in reasonable time
        time_per_operation = total_time / n_states * 1000  # ms per operation
        assert time_per_operation < 10, f"Quantum operations too slow: {time_per_operation:.2f}ms"
        
        print(f"Quantum interface: {time_per_operation:.2f}ms per operation")
    
    def test_training_performance(self, profiler):
        """Test optical training performance."""
        profiler.start_profiling()
        
        # Simple hybrid model
        class SimpleHybridModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mzi = pn.MZIMesh(size=(8, 8), topology='rectangular')
                self.linear = torch.nn.Linear(8, 4)
                
            def forward(self, x):
                x = self.mzi(x.to(torch.complex64))
                x = torch.abs(x)  # Photodetection
                return self.linear(x)
        
        model = SimpleHybridModel()
        optimizer = pn.OpticalAdam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        
        # Training data
        batch_size = 16
        n_batches = 20
        input_dim = 8
        
        # Training benchmark
        training_times = []
        
        for batch in range(n_batches):
            batch_start = time.perf_counter()
            
            # Generate batch
            inputs = torch.randn(batch_size, input_dim)
            targets = torch.randn(batch_size, 4)
            
            # Training step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_end = time.perf_counter()
            training_times.append(batch_end - batch_start)
            profiler.update_peak_memory()
        
        avg_batch_time = np.mean(training_times) * 1000  # ms
        metrics = profiler.get_metrics()
        
        # Training should be efficient
        assert avg_batch_time < 100, f"Training too slow: {avg_batch_time:.2f}ms per batch"
        assert metrics['peak_memory_mb'] < 500, f"Excessive training memory: {metrics['peak_memory_mb']:.1f}MB"
        
        print(f"Training performance: {avg_batch_time:.2f}ms per batch, {metrics['peak_memory_mb']:.1f}MB peak")


class TestSystemBenchmarks:
    """System-level performance benchmarks."""
    
    def test_end_to_end_inference(self, profiler):
        """Test complete end-to-end inference pipeline."""
        profiler.start_profiling()
        
        # Create complex photonic model
        class ComplexPhotonicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_encoder = torch.nn.Linear(784, 256)
                self.mzi_mesh = pn.MZIMesh(size=(16, 16), topology='rectangular')
                self.microring_bank = pn.MicroringArray(n_rings=32, free_spectral_range=20e9)
                self.output_decoder = torch.nn.Linear(256, 10)
                
            def forward(self, x):
                # Electronic preprocessing
                x = torch.tanh(self.input_encoder(x))
                
                # Photonic processing (simplified)
                x_complex = x[:16].to(torch.complex64)  # Take first 16 components
                x_optical = self.mzi_mesh(x_complex)
                
                # Simulate photodetection
                x_detected = torch.abs(x_optical)
                
                # Pad back to original size for decoder
                x_padded = torch.zeros(256)
                x_padded[:len(x_detected)] = x_detected
                
                # Electronic postprocessing
                output = self.output_decoder(x_padded)
                return output
        
        model = ComplexPhotonicModel()
        model.eval()
        
        # Benchmark inference
        n_samples = 100
        inference_times = []
        
        with torch.no_grad():
            for i in range(n_samples):
                # Generate input
                input_data = torch.randn(784)
                
                start_time = time.perf_counter()
                output = model(input_data)
                end_time = time.perf_counter()
                
                inference_times.append(end_time - start_time)
                profiler.update_peak_memory()
        
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        std_inference_time = np.std(inference_times) * 1000   # ms
        throughput = 1000 / avg_inference_time  # samples per second
        
        metrics = profiler.get_metrics()
        
        # Performance requirements
        assert avg_inference_time < 50, f"Inference too slow: {avg_inference_time:.2f}ms"
        assert std_inference_time < avg_inference_time * 0.5, f"High variance: {std_inference_time:.2f}ms"
        assert throughput > 20, f"Low throughput: {throughput:.1f} samples/s"
        
        print(f"End-to-end inference: {avg_inference_time:.2f}±{std_inference_time:.2f}ms, {throughput:.1f} samples/s")
    
    def test_concurrent_processing(self, profiler):
        """Test concurrent processing performance."""
        import threading
        import concurrent.futures
        
        profiler.start_profiling()
        
        def photonic_computation(thread_id):
            """Photonic computation task for threading test."""
            # Create independent model instance
            mzi = pn.MZIMesh(size=(8, 8), topology='rectangular')
            
            # Perform computations
            results = []
            for i in range(20):
                input_tensor = torch.randn(8, dtype=torch.complex64)
                output = mzi(input_tensor)
                results.append(torch.abs(output).sum().item())
            
            return results
        
        # Test concurrent execution
        n_threads = min(4, os.cpu_count() or 1)
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(photonic_computation, i) for i in range(n_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.perf_counter()
        
        concurrent_time = end_time - start_time
        
        # Test sequential execution for comparison
        start_time = time.perf_counter()
        sequential_results = [photonic_computation(i) for i in range(n_threads)]
        end_time = time.perf_counter()
        
        sequential_time = end_time - start_time
        
        # Calculate speedup
        speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
        
        metrics = profiler.get_metrics()
        
        # Concurrent execution should show some benefit
        assert speedup > 0.8, f"Poor concurrent performance: {speedup:.2f}x speedup"
        assert len(results) == n_threads, "Not all concurrent tasks completed"
        
        print(f"Concurrent processing: {speedup:.2f}x speedup with {n_threads} threads")
    
    @pytest.mark.slow
    def test_stress_test(self, profiler):
        """Stress test for system stability."""
        profiler.start_profiling()
        
        # Create multiple components
        components = [
            pn.MZIMesh(size=(16, 16), topology='rectangular'),
            pn.MicroringArray(n_rings=32, free_spectral_range=20e9),
            pn.PhotonicSNN(topology=[100, 50, 10], neuron_model='photonic_lif')
        ]
        
        # Run intensive computations
        n_iterations = 50
        error_count = 0
        
        for iteration in range(n_iterations):
            try:
                # MZI mesh computation
                mzi_input = torch.randn(16, dtype=torch.complex64)
                mzi_output = components[0](mzi_input)
                
                # Microring simulation
                wavelengths = torch.linspace(1540e-9, 1560e-9, 100)
                ring_transmission = components[1].simulate_transmission(
                    wavelengths, torch.ones(100) * 1e-3
                )
                
                # SNN computation
                spike_input = torch.rand(1, 10, 100) < 0.1
                snn_output = components[2](spike_input)
                
                profiler.update_peak_memory()
                
            except Exception as e:
                error_count += 1
                print(f"Error in iteration {iteration}: {e}")
        
        metrics = profiler.get_metrics()
        
        # System should remain stable under stress
        error_rate = error_count / n_iterations
        assert error_rate < 0.1, f"High error rate under stress: {error_rate:.1%}"
        assert metrics['peak_memory_mb'] < 1000, f"Excessive peak memory: {metrics['peak_memory_mb']:.1f}MB"
        
        print(f"Stress test: {error_rate:.1%} error rate, {metrics['peak_memory_mb']:.1f}MB peak memory")


def generate_performance_report(test_results: Dict[str, Any]) -> str:
    """Generate comprehensive performance report."""
    report = "# Photonic Neural Network Performance Report\n\n"
    
    report += "## System Information\n"
    report += f"- CPU Count: {os.cpu_count()}\n"
    report += f"- Available Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB\n"
    report += f"- Python Version: {os.sys.version.split()[0]}\n\n"
    
    report += "## Performance Summary\n"
    for test_name, results in test_results.items():
        report += f"### {test_name}\n"
        for metric, value in results.items():
            report += f"- {metric}: {value}\n"
        report += "\n"
    
    return report


@pytest.mark.performance
class TestPerformanceRegression:
    """Performance regression testing."""
    
    def test_performance_baselines(self, profiler):
        """Test against performance baselines."""
        baselines = {
            'mzi_8x8_forward_ms': 5.0,
            'microring_16_simulation_ms': 10.0,
            'snn_100_spikes_ms': 20.0,
            'e2e_inference_ms': 30.0
        }
        
        current_performance = {}
        
        # MZI performance
        mzi = pn.MZIMesh(size=(8, 8))
        input_tensor = torch.randn(8, dtype=torch.complex64)
        
        start_time = time.perf_counter()
        for _ in range(100):
            _ = mzi(input_tensor)
        end_time = time.perf_counter()
        
        current_performance['mzi_8x8_forward_ms'] = (end_time - start_time) / 100 * 1000
        
        # Compare against baselines
        for metric, baseline in baselines.items():
            if metric in current_performance:
                current = current_performance[metric]
                regression_ratio = current / baseline
                
                # Allow 20% degradation from baseline
                assert regression_ratio < 1.2, f"Performance regression in {metric}: {regression_ratio:.2f}x slower"
                
                if regression_ratio > 1.1:
                    print(f"Warning: {metric} is {regression_ratio:.2f}x slower than baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])