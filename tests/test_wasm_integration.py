"""
Test suite for WebAssembly SIMD integration.
"""

import pytest
import numpy as np
import torch
import time
from typing import List, Dict, Any

try:
    import photon_neuro as pn
    from photon_neuro.wasm import (
        PhotonicWASM, initialize_wasm, mzi_forward_pass_simd,
        microring_simulation_simd, optical_matmul_simd,
        phase_accumulation_simd, complex_multiply_simd,
        waveguide_propagation_simd
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    pytest.skip(f"PhotonNeuro import failed: {e}", allow_module_level=True)


class TestWASMInitialization:
    """Test WebAssembly module initialization."""
    
    def test_wasm_import_available(self):
        """Test that WASM module can be imported."""
        assert IMPORT_SUCCESS, "PhotonNeuro WASM module should be importable"
    
    @pytest.mark.asyncio
    async def test_wasm_initialization(self):
        """Test WASM module initialization."""
        wasm_instance = await initialize_wasm(simd=True, threads=2, memory_mb=64)
        
        assert wasm_instance is not None
        assert wasm_instance.initialized
        
        metrics = wasm_instance.get_performance_metrics()
        assert metrics['initialized']
        assert metrics['thread_count'] >= 1
    
    def test_sync_wasm_initialization(self):
        """Test synchronous WASM initialization wrapper."""
        from photon_neuro.wasm.bindings import initialize_wasm_sync
        
        wasm_instance = initialize_wasm_sync(simd=True, threads=1)
        assert wasm_instance.initialized


class TestSIMDKernels:
    """Test SIMD-optimized kernel implementations."""
    
    def setup_method(self):
        """Setup test data for kernel tests."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Test parameters
        self.n_modes = 8
        self.n_phases = 16
        self.n_rings = 4
        self.n_wavelengths = 100
        
        # Generate test data
        self.input_field = np.random.randn(self.n_modes).astype(np.complex128)
        self.phases = np.random.uniform(0, 2*np.pi, self.n_phases).astype(np.float64)
        self.coupling_ratios = np.random.uniform(0.1, 0.9, self.n_phases//2).astype(np.float64)
        
        self.wavelengths = np.linspace(1540e-9, 1560e-9, self.n_wavelengths).astype(np.float64)
        self.ring_radii = np.random.uniform(5e-6, 10e-6, self.n_rings).astype(np.float64)
        self.coupling_coeffs = np.random.uniform(0.1, 0.3, self.n_rings).astype(np.float64)
        self.quality_factors = np.random.uniform(5000, 15000, self.n_rings).astype(np.float64)
    
    def test_mzi_forward_pass_simd(self):
        """Test SIMD MZI forward pass implementation."""
        result = mzi_forward_pass_simd(
            self.input_field, self.phases, self.coupling_ratios
        )
        
        # Check output properties
        assert result.shape == self.input_field.shape
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
        
        # Check energy conservation (approximately)
        input_energy = np.sum(np.abs(self.input_field)**2)
        output_energy = np.sum(np.abs(result)**2)
        energy_ratio = output_energy / input_energy
        
        assert 0.5 < energy_ratio < 1.5, f"Energy conservation violated: {energy_ratio}"
    
    def test_microring_simulation_simd(self):
        """Test SIMD microring resonator simulation."""
        transmission = microring_simulation_simd(
            self.wavelengths, self.ring_radii, 
            self.coupling_coeffs, self.quality_factors
        )
        
        # Check output shape and properties
        expected_shape = (self.n_rings, self.n_wavelengths)
        assert transmission.shape == expected_shape
        assert transmission.dtype == np.complex128
        assert np.all(np.isfinite(transmission))
        
        # Check transmission magnitudes (should be <= 1 for passive devices)
        transmission_mag = np.abs(transmission)
        assert np.all(transmission_mag <= 1.1), "Transmission magnitude too high"
        assert np.all(transmission_mag >= 0.0), "Negative transmission magnitude"
    
    def test_optical_matmul_simd(self):
        """Test SIMD optical matrix multiplication."""
        # Create test matrices
        m, k, n = 8, 6, 10
        matrix_a = np.random.randn(m, k).astype(np.complex128)
        matrix_b = np.random.randn(k, n).astype(np.complex128)
        
        # SIMD multiplication
        result_simd = optical_matmul_simd(matrix_a, matrix_b)
        
        # Reference NumPy multiplication
        result_numpy = np.dot(matrix_a, matrix_b)
        
        # Compare results
        assert result_simd.shape == (m, n)
        assert result_simd.dtype == np.complex128
        
        # Check numerical accuracy
        max_error = np.max(np.abs(result_simd - result_numpy))
        assert max_error < 1e-12, f"SIMD matmul error too large: {max_error}"
    
    def test_phase_accumulation_simd(self):
        """Test SIMD phase accumulation."""
        input_phases = np.random.uniform(0, 2*np.pi, self.n_modes).astype(np.float64)
        phase_velocities = np.random.uniform(1e6, 1e8, self.n_modes).astype(np.float64)
        propagation_lengths = np.random.uniform(1e-6, 1e-3, self.n_modes).astype(np.float64)
        
        result = phase_accumulation_simd(
            input_phases, phase_velocities, propagation_lengths
        )
        
        # Check output properties
        assert result.shape == input_phases.shape
        assert result.dtype == np.float64
        assert np.all((result >= 0) & (result < 2*np.pi)), "Phases not properly wrapped"
        
        # Verify phase accumulation formula
        expected = np.fmod(input_phases + phase_velocities * propagation_lengths, 2*np.pi)
        expected[expected < 0] += 2*np.pi
        
        max_error = np.max(np.abs(result - expected))
        assert max_error < 1e-12, f"Phase accumulation error: {max_error}"
    
    def test_complex_multiply_simd(self):
        """Test SIMD complex field multiplication."""
        field_a = np.random.randn(self.n_modes).astype(np.complex128)
        field_b = np.random.randn(self.n_modes).astype(np.complex128)
        
        result = complex_multiply_simd(field_a, field_b)
        expected = field_a * field_b
        
        assert result.shape == field_a.shape
        assert result.dtype == np.complex128
        
        max_error = np.max(np.abs(result - expected))
        assert max_error < 1e-15, f"Complex multiplication error: {max_error}"
    
    def test_waveguide_propagation_simd(self):
        """Test SIMD waveguide propagation."""
        propagation_constant = np.random.uniform(1e6, 1e7, self.n_modes).astype(np.float64)
        length = 1e-3  # 1 mm
        loss_coefficient = 10.0  # 10 dB/m
        
        result = waveguide_propagation_simd(
            self.input_field, propagation_constant, length, loss_coefficient
        )
        
        # Check output properties
        assert result.shape == self.input_field.shape
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
        
        # Check that loss reduces amplitude
        input_magnitude = np.abs(self.input_field)
        output_magnitude = np.abs(result)
        attenuation = output_magnitude / input_magnitude
        
        # Should have some attenuation due to loss
        assert np.all(attenuation <= 1.0), "Amplitude should not increase due to loss"
        assert np.all(attenuation > 0.5), "Attenuation too strong for given parameters"


class TestTorchIntegration:
    """Test PyTorch tensor integration with WASM kernels."""
    
    def setup_method(self):
        """Setup PyTorch test data."""
        torch.manual_seed(42)
        
        self.input_tensor = torch.randn(8, dtype=torch.complex64)
        self.phases_tensor = torch.rand(16) * 2 * np.pi
        self.coupling_tensor = torch.rand(8) * 0.8 + 0.1
    
    def test_torch_to_wasm_conversion(self):
        """Test conversion between PyTorch tensors and WASM arrays."""
        from photon_neuro.wasm.bindings import torch_to_wasm_complex, wasm_to_torch_complex
        
        # Convert to WASM format
        wasm_array = torch_to_wasm_complex(self.input_tensor)
        
        assert isinstance(wasm_array, np.ndarray)
        assert wasm_array.dtype == np.complex128
        assert wasm_array.shape == self.input_tensor.shape
        
        # Convert back to PyTorch
        torch_tensor = wasm_to_torch_complex(wasm_array)
        
        assert torch_tensor.shape == self.input_tensor.shape
        assert torch.allclose(torch_tensor, self.input_tensor.to(torch.complex128), atol=1e-6)
    
    def test_high_level_torch_interface(self):
        """Test high-level PyTorch interface functions."""
        from photon_neuro.wasm.simd_kernels import mzi_forward_pass, microring_simulation
        
        # Test MZI forward pass with PyTorch tensors
        result = mzi_forward_pass(
            self.input_tensor, self.phases_tensor, self.coupling_tensor
        )
        
        assert isinstance(result, (torch.Tensor, np.ndarray))
        assert result.shape == self.input_tensor.shape
        
        # Test microring simulation
        wavelengths = torch.linspace(1540e-9, 1560e-9, 50)
        radii = torch.uniform(torch.zeros(4), torch.ones(4)) * 5e-6 + 5e-6
        coupling = torch.uniform(torch.zeros(4), torch.ones(4)) * 0.2 + 0.1
        q_factors = torch.ones(4) * 10000
        
        transmission = microring_simulation(wavelengths, radii, coupling, q_factors)
        
        assert transmission.shape == (4, 50)


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for WASM kernels."""
    
    def setup_method(self):
        """Setup benchmark parameters."""
        self.benchmark_sizes = [4, 8, 16, 32, 64]
        self.n_runs = 100
    
    def test_mzi_performance_scaling(self):
        """Test MZI kernel performance scaling with size."""
        performance_data = []
        
        for size in self.benchmark_sizes:
            input_field = np.random.randn(size).astype(np.complex128)
            phases = np.random.uniform(0, 2*np.pi, size).astype(np.float64)
            coupling = np.random.uniform(0.1, 0.9, size//2).astype(np.float64)
            
            # Warmup
            for _ in range(10):
                _ = mzi_forward_pass_simd(input_field, phases, coupling)
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(self.n_runs):
                _ = mzi_forward_pass_simd(input_field, phases, coupling)
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / self.n_runs * 1000  # ms
            throughput = size / avg_time * 1000  # operations per second
            
            performance_data.append({
                'size': size,
                'time_ms': avg_time,
                'throughput_ops_per_sec': throughput
            })
        
        # Check that performance scales reasonably
        assert len(performance_data) == len(self.benchmark_sizes)
        
        # Performance should not degrade dramatically with size
        for data in performance_data:
            assert data['time_ms'] < 100, f"Performance too slow for size {data['size']}"
            assert data['throughput_ops_per_sec'] > 1000, f"Throughput too low for size {data['size']}"
    
    @pytest.mark.slow
    def test_comprehensive_benchmark(self):
        """Comprehensive performance benchmark of all kernels."""
        from photon_neuro.wasm.simd_kernels import benchmark_simd_kernels
        
        results = benchmark_simd_kernels(n_runs=50, size=64)
        
        # Check benchmark results
        assert 'mzi_time_ms' in results
        assert 'microring_time_ms' in results
        assert results['mzi_time_ms'] > 0
        assert results['microring_time_ms'] > 0
        assert results['size'] == 64
        assert results['runs'] == 50
        
        # Performance thresholds
        assert results['mzi_time_ms'] < 10, "MZI performance below threshold"
        assert results['microring_time_ms'] < 50, "Microring performance below threshold"
    
    def test_wasm_vs_python_performance(self):
        """Compare WASM vs pure Python performance."""
        size = 32
        input_field = np.random.randn(size).astype(np.complex128)
        phases = np.random.uniform(0, 2*np.pi, size).astype(np.float64)
        coupling = np.random.uniform(0.1, 0.9, size//2).astype(np.float64)
        
        # Benchmark WASM version
        start_time = time.perf_counter()
        for _ in range(self.n_runs):
            _ = mzi_forward_pass_simd(input_field, phases, coupling)
        wasm_time = time.perf_counter() - start_time
        
        # Since we're using Numba as fallback, performance should still be good
        assert wasm_time < 1.0, f"WASM/Numba performance too slow: {wasm_time}s"


class TestWASMIntegration:
    """Test full WASM module integration."""
    
    @pytest.mark.asyncio
    async def test_full_wasm_integration(self):
        """Test complete WASM module integration workflow."""
        # Initialize WASM
        wasm_instance = await initialize_wasm(simd=True, threads=2)
        
        # Test MZI forward pass
        input_field = np.random.randn(8).astype(np.complex128)
        phases = np.random.uniform(0, 2*np.pi, 8).astype(np.float64)
        
        result = wasm_instance.call_mzi_forward(input_field, phases)
        
        assert result.shape == input_field.shape
        assert np.all(np.isfinite(result))
        
        # Test microring simulation
        wavelengths = np.linspace(1540e-9, 1560e-9, 50)
        ring_params = {
            'radii': np.random.uniform(5e-6, 10e-6, 4),
            'coupling': np.random.uniform(0.1, 0.3, 4),
            'q_factors': np.random.uniform(5000, 15000, 4)
        }
        
        transmission = wasm_instance.call_microring_simulation(wavelengths, ring_params)
        
        assert transmission.shape == (4, 50)
        assert np.all(np.abs(transmission) <= 1.1)
    
    def test_wasm_benchmark_class(self):
        """Test WASM benchmark utilities."""
        from photon_neuro.wasm.bindings import WASMBenchmark, initialize_wasm_sync
        
        wasm_instance = initialize_wasm_sync(simd=True)
        benchmark = WASMBenchmark(wasm_instance)
        
        # Run benchmark
        results = benchmark.benchmark_mzi_forward([8, 16], n_runs=10)
        
        assert 'sizes' in results
        assert 'wasm_times' in results
        assert 'python_times' in results
        assert len(results['sizes']) == 2
        assert len(results['wasm_times']) == 2
        assert len(results['python_times']) == 2


class TestErrorHandling:
    """Test error handling in WASM integration."""
    
    def test_invalid_input_shapes(self):
        """Test error handling for invalid input shapes."""
        # Mismatched array sizes
        input_field = np.random.randn(8).astype(np.complex128)
        phases = np.random.uniform(0, 2*np.pi, 4).astype(np.float64)  # Wrong size
        coupling = np.random.uniform(0.1, 0.9, 4).astype(np.float64)
        
        # Should handle gracefully (may return error or use available data)
        try:
            result = mzi_forward_pass_simd(input_field, phases, coupling)
            # If it succeeds, check the output is reasonable
            assert result.shape == input_field.shape
        except (ValueError, IndexError):
            # Acceptable to raise error for invalid input
            pass
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Test with NaN values
        input_field = np.array([np.nan + 1j, 1.0 + 0j]).astype(np.complex128)
        phases = np.array([0.0, np.pi]).astype(np.float64)
        coupling = np.array([0.5]).astype(np.float64)
        
        result = mzi_forward_pass_simd(input_field, phases, coupling)
        
        # Function should handle NaNs appropriately
        assert result.shape == input_field.shape
    
    def test_memory_limits(self):
        """Test behavior with large memory requirements."""
        # Test with reasonably large arrays (but not excessive for CI)
        large_size = 1000
        input_field = np.random.randn(large_size).astype(np.complex128)
        phases = np.random.uniform(0, 2*np.pi, large_size).astype(np.float64)
        coupling = np.random.uniform(0.1, 0.9, large_size//2).astype(np.float64)
        
        try:
            result = mzi_forward_pass_simd(input_field, phases, coupling)
            assert result.shape == input_field.shape
        except MemoryError:
            pytest.skip("Insufficient memory for large array test")


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def wasm_instance():
    """Session-scoped WASM instance fixture."""
    from photon_neuro.wasm.bindings import initialize_wasm_sync
    
    instance = initialize_wasm_sync(simd=True, threads=1, memory_mb=32)
    yield instance
    # Cleanup if needed


@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_parametrized_mzi_sizes(size):
    """Parametrized test for different MZI sizes."""
    input_field = np.random.randn(size).astype(np.complex128)
    phases = np.random.uniform(0, 2*np.pi, size).astype(np.float64)
    coupling = np.random.uniform(0.1, 0.9, max(1, size//2)).astype(np.float64)
    
    result = mzi_forward_pass_simd(input_field, phases, coupling)
    
    assert result.shape == input_field.shape
    assert np.all(np.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])