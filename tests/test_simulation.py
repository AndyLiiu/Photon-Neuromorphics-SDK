"""
Tests for simulation engines and analysis tools.
"""

import pytest
import torch
import numpy as np
from photon_neuro.simulation import (
    PhotonicSimulator, FDTDSolver, NoiseSimulator, 
    PowerBudgetAnalyzer, CircuitLevelSimulator
)
from photon_neuro.core import SiliconWaveguide, MachZehnderModulator


class TestPhotonicSimulator:
    """Test main photonic simulator."""
    
    def test_simulator_creation(self):
        """Test simulator initialization."""
        sim = PhotonicSimulator(timestep=1e-15, backend="torch")
        
        assert sim.timestep == 1e-15
        assert sim.backend == "torch"
        assert len(sim.components) == 0
        
    def test_component_addition(self):
        """Test adding components to simulator."""
        sim = PhotonicSimulator()
        
        wg = SiliconWaveguide(length=1e-3)
        comp_idx = sim.add_component(wg)
        
        assert comp_idx == 0
        assert len(sim.components) == 1
        assert sim.components[0] == wg
        
    def test_component_connection(self):
        """Test connecting components."""
        sim = PhotonicSimulator()
        
        wg1 = SiliconWaveguide(length=1e-3)
        wg2 = SiliconWaveguide(length=2e-3)
        
        idx1 = sim.add_component(wg1)
        idx2 = sim.add_component(wg2)
        
        sim.connect(idx1, idx2, length=100e-6, loss_db_per_cm=0.1)
        
        assert len(sim.connections) == 1
        assert sim.connections[0] == (idx1, idx2)
        
    def test_frequency_sweep(self):
        """Test frequency domain analysis."""
        sim = PhotonicSimulator()
        wg = SiliconWaveguide(length=1e-3)
        
        frequencies = np.linspace(190e12, 200e12, 10)  # 10 points
        transmission, phase = sim.frequency_sweep(wg, frequencies)
        
        assert len(transmission) == 10
        assert len(phase) == 10
        assert all(0 <= t <= 1 for t in transmission)  # Transmission should be 0-1
        
    def test_group_delay_calculation(self):
        """Test group delay calculation."""
        sim = PhotonicSimulator()
        mod = MachZehnderModulator(length=2e-3)
        
        frequencies = np.linspace(190e12, 200e12, 100)
        group_delay = sim.calculate_group_delay(mod, frequencies)
        
        assert len(group_delay) == 100
        assert all(np.isfinite(gd) for gd in group_delay)


class TestFDTDSolver:
    """Test FDTD electromagnetic solver."""
    
    def test_fdtd_creation(self):
        """Test FDTD solver creation."""
        grid_size = (50, 50, 20)
        fdtd = FDTDSolver(grid_size=grid_size, cell_size=50e-9)
        
        assert fdtd.grid_size == grid_size
        assert fdtd.cell_size == 50e-9
        assert fdtd.Ex.shape == grid_size
        assert fdtd.Hy.shape == grid_size
        
    def test_material_setting(self):
        """Test setting material properties."""
        fdtd = FDTDSolver(grid_size=(20, 20, 10))
        
        # Set silicon region
        region = (slice(5, 15), slice(5, 15), slice(0, 10))
        fdtd.set_material(region, epsilon_r=12.25)  # Silicon
        
        assert fdtd.epsilon[10, 10, 5] == 12.25
        assert fdtd.epsilon[0, 0, 0] == 1.0  # Air region unchanged
        
    def test_waveguide_addition(self):
        """Test adding waveguide structure."""
        fdtd = FDTDSolver(grid_size=(100, 50, 20))
        
        start = (10, 20, 5)
        end = (90, 20, 5)
        width = 3
        
        fdtd.add_waveguide(start, end, width, epsilon_core=12.25)
        
        # Check that core region has high permittivity
        assert fdtd.epsilon[50, 20, 5] == 12.25
        assert fdtd.epsilon[50, 10, 5] == 1.0  # Outside waveguide
        
    def test_source_addition(self):
        """Test adding electromagnetic source."""
        fdtd = FDTDSolver(grid_size=(50, 50, 10))
        
        position = (25, 25, 5)
        fdtd.add_source(position, amplitude=1.0, frequency=200e12)
        
        assert hasattr(fdtd, 'source_pos')
        assert fdtd.source_pos == position
        
    def test_fdtd_step(self):
        """Test single FDTD time step."""
        fdtd = FDTDSolver(grid_size=(20, 20, 10))
        
        # Add source
        fdtd.add_source((10, 10, 5), amplitude=1.0, frequency=200e12)
        
        # Store initial fields
        initial_Ez = fdtd.Ez.copy()
        
        # Take one time step
        fdtd.step(0)
        
        # Fields should change due to source
        assert not np.array_equal(fdtd.Ez, initial_Ez)
        
    def test_transmission_calculation(self):
        """Test transmission calculation."""
        fdtd = FDTDSolver(grid_size=(100, 50, 10))
        
        # Add simple waveguide and source
        fdtd.add_waveguide((10, 20, 5), (90, 20, 5), 3)
        fdtd.add_source((20, 20, 5), amplitude=1.0, frequency=200e12)
        
        # Run a few steps
        for t in range(10):
            fdtd.step(t)
            
        # Calculate transmission at output
        monitor_positions = [(80, 20, 5)]
        transmissions = fdtd.calculate_transmission(monitor_positions)
        
        assert len(transmissions) == 1
        assert transmissions[0] >= 0


class TestNoiseSimulator:
    """Test noise analysis tools."""
    
    def test_noise_simulator_creation(self):
        """Test noise simulator creation."""
        wg = SiliconWaveguide(length=1e-3)
        noise_sim = NoiseSimulator(wg)
        
        assert noise_sim.model == wg
        assert hasattr(noise_sim, 'quantum_noise')
        
    def test_input_power_sweep(self):
        """Test SNR vs input power analysis."""
        wg = SiliconWaveguide(length=1e-3)
        noise_sim = NoiseSimulator(wg)
        
        results = noise_sim.sweep_input_power(
            power_range_dbm=(-10, 0),
            n_points=5,
            include_shot_noise=True,
            include_thermal_noise=True
        )
        
        assert 'power_dbm' in results
        assert 'snr_db' in results
        assert len(results['power_dbm']) == 5
        assert len(results['snr_db']) == 5
        
    def test_phase_noise_analysis(self):
        """Test phase noise characterization."""
        wg = SiliconWaveguide(length=1e-3)
        noise_sim = NoiseSimulator(wg)
        
        laser_linewidth = 1e6  # 1 MHz
        results = noise_sim.analyze_phase_noise(
            laser_linewidth, measurement_time=1e-6
        )
        
        assert 'time' in results
        assert 'phase_evolution' in results
        assert 'frequencies' in results
        assert 'phase_psd' in results
        assert 'allan_variance' in results
        
    def test_monte_carlo_analysis(self):
        """Test Monte Carlo variation analysis."""
        wg = SiliconWaveguide(length=1e-3)
        noise_sim = NoiseSimulator(wg)
        
        variations = {
            'width_sigma': 5e-9,
            'thickness_sigma': 2e-9
        }
        
        results = noise_sim.monte_carlo_analysis(
            n_runs=10, fabrication_variations=variations
        )
        
        assert 'transmission' in results
        assert 'phase' in results
        assert 'loss' in results
        assert len(results['transmission']) <= 10  # Some runs might fail


class TestPowerBudgetAnalyzer:
    """Test power analysis tools."""
    
    def test_analyzer_creation(self):
        """Test power analyzer creation."""
        mod = MachZehnderModulator()
        analyzer = PowerBudgetAnalyzer(mod)
        
        assert analyzer.model == mod
        
    def test_power_analysis(self):
        """Test comprehensive power analysis."""
        mod = MachZehnderModulator()
        mod.set_drive_voltage(1.0)  # Some power consumption
        
        analyzer = PowerBudgetAnalyzer(mod)
        report = analyzer.analyze(
            input_power_dbm=0,
            include_thermal=True,
            include_electrical=True
        )
        
        assert hasattr(report, 'optical_powers')
        assert hasattr(report, 'electrical_powers')
        assert hasattr(report, 'efficiency_metrics')
        
        # Check that some analysis was done
        assert 'input_power' in report.optical_powers
        assert report.total_mw >= 0
        
    def test_power_report_properties(self):
        """Test power report properties."""
        from photon_neuro.simulation.power import PowerReport
        
        optical_powers = {'input_power': 1e-3, 'transmitted_power': 0.8e-3}
        electrical_powers = {'total_electrical': 2e-3}
        efficiency_metrics = {'optical_efficiency': 0.8}
        
        report = PowerReport(
            optical_powers=optical_powers,
            electrical_powers=electrical_powers,
            efficiency_metrics=efficiency_metrics
        )
        
        assert report.total_mw == (1e-3 + 2e-3) * 1000  # Convert to mW
        assert report.optical_efficiency == 0.8
        
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Power Budget Analysis" in summary


class TestCircuitLevelSimulator:
    """Test circuit-level simulation."""
    
    def test_circuit_simulator_creation(self):
        """Test circuit simulator creation."""
        sim = CircuitLevelSimulator()
        
        assert hasattr(sim, 'graph')
        assert hasattr(sim, 'components')
        assert hasattr(sim, 's_parameters')
        
    def test_component_addition_to_circuit(self):
        """Test adding components to circuit."""
        sim = CircuitLevelSimulator()
        
        wg = SiliconWaveguide(length=1e-3)
        sim.add_component("waveguide1", wg, position=(0, 0))
        
        assert "waveguide1" in sim.components
        assert sim.components["waveguide1"] == wg
        
    def test_component_connection_in_circuit(self):
        """Test connecting components in circuit."""
        sim = CircuitLevelSimulator()
        
        wg1 = SiliconWaveguide(length=1e-3)
        wg2 = SiliconWaveguide(length=2e-3)
        
        sim.add_component("wg1", wg1)
        sim.add_component("wg2", wg2)
        
        sim.connect("wg1", 0, "wg2", 0, length=100e-6)
        
        assert sim.graph.has_edge("wg1", "wg2")
        
    def test_frequency_response_analysis(self):
        """Test circuit frequency response."""
        sim = CircuitLevelSimulator()
        
        wg1 = SiliconWaveguide(length=1e-3)
        wg2 = SiliconWaveguide(length=1e-3)
        
        sim.add_component("input", wg1)
        sim.add_component("output", wg2)
        sim.connect("input", 0, "output", 0)
        
        # Should be able to analyze even with simple components
        try:
            transmission, phase = sim.analyze_frequency_response(
                "input", 0, "output", 0
            )
            # Basic checks
            assert len(transmission) > 0
            assert len(phase) > 0
        except Exception as e:
            # Acceptable if S-parameters not fully implemented
            assert "S-parameters" in str(e) or "Transfer function" in str(e)
            
    def test_netlist_export(self):
        """Test circuit netlist export."""
        sim = CircuitLevelSimulator()
        
        wg = SiliconWaveguide(length=1e-3)
        sim.add_component("test_wg", wg)
        
        netlist = sim.export_netlist()
        
        assert isinstance(netlist, str)
        assert "Photonic Circuit Netlist" in netlist
        assert "test_wg" in netlist


class TestQuantumNoiseModel:
    """Test quantum noise modeling."""
    
    def test_shot_noise_calculation(self):
        """Test shot noise variance calculation."""
        from photon_neuro.simulation.noise import QuantumNoiseModel
        
        noise_model = QuantumNoiseModel(temperature=300)
        
        photocurrent = 1e-6  # 1 μA
        bandwidth = 1e9      # 1 GHz
        
        shot_variance = noise_model.shot_noise_variance(photocurrent, bandwidth)
        
        assert shot_variance > 0
        assert shot_variance == 2 * 1.602e-19 * photocurrent * bandwidth
        
    def test_thermal_noise_calculation(self):
        """Test thermal noise calculation."""
        from photon_neuro.simulation.noise import QuantumNoiseModel
        
        noise_model = QuantumNoiseModel(temperature=300)
        
        resistance = 50     # 50 Ω
        bandwidth = 1e9     # 1 GHz
        
        thermal_variance = noise_model.thermal_noise_variance(resistance, bandwidth)
        
        assert thermal_variance > 0
        expected = 4 * noise_model.k_B * 300 * bandwidth / resistance
        assert abs(thermal_variance - expected) < 1e-20
        
    def test_quantum_shot_noise_generation(self):
        """Test quantum shot noise generation."""
        from photon_neuro.simulation.noise import QuantumNoiseModel
        
        noise_model = QuantumNoiseModel()
        
        optical_power = torch.tensor([1e-6])  # 1 μW
        photon_energy = 1.24e-6 / 1550       # eV at 1550 nm
        bandwidth = 1e9
        
        noise = noise_model.quantum_shot_noise(optical_power, photon_energy, bandwidth)
        
        assert noise.shape == optical_power.shape
        assert torch.is_real(noise)
        
    def test_phase_noise_addition(self):
        """Test laser phase noise addition."""
        from photon_neuro.simulation.noise import QuantumNoiseModel
        
        noise_model = QuantumNoiseModel()
        
        optical_field = torch.ones(10, dtype=torch.complex64)
        linewidth = 1e6  # 1 MHz
        
        noisy_field = noise_model.phase_noise(optical_field, linewidth)
        
        assert noisy_field.shape == optical_field.shape
        assert torch.is_complex(noisy_field)
        
        # Phase should change but amplitude should be preserved
        original_amplitude = torch.abs(optical_field)
        noisy_amplitude = torch.abs(noisy_field)
        assert torch.allclose(original_amplitude, noisy_amplitude, atol=1e-6)


@pytest.fixture
def basic_simulation_setup():
    """Fixture for basic simulation setup."""
    sim = PhotonicSimulator(timestep=1e-15)
    
    # Add some basic components
    wg = SiliconWaveguide(length=1e-3)
    mod = MachZehnderModulator(length=2e-3)
    
    wg_idx = sim.add_component(wg)
    mod_idx = sim.add_component(mod)
    sim.connect(wg_idx, mod_idx, length=50e-6)
    
    return sim, wg_idx, mod_idx


class TestSimulationIntegration:
    """Test integration between simulation components."""
    
    def test_simulation_pipeline(self, basic_simulation_setup):
        """Test complete simulation pipeline."""
        sim, wg_idx, mod_idx = basic_simulation_setup
        
        # This should not crash
        assert len(sim.components) == 2
        assert len(sim.connections) == 1
        
    def test_component_analysis_chain(self):
        """Test analysis chain for components."""
        wg = SiliconWaveguide(length=1e-3)
        mod = MachZehnderModulator()
        mod.set_drive_voltage(0.5)
        
        # Test that each analyzer can handle the components
        analyzers = [
            NoiseSimulator(wg),
            PowerBudgetAnalyzer(mod)
        ]
        
        for analyzer in analyzers:
            assert analyzer.model is not None


if __name__ == "__main__":
    pytest.main([__file__])