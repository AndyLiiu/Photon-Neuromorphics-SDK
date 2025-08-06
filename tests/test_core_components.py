"""
Tests for core photonic components.
"""

import pytest
import torch
import numpy as np
from photon_neuro.core import (
    PhotonicComponent, WaveguideBase, ModulatorBase, DetectorBase,
    SiliconWaveguide, MachZehnderModulator, Photodetector
)


class TestPhotonicComponent:
    """Test base PhotonicComponent class."""
    
    def test_component_creation(self):
        """Test basic component creation."""
        # Create a simple component for testing
        class TestComponent(PhotonicComponent):
            def forward(self, input_field):
                return input_field
                
            def to_netlist(self):
                return {"type": "test_component"}
                
        component = TestComponent()
        assert component.name == "TestComponent"
        
    def test_loss_addition(self):
        """Test adding insertion loss."""
        class TestComponent(PhotonicComponent):
            def forward(self, input_field):
                return input_field * self.get_loss_linear()
                
            def to_netlist(self):
                return {"type": "test_component"}
                
        component = TestComponent()
        component.add_loss(3.0)  # 3 dB loss
        
        loss_linear = component.get_loss_linear()
        expected = 10**(-3.0/20)  # -3dB to linear
        assert abs(loss_linear - expected) < 1e-6
        
    def test_power_consumption(self):
        """Test power consumption calculation."""
        class TestComponent(PhotonicComponent):
            def forward(self, input_field):
                return input_field
                
            def to_netlist(self):
                return {"type": "test_component"}
                
            def calculate_power_consumption(self):
                return 0.001  # 1 mW
                
        component = TestComponent()
        assert component.calculate_power_consumption() == 0.001


class TestWaveguide:
    """Test waveguide components."""
    
    def test_silicon_waveguide(self):
        """Test silicon waveguide creation and propagation."""
        wg = SiliconWaveguide(length=1e-3, width=450e-9)  # 1 mm long, 450 nm wide
        
        assert wg.length == 1e-3
        assert wg.width == 450e-9
        assert wg.material == "silicon"
        assert wg.n_eff > 2.0  # Silicon should have high effective index
        
    def test_waveguide_propagation(self):
        """Test field propagation through waveguide."""
        wg = SiliconWaveguide(length=1e-3)
        
        input_field = torch.ones(1, dtype=torch.complex64)
        output_field = wg.forward(input_field)
        
        # Output should be complex (phase shift) and attenuated (loss)
        assert torch.is_complex(output_field)
        assert torch.abs(output_field) < torch.abs(input_field)  # Some loss
        
    def test_waveguide_netlist(self):
        """Test waveguide netlist generation."""
        wg = SiliconWaveguide(length=2e-3, width=500e-9)
        netlist = wg.to_netlist()
        
        assert netlist["type"] == "waveguide"
        assert netlist["material"] == "silicon"
        assert netlist["length"] == 2e-3
        assert netlist["width"] == 500e-9


class TestModulator:
    """Test modulator components."""
    
    def test_mzi_modulator(self):
        """Test Mach-Zehnder modulator."""
        modulator = MachZehnderModulator(length=2e-3, v_pi=1.5)
        
        assert modulator.length == 2e-3
        assert modulator.v_pi == 1.5
        assert modulator.modulation_type == "amplitude"
        
    def test_modulation_response(self):
        """Test modulation response."""
        modulator = MachZehnderModulator(v_pi=2.0)
        
        # Test different drive voltages
        modulator.set_drive_voltage(0.0)  # No modulation
        response_0 = modulator.get_modulation_response()
        
        modulator.set_drive_voltage(2.0)  # π phase shift
        response_pi = modulator.get_modulation_response()
        
        assert abs(response_0) < 1e-6  # Should be near zero
        assert abs(response_pi - np.pi) < 1e-6  # Should be π
        
    def test_modulator_forward(self):
        """Test modulator forward pass."""
        modulator = MachZehnderModulator()
        modulator.set_drive_voltage(0.5)  # Some modulation
        
        input_field = torch.ones(1, dtype=torch.complex64)
        output_field = modulator.forward(input_field)
        
        assert torch.is_complex(output_field)
        assert output_field.shape == input_field.shape
        
    def test_power_calculation(self):
        """Test modulator power consumption."""
        modulator = MachZehnderModulator()
        modulator.set_drive_voltage(1.0)
        
        power = modulator.calculate_power_consumption()
        assert power >= 0  # Power should be non-negative


class TestDetector:
    """Test photodetector components."""
    
    def test_photodetector_creation(self):
        """Test photodetector creation."""
        detector = Photodetector(responsivity=1.2, dark_current=1e-9, bandwidth=10e9)
        
        assert detector.responsivity == 1.2
        assert detector.dark_current == 1e-9
        assert detector.bandwidth == 10e9
        
    def test_detection_process(self):
        """Test optical to electrical conversion."""
        detector = Photodetector(responsivity=1.0, dark_current=0.0)
        
        # Test with optical power
        optical_field = torch.ones(1, dtype=torch.complex64) * np.sqrt(1e-3)  # 1 mW
        photocurrent = detector.forward(optical_field)
        
        expected_current = 1e-3 * 1.0  # Power * responsivity
        assert abs(photocurrent.item() - expected_current) < 1e-6
        
    def test_noise_in_training(self):
        """Test noise addition during training."""
        detector = Photodetector(responsivity=1.0)
        detector.train()  # Enable training mode
        
        optical_field = torch.ones(10, dtype=torch.complex64) * 0.1
        photocurrent = detector.forward(optical_field)
        
        # In training mode, noise should be added
        # Check that outputs are not all identical (due to noise)
        assert not torch.allclose(photocurrent, photocurrent[0].expand_as(photocurrent))
        
    def test_snr_calculation(self):
        """Test SNR calculation."""
        detector = Photodetector(responsivity=1.0, dark_current=1e-9, bandwidth=1e9)
        
        optical_power = 1e-6  # 1 μW
        snr = detector.calculate_snr(optical_power)
        
        assert snr > 0  # SNR should be positive
        assert not np.isinf(snr)  # SNR should be finite


class TestComponentRegistry:
    """Test component registration system."""
    
    def test_component_registration(self):
        """Test registering custom components."""
        from photon_neuro.core.registry import register_component, ComponentRegistry
        
        @register_component("custom_test")
        class CustomComponent(PhotonicComponent):
            def forward(self, input_field):
                return input_field * 0.5
                
            def to_netlist(self):
                return {"type": "custom_test"}
                
        # Check if component was registered
        registry = ComponentRegistry()
        assert "custom_test" in registry.list_components()
        
        # Test creating component from registry
        component = registry.create("custom_test")
        assert isinstance(component, CustomComponent)
        
    def test_registry_validation(self):
        """Test registry validation of components."""
        from photon_neuro.core.registry import ComponentRegistry
        
        registry = ComponentRegistry()
        
        # Try to register invalid component (missing methods)
        class InvalidComponent:
            pass
            
        with pytest.raises(ValueError):
            registry.register("invalid", InvalidComponent)


@pytest.fixture
def sample_components():
    """Fixture providing sample components for testing."""
    return {
        'waveguide': SiliconWaveguide(length=1e-3),
        'modulator': MachZehnderModulator(),
        'detector': Photodetector()
    }


class TestComponentIntegration:
    """Test integration between components."""
    
    def test_cascade_components(self, sample_components):
        """Test cascading multiple components."""
        wg = sample_components['waveguide']
        mod = sample_components['modulator']
        det = sample_components['detector']
        
        # Set up modulator
        mod.set_drive_voltage(0.5)
        
        # Create signal chain
        input_field = torch.ones(1, dtype=torch.complex64)
        
        # Propagate through components
        field_after_wg = wg.forward(input_field)
        field_after_mod = mod.forward(field_after_wg)
        photocurrent = det.forward(field_after_mod)
        
        assert torch.is_real(photocurrent)  # Final output should be real current
        assert photocurrent.item() >= 0  # Current should be non-negative
        
    def test_component_netlist_export(self, sample_components):
        """Test exporting component netlists."""
        for name, component in sample_components.items():
            netlist = component.to_netlist()
            
            assert isinstance(netlist, dict)
            assert "type" in netlist
            assert isinstance(netlist["type"], str)


if __name__ == "__main__":
    pytest.main([__file__])