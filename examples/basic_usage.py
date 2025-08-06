#!/usr/bin/env python3
"""
Basic Usage Examples for Photon Neuromorphics SDK
================================================

This script demonstrates basic usage patterns of the Photon Neuromorphics SDK,
including component creation, network simulation, and analysis.

Run with: python examples/basic_usage.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import the Photon Neuromorphics SDK
import photon_neuro as pn


def example_waveguide_propagation():
    """Demonstrate basic waveguide field propagation."""
    print("=== Waveguide Propagation Example ===")
    
    # Create a silicon waveguide
    waveguide = pn.SiliconWaveguide(
        length=2e-3,      # 2 mm long
        width=450e-9,     # 450 nm wide
        height=220e-9     # 220 nm thick (standard SOI)
    )
    
    print(f"Waveguide effective index: {waveguide.n_eff:.3f}")
    print(f"Waveguide loss: {waveguide.loss_db_per_cm:.2f} dB/cm")
    
    # Create input optical field
    input_power_mw = 1.0  # 1 mW input power
    input_field = torch.sqrt(torch.tensor(input_power_mw * 1e-3)) * torch.ones(1, dtype=torch.complex64)
    
    # Propagate through waveguide
    output_field = waveguide.forward(input_field)
    
    # Calculate results
    output_power_mw = torch.abs(output_field)**2 * 1000
    insertion_loss_db = -10 * torch.log10(output_power_mw / input_power_mw)
    phase_shift_rad = torch.angle(output_field)
    
    print(f"Output power: {output_power_mw.item():.3f} mW")
    print(f"Insertion loss: {insertion_loss_db.item():.3f} dB")
    print(f"Phase shift: {phase_shift_rad.item():.3f} rad")
    print()


def example_mzi_modulator():
    """Demonstrate Mach-Zehnder interferometer modulation."""
    print("=== MZI Modulator Example ===")
    
    # Create MZI modulator
    mzi = pn.MachZehnderModulator(
        length=2e-3,      # 2 mm interaction length
        v_pi=1.5          # 1.5V for π phase shift
    )
    
    print(f"Modulator Vπ: {mzi.v_pi} V")
    print(f"Modulator length: {mzi.length*1000:.1f} mm")
    
    # Test modulation at different voltages
    voltages = np.linspace(0, 3, 11)  # 0 to 3V
    transmissions = []
    
    input_field = torch.ones(1, dtype=torch.complex64)
    
    for voltage in voltages:
        mzi.set_drive_voltage(voltage)
        output_field = mzi.forward(input_field)
        transmission = torch.abs(output_field)**2
        transmissions.append(transmission.item())
    
    # Find extinction ratio
    max_transmission = max(transmissions)
    min_transmission = min(transmissions)
    extinction_ratio_db = 10 * np.log10(max_transmission / min_transmission)
    
    print(f"Maximum transmission: {max_transmission:.3f}")
    print(f"Minimum transmission: {min_transmission:.6f}")
    print(f"Extinction ratio: {extinction_ratio_db:.1f} dB")
    
    # Plot modulation curve
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, transmissions, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Drive Voltage (V)')
    plt.ylabel('Optical Transmission')
    plt.title('MZI Modulator Transfer Function')
    plt.grid(True, alpha=0.3)
    plt.savefig('mzi_modulation_curve.png', dpi=150, bbox_inches='tight')
    print("Modulation curve saved as 'mzi_modulation_curve.png'")
    print()


def example_photodetector():
    """Demonstrate photodetector operation with noise."""
    print("=== Photodetector Example ===")
    
    # Create photodetector
    detector = pn.Photodetector(
        responsivity=1.2,     # A/W at 1550nm
        dark_current=1e-9,    # 1 nA dark current
        bandwidth=10e9        # 10 GHz bandwidth
    )
    
    print(f"Detector responsivity: {detector.responsivity} A/W")
    print(f"Dark current: {detector.dark_current*1e9:.1f} nA")
    print(f"Bandwidth: {detector.bandwidth/1e9:.1f} GHz")
    
    # Test detection at different power levels
    power_levels_dbm = np.arange(-20, 5, 2.5)
    power_levels_w = 1e-3 * 10**(power_levels_dbm / 10)
    
    snr_values = []
    for power_w in power_levels_w:
        snr_db = detector.calculate_snr(power_w)
        snr_values.append(snr_db)
    
    # Plot SNR vs power
    plt.figure(figsize=(10, 6))
    plt.semilogx(power_levels_w * 1000, snr_values, 'r-o', linewidth=2, markersize=6)
    plt.xlabel('Optical Power (mW)')
    plt.ylabel('SNR (dB)')
    plt.title('Photodetector SNR vs Optical Power')
    plt.grid(True, alpha=0.3)
    plt.savefig('detector_snr_curve.png', dpi=150, bbox_inches='tight')
    print("SNR curve saved as 'detector_snr_curve.png'")
    print()


def example_mzi_mesh():
    """Demonstrate MZI mesh neural network."""
    print("=== MZI Mesh Neural Network Example ===")
    
    # Create 4x4 MZI mesh
    mesh_size = (4, 4)
    mzi_mesh = pn.MZIMesh(
        size=mesh_size,
        topology='rectangular',
        phase_encoding='differential'
    )
    
    print(f"Mesh size: {mesh_size}")
    print(f"Number of phase shifters: {mzi_mesh.n_phases}")
    
    # Create target unitary matrix (random orthogonal)
    target_unitary = pn.random_unitary(4)
    print(f"Target matrix shape: {target_unitary.shape}")
    
    # Decompose into phase shifter settings
    phases = mzi_mesh.decompose(target_unitary)
    mzi_mesh.set_phases(phases)
    
    # Test the implementation
    implemented_unitary = mzi_mesh.measure_unitary()
    fidelity = pn.matrix_fidelity(target_unitary, implemented_unitary)
    
    print(f"Implementation fidelity: {fidelity:.4f}")
    
    # Test forward pass
    batch_size = 3
    input_fields = torch.randn(batch_size, 4, dtype=torch.complex64)
    output_fields = mzi_mesh.forward(input_fields)
    
    print(f"Input shape: {input_fields.shape}")
    print(f"Output shape: {output_fields.shape}")
    
    # Check power conservation (unitarity)
    input_power = torch.sum(torch.abs(input_fields)**2, dim=1)
    output_power = torch.sum(torch.abs(output_fields)**2, dim=1)
    power_conservation_error = torch.mean(torch.abs(input_power - output_power))
    
    print(f"Power conservation error: {power_conservation_error.item():.6f}")
    print()


def example_spiking_network():
    """Demonstrate photonic spiking neural network."""
    print("=== Photonic Spiking Neural Network Example ===")
    
    # Create SNN topology
    topology = [8, 12, 4]  # 8 inputs, 12 hidden, 4 outputs
    snn = pn.PhotonicSNN(
        topology=topology,
        neuron_model='photonic_lif',
        synapse_type='microring',
        timestep=1e-12  # 1 ps timestep
    )
    
    print(f"Network topology: {topology}")
    print(f"Timestep: {snn.timestep*1e12:.1f} ps")
    print(f"Number of layers: {len(snn.layers)}")
    
    # Configure optical parameters
    snn.configure_optics(
        waveguide_width=450e-9,
        microring_radius=5e-6,
        coupling_gap=200e-9,
        phase_shifter='thermal'
    )
    
    # Generate input spike pattern
    batch_size = 2
    n_timesteps = 50
    input_spikes = torch.randn(batch_size, topology[0])
    
    print(f"Input shape: {input_spikes.shape}")
    
    # Run network simulation
    output_spikes = snn.forward(input_spikes, n_timesteps=n_timesteps)
    
    print(f"Output shape: {output_spikes.shape}")  # [batch, neurons, time]
    
    # Calculate spike rates
    input_rate = torch.mean(torch.abs(input_spikes))
    output_rates = torch.mean(torch.abs(output_spikes), dim=2)  # Average over time
    
    print(f"Average input spike rate: {input_rate:.3f}")
    print(f"Output spike rates shape: {output_rates.shape}")
    
    # Network performance metrics
    print(f"Optical efficiency: {snn.efficiency:.1%}")
    print(f"Network latency: {snn.latency_ps:.1f} ps")
    print()


def example_power_analysis():
    """Demonstrate power budget analysis."""
    print("=== Power Budget Analysis Example ===")
    
    # Create a complex photonic system
    mzi_modulator = pn.MachZehnderModulator(length=2e-3, v_pi=1.5)
    mzi_modulator.set_drive_voltage(1.0)  # 1V drive
    
    # Analyze power consumption
    analyzer = pn.PowerBudgetAnalyzer(mzi_modulator)
    power_report = analyzer.analyze(
        input_power_dbm=0,      # 0 dBm input
        include_thermal=True,
        include_electrical=True
    )
    
    print("Power Analysis Results:")
    print(f"Total power consumption: {power_report.total_mw:.2f} mW")
    print(f"Optical efficiency: {power_report.optical_efficiency:.1%}")
    print(f"Wall-plug efficiency: {power_report.wall_plug_efficiency:.1%}")
    
    # Display detailed breakdown
    print("\nOptical Powers:")
    for component, power in power_report.optical_powers.items():
        print(f"  {component}: {power*1000:.3f} mW")
    
    print("\nElectrical Powers:")
    for component, power in power_report.electrical_powers.items():
        print(f"  {component}: {power*1000:.3f} mW")
    
    print()


def example_noise_simulation():
    """Demonstrate noise analysis."""
    print("=== Noise Analysis Example ===")
    
    # Create system for noise analysis
    waveguide = pn.SiliconWaveguide(length=1e-3)
    noise_sim = pn.NoiseSimulator(waveguide)
    
    print("Running SNR analysis...")
    
    # Analyze SNR vs input power
    snr_results = noise_sim.sweep_input_power(
        power_range_dbm=(-15, 5),
        n_points=21,
        include_shot_noise=True,
        include_thermal_noise=True,
        temperature=300
    )
    
    # Find optimum operating point
    max_snr_idx = np.argmax(snr_results['snr_db'])
    optimum_power_dbm = snr_results['power_dbm'][max_snr_idx]
    max_snr_db = snr_results['snr_db'][max_snr_idx]
    
    print(f"Optimum input power: {optimum_power_dbm:.1f} dBm")
    print(f"Maximum SNR: {max_snr_db:.1f} dB")
    
    # Plot SNR curve
    plt.figure(figsize=(10, 6))
    plt.plot(snr_results['power_dbm'], snr_results['snr_db'], 'g-o', linewidth=2, markersize=4, label='Total SNR')
    plt.plot(snr_results['power_dbm'], snr_results['shot_noise_limited'], 'r--', alpha=0.7, label='Shot noise limited')
    plt.plot(snr_results['power_dbm'], snr_results['thermal_noise_limited'], 'b--', alpha=0.7, label='Thermal noise limited')
    plt.axvline(optimum_power_dbm, color='orange', linestyle=':', label=f'Optimum ({optimum_power_dbm:.1f} dBm)')
    
    plt.xlabel('Input Power (dBm)')
    plt.ylabel('SNR (dB)')
    plt.title('Signal-to-Noise Ratio Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('snr_analysis.png', dpi=150, bbox_inches='tight')
    print("SNR analysis saved as 'snr_analysis.png'")
    print()


def example_model_compilation():
    """Demonstrate ONNX model compilation to photonic implementation."""
    print("=== Model Compilation Example ===")
    
    # Create a simple PyTorch model
    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 8)
            self.layer2 = torch.nn.Linear(8, 4)
            self.layer3 = torch.nn.Linear(4, 2)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
            return x
    
    # Create and compile model
    pytorch_model = SimpleNN()
    
    print("Original PyTorch model:")
    print(f"  Layer 1: {pytorch_model.layer1}")
    print(f"  Layer 2: {pytorch_model.layer2}")
    print(f"  Layer 3: {pytorch_model.layer3}")
    
    # Compile to photonic implementation
    photonic_model = pn.compile_to_photonic(
        pytorch_model,
        backend='mach_zehnder',
        wavelength=1550e-9,
        optimization_level=2
    )
    
    print(f"\nPhotonic model type: {type(photonic_model).__name__}")
    print(f"Layer sizes: {photonic_model.layer_sizes}")
    
    # Test both models
    test_input = torch.randn(3, 10)  # Batch of 3 samples
    
    pytorch_output = pytorch_model(test_input)
    photonic_output = photonic_model(test_input.to(torch.complex64))
    
    print(f"\nPyTorch output shape: {pytorch_output.shape}")
    print(f"Photonic output shape: {photonic_output.shape}")
    
    # Compare performance (simplified)
    print(f"Photonic model efficiency: {getattr(photonic_model, 'efficiency', 'N/A')}")
    print()


def main():
    """Run all basic usage examples."""
    print("Photon Neuromorphics SDK - Basic Usage Examples")
    print("=" * 50)
    print()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Run all examples
        example_waveguide_propagation()
        example_mzi_modulator()
        example_photodetector()
        example_mzi_mesh()
        example_spiking_network()
        example_power_analysis()
        example_noise_simulation()
        example_model_compilation()
        
        print("All examples completed successfully! 🌟")
        print("\nGenerated files:")
        print("  - mzi_modulation_curve.png")
        print("  - detector_snr_curve.png")
        print("  - snr_analysis.png")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please check your installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())