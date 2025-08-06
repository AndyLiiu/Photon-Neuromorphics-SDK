#!/usr/bin/env python3
"""
Demonstration of Generation 2 "Make It Robust" features in Photon Neuromorphics SDK.

This example showcases:
- Comprehensive error handling and recovery
- Advanced quantum noise modeling with crosstalk
- Thermal simulation with cooling systems  
- S-parameter analysis and multi-mode waveguides
- Adaptive calibration algorithms
- Real-time monitoring and logging

Author: Claude AI Assistant
Version: 0.2.0-robust
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import the enhanced Photon Neuromorphics SDK
import photon_neuro as pn


def demonstrate_error_handling():
    """Demonstrate robust error handling system."""
    print("\n=== Error Handling & Recovery Demo ===")
    
    try:
        # Create a component with invalid parameters (triggers validation)
        waveguide = pn.WaveguideBase(length=-1.0, width=50e-9, material="invalid_material")
    except pn.ValidationError as e:
        print(f"✓ Validation error caught: {e}")
        
        # Use error recovery to get fallback component
        recovery_context = {'expected_output_shape': (1,)}
        recovered_result = pn.global_error_recovery.handle_error(e, recovery_context)
        print(f"✓ Error recovery provided fallback: {recovered_result}")
    
    # Demonstrate safe execution decorator
    @pn.safe_execution(fallback_value=torch.ones(1, dtype=torch.complex64))
    def potentially_failing_simulation(input_field):
        if torch.rand(1) < 0.5:  # 50% chance of failure
            raise pn.SimulationError("Random simulation failure")
        return input_field * 2.0
    
    input_field = torch.tensor([1.0 + 0j])
    result = potentially_failing_simulation(input_field)
    print(f"✓ Safe execution result: {result}")


def demonstrate_advanced_noise_modeling():
    """Demonstrate advanced quantum noise and crosstalk modeling."""
    print("\n=== Advanced Noise Modeling Demo ===")
    
    # Create enhanced quantum noise model
    noise_model = pn.QuantumNoiseModel(temperature=300)
    
    # Model shot noise with quantum corrections
    photocurrent = 1e-6  # 1 µA
    bandwidth = 10e9     # 10 GHz
    shot_variance = noise_model.shot_noise_variance(photocurrent, bandwidth)
    print(f"✓ Shot noise variance: {shot_variance:.2e} A²")
    
    # Model pump RIN noise
    laser_power = 1e-3   # 1 mW
    frequency = 1e9      # 1 GHz
    rin_noise = noise_model.pump_relative_intensity_noise(laser_power, frequency)
    print(f"✓ RIN noise: {rin_noise:.2e}")
    
    # Add coherence noise to optical field
    optical_field = torch.tensor([1.0 + 0j, 0.8 + 0.2j, 0.6 + 0.4j])
    noisy_field = noise_model.coherence_noise(optical_field, coherence_length=1e-3)
    print(f"✓ Coherence noise applied: {torch.abs(noisy_field)}")
    
    # Demonstrate waveguide crosstalk
    crosstalk_model = pn.WaveguideCrosstalk(
        n_guides=4, spacing=2e-6, coupling_length=1e-3
    )
    
    input_powers = np.array([1.0, 0.5, 0.8, 0.3])  # mW
    output_powers = crosstalk_model.calculate_crosstalk(input_powers)
    print(f"✓ Crosstalk analysis:")
    print(f"  Input powers: {input_powers}")
    print(f"  Output powers: {output_powers}")
    
    # Temperature-dependent noise
    temp_noise = pn.TemperatureDependentNoise(base_temperature=25.0)
    phase_noise = temp_noise.thermal_phase_noise(
        component_length=1e-3, temperature_fluctuation=1.0, material="silicon"
    )
    print(f"✓ Thermal phase noise: {phase_noise:.4f} rad")


def demonstrate_thermal_simulation():
    """Demonstrate advanced thermal simulation."""
    print("\n=== Thermal Simulation Demo ===")
    
    # Create thermal simulator with chip geometry
    geometry = {
        'length': 10e-3,    # 10 mm
        'width': 5e-3,      # 5 mm
        'thickness': 300e-6, # 300 µm
        'grid_points': (20, 10, 5)  # Coarse grid for demo
    }
    
    thermal_sim = pn.ThermalSimulator(geometry=geometry)
    
    # Add heat sources (e.g., modulators, phase shifters)
    thermal_sim.add_heat_source(
        location=(2e-3, 2e-3, 150e-6),  # Center of chip
        power=50e-3,  # 50 mW dissipation
        size=(500e-6, 500e-6, 100e-6)
    )
    
    thermal_sim.add_heat_source(
        location=(7e-3, 3e-3, 150e-6),  # Another hotspot
        power=30e-3   # 30 mW
    )
    
    print("✓ Heat sources added to thermal model")
    
    # Set up cooling system
    thermal_sim.cooling_system.set_liquid_cooling(
        pump_power=5.0,    # 5W pump
        flow_rate=2e-3,    # 2 mL/s flow rate
        coolant_temperature=20.0  # 20°C coolant
    )
    print("✓ Liquid cooling system configured")
    
    # Solve steady-state thermal distribution
    try:
        temperature_field = thermal_sim.solve_steady_state()
        
        if temperature_field is not None:
            max_temp = np.max(temperature_field)
            avg_temp = np.mean(temperature_field)
            print(f"✓ Thermal simulation completed:")
            print(f"  Maximum temperature: {max_temp:.1f} °C")
            print(f"  Average temperature: {avg_temp:.1f} °C")
            
            # Analyze hotspots
            hotspots = thermal_sim.analyze_hotspots(threshold_temperature=40.0)
            print(f"  Number of hotspots > 40°C: {len(hotspots)}")
            
        else:
            print("⚠ Thermal simulation returned None (using fallback)")
            
    except Exception as e:
        print(f"⚠ Thermal simulation failed: {e}")
    
    # Demonstrate thermo-optic coupling
    thermo_optic = pn.ThermalOpticalCoupling()
    
    temperature_change = 10.0  # 10°C increase
    phase_shift = thermo_optic.calculate_phase_shift(
        material="silicon", 
        length=1e-3, 
        temperature_change=temperature_change
    )
    print(f"✓ Thermo-optic phase shift: {phase_shift:.4f} rad for {temperature_change}°C change")


def demonstrate_circuit_analysis():
    """Demonstrate advanced circuit analysis features."""
    print("\n=== Circuit Analysis Demo ===")
    
    # Multi-mode waveguide analysis
    multimode_wg = pn.MultiModeWaveguide(
        width=2e-6,      # 2 µm wide (supports multiple modes)
        height=220e-9,   # 220 nm thick
        material="silicon"
    )
    
    print(f"✓ Multi-mode waveguide created with {len(multimode_wg.modes)} supported modes:")
    for i, mode in enumerate(multimode_wg.modes[:3]):  # Show first 3 modes
        print(f"  Mode {i}: {mode['type']}{mode['m']}{mode['n']}, n_eff = {mode['n_eff']:.3f}")
    
    # Calculate mode coupling
    if len(multimode_wg.modes) >= 2:
        coupling = multimode_wg.calculate_mode_coupling(0, 1, perturbation_strength=1e-4)
        print(f"✓ Coupling between modes 0-1: {coupling:.2e} m⁻¹")
    
    # Polarization analysis
    pol_model = pn.PolarizationModel()
    
    # Create polarization components
    linear_pol = pol_model.linear_polarizer(angle=45)  # 45° polarizer
    qwp = pol_model.quarter_wave_plate(angle=45)       # Quarter-wave plate
    
    # Calculate PDL
    pdl_db = pol_model.calculate_pdl(linear_pol)
    print(f"✓ Linear polarizer PDL: {pdl_db:.1f} dB")
    
    # Dispersion modeling
    dispersion_model = pn.DispersionModel()
    
    # Calculate group delay for silicon waveguide
    wavelengths = np.linspace(1520e-9, 1580e-9, 100)  # C-band
    group_delays = dispersion_model.calculate_group_delay(
        material="silicon", length=1e-3, wavelengths=wavelengths
    )
    
    dispersion_param = dispersion_model.calculate_dispersion_parameter(
        material="silicon", wavelength=1550e-9
    )
    print(f"✓ Silicon dispersion parameter: {dispersion_param:.2f} ps/(nm·km)")
    print(f"✓ Group delay variation: {np.ptp(group_delays)*1e12:.2f} ps over C-band")


def demonstrate_calibration_system():
    """Demonstrate adaptive calibration algorithms."""
    print("\n=== Calibration System Demo ===")
    
    # Create calibration manager
    cal_manager = pn.CalibrationManager()
    
    # Register different calibration algorithms
    pid_calibrator = pn.PIDCalibrator("phase_shifter_pid", kp=2.0, ki=0.5, kd=0.1)
    ml_calibrator = pn.MLCalibrator("modulator_ml", model_type="linear")
    
    cal_manager.register_calibrator("pid", pid_calibrator)
    cal_manager.register_calibrator("ml", ml_calibrator)
    
    # Simulate a drift parameter that needs calibration
    class DriftingParameter:
        def __init__(self, target=0.0, drift_rate=0.01):
            self.target = target
            self.current = target
            self.drift_rate = drift_rate
            self.correction = 0.0
            
        def measure(self):
            # Simulate measurement with noise
            self.current += np.random.normal(0, 0.001) + self.drift_rate * 0.1
            return self.current + np.random.normal(0, 0.005)
            
        def adjust(self, correction):
            self.correction += correction
            self.current -= correction * 0.8  # 80% correction efficiency
    
    # Create a parameter that needs calibration
    parameter = DriftingParameter(target=1.0, drift_rate=0.02)
    
    print("✓ Simulating parameter calibration...")
    
    # Perform calibration
    try:
        result = cal_manager.calibrate_component(
            component="test_modulator",
            parameter="phase_bias",
            target_value=1.0,
            measurement_func=parameter.measure,
            adjustment_func=parameter.adjust,
            calibrator_name="pid"
        )
        
        print(f"✓ Calibration result:")
        print(f"  Success: {result.success}")
        print(f"  Initial error: {result.initial_error:.4f}")
        print(f"  Final error: {result.final_error:.4f}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Convergence time: {result.convergence_time:.3f} s")
        print(f"  Confidence: {result.confidence:.3f}")
        
    except Exception as e:
        print(f"⚠ Calibration failed: {e}")
    
    # Demonstrate safety checking
    safety_checker = pn.SafetyChecker()
    safety_checker.set_safety_limit("voltage", -5.0, 5.0)
    
    safe_voltage = safety_checker.check_safety("voltage", 3.0)
    unsafe_voltage = safety_checker.check_safety("voltage", 10.0)
    
    print(f"✓ Safety checks: 3V safe={safe_voltage}, 10V safe={unsafe_voltage}")


def demonstrate_logging_monitoring():
    """Demonstrate comprehensive logging and monitoring."""
    print("\n=== Logging & Monitoring Demo ===")
    
    # Initialize enhanced logger
    logger = pn.PhotonLogger(
        name="demo_logger",
        log_level="INFO",
        enable_console=True,
        enable_file=False,  # Disable file logging for demo
        enable_metrics=True
    )
    
    # Start performance monitoring
    logger.start_performance_monitoring(interval=1.0)
    
    # Demonstrate structured logging
    logger.info("Starting robust features demonstration", 
                component="demo", operation="startup")
    
    # Record performance metrics
    logger.metric("cpu_usage", 45.2, "percent", component="system")
    logger.metric("memory_usage", 2.1, "GB", component="system")
    logger.metric("simulation_throughput", 1250.0, "ops/sec", component="simulator")
    
    # Demonstrate progress tracking
    progress_tracker = logger.progress_tracker
    progress_tracker.start_task("demo_simulation", total_steps=100)
    
    # Simulate work with progress updates
    for step in range(0, 101, 20):
        progress_tracker.update_progress("demo_simulation", step=step, 
                                       stage=f"processing_step_{step}")
        time.sleep(0.1)  # Simulate work
    
    progress_tracker.finish_task("demo_simulation")
    
    print("✓ Progress tracking completed")
    
    # Demonstrate diagnostic mode
    diagnostic = logger.diagnostic_mode
    diagnostic.enable(capture_tensors=True, capture_memory=True)
    
    # Simulate tensor operations with diagnostic capture
    test_tensor = torch.randn(100, 100)
    gradient_tensor = torch.randn(100, 100, requires_grad=True)
    
    diagnostic.capture_state("tensor_operation", 
                           input=test_tensor,
                           gradients=gradient_tensor)
    
    print("✓ Diagnostic data captured")
    
    # Get metrics summary
    metrics_stats = logger.metrics_collector.get_metric_stats("cpu_usage")
    if metrics_stats:
        print(f"✓ CPU usage statistics: mean={metrics_stats.get('mean', 0):.1f}%")
    
    # Stop monitoring
    logger.stop_performance_monitoring()


@pn.log_execution_time(metric_name="demo_execution_time")
@pn.monitor_memory_usage()
def run_comprehensive_demo():
    """Run comprehensive demonstration of all robust features."""
    print("🔬 Photon Neuromorphics SDK - Generation 2 'Make It Robust' Demo")
    print("=" * 70)
    
    # Initialize global logger for the demo
    pn.global_logger.info("Starting comprehensive robust features demo")
    
    try:
        # Run all demonstrations
        demonstrate_error_handling()
        demonstrate_advanced_noise_modeling()
        demonstrate_thermal_simulation()
        demonstrate_circuit_analysis()
        demonstrate_calibration_system()
        demonstrate_logging_monitoring()
        
        print("\n" + "=" * 70)
        print("✅ All robust features demonstrated successfully!")
        print(f"SDK Version: {pn.get_version()}")
        print("Ready for production-grade photonic neural network research! 🚀")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        pn.global_logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Clean shutdown
        pn.global_logger.info("Demo completed")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run the comprehensive demo
    run_comprehensive_demo()