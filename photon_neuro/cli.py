"""
Command line interface for Photon Neuromorphics SDK.
"""

import argparse
import sys
import os
import json
from typing import Dict, Any, List, Optional

import torch
import numpy as np

from . import __version__
from .compiler import compile_to_photonic
from .simulation import PhotonicSimulator
from .hardware import PhotonicChip
from .networks import PhotonicSNN, MZIMesh
from .utils import PhysicalConstants


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    
    parser = argparse.ArgumentParser(
        prog='photon-neuro',
        description='Photon Neuromorphics SDK - Silicon-photonic neural networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  photon-neuro compile model.onnx --backend mach_zehnder
  photon-neuro simulate --network snn --topology 784,256,10
  photon-neuro hardware --chip-address visa://192.168.1.100 --calibrate
  photon-neuro benchmark --device cuda --batch-size 32
        """
    )
    
    parser.add_argument('--version', action='version', version=f'photon-neuro {__version__}')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile models to photonic implementations')
    compile_parser.add_argument('model', type=str, help='Input model file (ONNX or PyTorch)')
    compile_parser.add_argument('--backend', choices=['mach_zehnder', 'microring'], default='mach_zehnder')
    compile_parser.add_argument('--wavelength', type=float, default=1550e-9, help='Operating wavelength (m)')
    compile_parser.add_argument('--output', '-o', type=str, help='Output file path')
    compile_parser.add_argument('--optimization-level', type=int, default=2, choices=[0, 1, 2, 3])
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run photonic simulations')
    sim_parser.add_argument('--network', choices=['snn', 'mzi', 'microring'], required=True)
    sim_parser.add_argument('--topology', type=str, help='Network topology (e.g., 784,256,10)')
    sim_parser.add_argument('--input-data', type=str, help='Input data file (npy)')
    sim_parser.add_argument('--timesteps', type=int, default=100, help='Number of timesteps for SNN')
    sim_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    sim_parser.add_argument('--output-dir', type=str, default='./results')
    
    # Hardware command  
    hw_parser = subparsers.add_parser('hardware', help='Hardware interface and control')
    hw_parser.add_argument('--chip-address', type=str, default='visa://192.168.1.100')
    hw_parser.add_argument('--calibrate', action='store_true', help='Run calibration sequence')
    hw_parser.add_argument('--self-test', action='store_true', help='Run hardware self-test')
    hw_parser.add_argument('--temperature', type=float, help='Set chip temperature')
    hw_parser.add_argument('--power-budget', action='store_true', help='Analyze power consumption')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    bench_parser.add_argument('--batch-size', type=int, default=16)
    bench_parser.add_argument('--network-size', type=str, default='8,8', help='Network size (e.g., 8,8)')
    bench_parser.add_argument('--iterations', type=int, default=100)
    bench_parser.add_argument('--profile', action='store_true', help='Enable detailed profiling')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analysis', help='Analyze photonic designs')
    analysis_parser.add_argument('--model', type=str, required=True, help='Model file to analyze')
    analysis_parser.add_argument('--noise-analysis', action='store_true')
    analysis_parser.add_argument('--power-analysis', action='store_true')
    analysis_parser.add_argument('--sensitivity-analysis', action='store_true')
    analysis_parser.add_argument('--output-format', choices=['json', 'csv', 'html'], default='json')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between model formats')
    convert_parser.add_argument('input_file', type=str, help='Input model file')
    convert_parser.add_argument('output_file', type=str, help='Output model file')
    convert_parser.add_argument('--format', choices=['onnx', 'torch', 'netlist'], required=True)
    
    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}


def compile_command(args) -> int:
    """Handle compile command."""
    try:
        print(f"Compiling {args.model} to photonic implementation...")
        
        # Load model and compile
        photonic_model = compile_to_photonic(
            args.model,
            backend=args.backend,
            wavelength=args.wavelength,
            optimization_level=args.optimization_level
        )
        
        # Save compiled model
        output_path = args.output or f"{os.path.splitext(args.model)[0]}_photonic.pt"
        torch.save(photonic_model.state_dict(), output_path)
        
        print(f"Compiled model saved to {output_path}")
        
        # Print summary
        if hasattr(photonic_model, 'to_netlist'):
            netlist = photonic_model.to_netlist()
            print(f"Network type: {netlist.get('type', 'unknown')}")
            if 'topology' in netlist:
                print(f"Topology: {netlist['topology']}")
                
        return 0
        
    except Exception as e:
        print(f"Compilation failed: {e}")
        return 1


def simulate_command(args) -> int:
    """Handle simulate command."""
    try:
        print(f"Running {args.network} simulation...")
        
        # Parse topology
        if args.topology:
            topology = [int(x.strip()) for x in args.topology.split(',')]
        else:
            topology = [10, 20, 10]  # Default
            
        # Create network
        if args.network == 'snn':
            model = PhotonicSNN(topology, timestep=1e-12)
        elif args.network == 'mzi':
            if len(topology) == 2:
                model = MZIMesh(size=tuple(topology))
            else:
                print(f"MZI mesh requires 2D topology, got {len(topology)}D")
                return 1
        else:
            print(f"Unsupported network type: {args.network}")
            return 1
            
        # Move to device
        device = torch.device(args.device)
        model = model.to(device)
        
        # Load input data
        if args.input_data:
            input_data = torch.from_numpy(np.load(args.input_data)).float().to(device)
        else:
            # Generate random input
            input_data = torch.randn(1, topology[0], device=device)
            
        # Run simulation
        print("Running forward pass...")
        
        if args.network == 'snn':
            outputs = model(input_data, n_timesteps=args.timesteps)
            print(f"Output shape: {outputs.shape}")
            print(f"Final spike rates: {torch.sum(outputs, dim=2).mean(dim=0)}")
        else:
            outputs = model(input_data)
            print(f"Output shape: {outputs.shape}")
            print(f"Output values: {outputs}")
            
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{args.network}_output.npy")
        np.save(output_file, outputs.detach().cpu().numpy())
        print(f"Results saved to {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return 1


def hardware_command(args) -> int:
    """Handle hardware command."""
    try:
        print(f"Connecting to photonic chip at {args.chip_address}...")
        
        # Connect to hardware
        chip = PhotonicChip(args.chip_address)
        if not chip.initialize():
            print("Failed to initialize photonic chip")
            return 1
            
        print("Chip initialized successfully")
        
        # Run self-test
        if args.self_test:
            print("Running self-test...")
            results = chip.run_self_test()
            
            for component, passed in results.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {component}: {status}")
                
        # Run calibration
        if args.calibrate:
            print("Running calibration sequence...")
            from .hardware import HardwareCalibrator
            
            calibrator = HardwareCalibrator(chip)
            cal_data = calibrator.run_full_calibration()
            
            cal_file = "calibration_data.json"
            with open(cal_file, 'w') as f:
                json.dump(cal_data, f, indent=2)
            print(f"Calibration data saved to {cal_file}")
            
        # Set temperature
        if args.temperature is not None:
            print(f"Setting chip temperature to {args.temperature}°C...")
            chip.thermal_controllers.set_temperature(args.temperature)
            
        # Power budget analysis
        if args.power_budget:
            print("Analyzing power budget...")
            from .simulation import PowerBudgetAnalyzer
            
            analyzer = PowerBudgetAnalyzer(chip)
            report = analyzer.analyze()
            
            print(f"Total power: {report.total_mw:.1f} mW")
            print(f"Optical efficiency: {report.optical_efficiency:.1%}")
            
        return 0
        
    except Exception as e:
        print(f"Hardware operation failed: {e}")
        return 1


def benchmark_command(args) -> int:
    """Handle benchmark command."""
    try:
        print("Running performance benchmarks...")
        
        # Parse network size
        size = [int(x.strip()) for x in args.network_size.split(',')]
        device = torch.device(args.device)
        
        # Create test network
        if len(size) == 2:
            model = MZIMesh(size=tuple(size)).to(device)
        else:
            print(f"Invalid network size: {args.network_size}")
            return 1
            
        # Generate test data
        input_data = torch.randn(args.batch_size, size[0], device=device, dtype=torch.complex64)
        
        # Warmup
        print("Warming up...")
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_data)
                
        # Benchmark
        print(f"Running {args.iterations} iterations...")
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        import time
        cpu_start = time.time()
        
        if start_time:
            start_time.record()
            
        for _ in range(args.iterations):
            with torch.no_grad():
                outputs = model(input_data)
                
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            
        cpu_end = time.time()
        
        # Calculate metrics
        if start_time and end_time:
            gpu_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            avg_time = gpu_time / args.iterations
        else:
            total_time = cpu_end - cpu_start
            avg_time = total_time / args.iterations
            
        throughput = args.batch_size / avg_time
        
        print(f"\nBenchmark Results:")
        print(f"  Average time per batch: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")
        print(f"  Device: {device}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Network size: {size}")
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
            print(f"  GPU Memory - Allocated: {memory_allocated:.1f} MB, Reserved: {memory_reserved:.1f} MB")
            
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1


def analysis_command(args) -> int:
    """Handle analysis command."""
    try:
        print(f"Analyzing model: {args.model}")
        
        # Load model
        if args.model.endswith('.pt') or args.model.endswith('.pth'):
            model = torch.load(args.model, map_location='cpu')
        else:
            print(f"Unsupported model format: {args.model}")
            return 1
            
        results = {}
        
        # Noise analysis
        if args.noise_analysis:
            print("Running noise analysis...")
            from .simulation import NoiseSimulator
            
            noise_sim = NoiseSimulator(model)
            snr_results = noise_sim.sweep_input_power((-20, 10))
            
            results['noise_analysis'] = {
                'min_snr_db': float(np.min(snr_results['snr_db'])),
                'max_snr_db': float(np.max(snr_results['snr_db'])),
                'avg_snr_db': float(np.mean(snr_results['snr_db']))
            }
            
        # Power analysis
        if args.power_analysis:
            print("Running power analysis...")
            from .simulation import PowerBudgetAnalyzer
            
            power_analyzer = PowerBudgetAnalyzer(model)
            power_report = power_analyzer.analyze()
            
            results['power_analysis'] = {
                'total_power_mw': power_report.total_mw,
                'optical_efficiency': power_report.optical_efficiency,
                'wall_plug_efficiency': power_report.wall_plug_efficiency
            }
            
        # Output results
        if args.output_format == 'json':
            print(json.dumps(results, indent=2))
        elif args.output_format == 'csv':
            # Flatten results for CSV
            flat_results = {}
            for category, values in results.items():
                for key, value in values.items():
                    flat_results[f"{category}_{key}"] = value
                    
            print(','.join(flat_results.keys()))
            print(','.join([str(v) for v in flat_results.values()]))
        
        return 0
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        
    # Set verbosity
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
        
    # Route to command handlers
    if args.command == 'compile':
        return compile_command(args)
    elif args.command == 'simulate':
        return simulate_command(args)
    elif args.command == 'hardware':
        return hardware_command(args)
    elif args.command == 'benchmark':
        return benchmark_command(args)
    elif args.command == 'analysis':
        return analysis_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())