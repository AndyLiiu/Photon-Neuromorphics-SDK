"""
Circuit-level simulation for photonic systems.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from ..core import PhotonicComponent


class CircuitLevelSimulator:
    """Circuit-level simulator using scattering matrix approach."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.components: Dict[str, PhotonicComponent] = {}
        self.s_parameters: Dict[str, np.ndarray] = {}
        self.frequencies = None
        self.wavelengths = None
        
    def add_component(self, name: str, component: PhotonicComponent, 
                     position: Tuple[float, float] = (0, 0)):
        """Add a component to the circuit."""
        self.components[name] = component
        self.graph.add_node(name, component=component, position=position)
        
        # Calculate S-parameters if method exists
        if hasattr(component, 'get_s_parameters'):
            try:
                # Default frequency range
                if self.frequencies is None:
                    self.frequencies = np.linspace(190e12, 200e12, 1000)  # ~1500-1580 nm
                    
                s_params = component.get_s_parameters(self.frequencies)
                self.s_parameters[name] = s_params
            except Exception as e:
                print(f"Warning: Could not calculate S-parameters for {name}: {e}")
                # Default to unity transmission
                n_ports = 2
                s_params = np.zeros((len(self.frequencies), n_ports, n_ports), dtype=complex)
                for i, freq in enumerate(self.frequencies):
                    s_params[i] = np.eye(n_ports)
                self.s_parameters[name] = s_params
                
    def connect(self, component1: str, port1: int, component2: str, port2: int,
               length: float = 0.0, loss_db_per_cm: float = 0.1):
        """Connect two components with a waveguide."""
        if component1 not in self.components or component2 not in self.components:
            raise ValueError("Both components must be added to circuit first")
            
        # Calculate waveguide parameters
        loss_linear = 10**(-loss_db_per_cm * length * 100 / 20)  # Convert to linear
        phase_delay = 2 * np.pi * 2.4 * length / 1550e-9  # Assume n_eff = 2.4
        
        self.graph.add_edge(component1, component2, 
                          port1=port1, port2=port2, 
                          length=length, loss=loss_linear, phase=phase_delay)
        
    def analyze_frequency_response(self, input_component: str, input_port: int,
                                 output_component: str, output_port: int) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze frequency response between two ports."""
        if self.frequencies is None:
            raise ValueError("No frequency range defined")
            
        transmission = []
        phase = []
        
        for i, freq in enumerate(self.frequencies):
            try:
                # Calculate transfer function at this frequency
                h_transfer = self._calculate_transfer_function(
                    input_component, input_port, output_component, output_port, i
                )
                
                transmission.append(np.abs(h_transfer))
                phase.append(np.angle(h_transfer))
                
            except Exception as e:
                print(f"Warning: Transfer function calculation failed at {freq}: {e}")
                transmission.append(0.0)
                phase.append(0.0)
                
        return np.array(transmission), np.array(phase)
        
    def _calculate_transfer_function(self, input_comp: str, input_port: int,
                                   output_comp: str, output_port: int, freq_idx: int) -> complex:
        """Calculate transfer function using proper S-parameter matrix operations."""
        try:
            # Find path from input to output
            if input_comp == output_comp:
                # Same component - use S-parameter directly
                if input_comp in self.s_parameters:
                    s_matrix = self.s_parameters[input_comp][freq_idx]
                    if (input_port < s_matrix.shape[1] and 
                        output_port < s_matrix.shape[0]):
                        return s_matrix[output_port, input_port]
                return 1.0 if input_port == output_port else 0.0
                
            # Find shortest path
            if nx.has_path(self.graph, input_comp, output_comp):
                path = nx.shortest_path(self.graph, input_comp, output_comp)
                
                # Build cascaded S-parameter matrix
                return self._cascade_s_parameters(path, input_port, output_port, freq_idx)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Warning: Transfer function calculation failed: {e}")
            return 0.0
            
    def _cascade_s_parameters(self, path: list, input_port: int, output_port: int, freq_idx: int) -> complex:
        """Cascade S-parameters along a path using proper matrix operations."""
        if len(path) < 2:
            return 1.0
            
        # Start with first component
        if path[0] in self.s_parameters:
            s_total = self.s_parameters[path[0]][freq_idx].copy()
        else:
            s_total = np.eye(2, dtype=complex)
            
        # Cascade through path
        for i in range(len(path) - 1):
            comp1, comp2 = path[i], path[i+1]
            edge_data = self.graph[comp1][comp2] if self.graph.has_edge(comp1, comp2) else {}
            
            # Get waveguide S-parameters between components
            s_waveguide = self._get_waveguide_s_params(edge_data, freq_idx)
            
            # Get next component S-parameters
            if comp2 in self.s_parameters:
                s_comp2 = self.s_parameters[comp2][freq_idx]
            else:
                s_comp2 = np.eye(2, dtype=complex)
                
            # Cascade waveguide and component
            s_total = self._cascade_two_port_s_params(s_total, s_waveguide)
            s_total = self._cascade_two_port_s_params(s_total, s_comp2)
            
        # Extract desired transfer function
        if (input_port < s_total.shape[1] and output_port < s_total.shape[0]):
            return s_total[output_port, input_port]
        else:
            return 0.0
            
    def _get_waveguide_s_params(self, edge_data: dict, freq_idx: int) -> np.ndarray:
        """Generate S-parameters for waveguide connection."""
        loss = edge_data.get('loss', 1.0)
        phase = edge_data.get('phase', 0.0)
        
        # Ideal waveguide S-parameters
        s_waveguide = np.array([[0, loss * np.exp(1j * phase)],
                               [loss * np.exp(1j * phase), 0]], dtype=complex)
        return s_waveguide
        
    def _cascade_two_port_s_params(self, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
        """Cascade two 2x2 S-parameter matrices."""
        # Standard S-parameter cascading formulas
        try:
            denominator = 1 - s1[1, 0] * s2[0, 1]
            
            if abs(denominator) < 1e-12:  # Avoid division by zero
                return s2  # Fallback
                
            s11 = s1[0, 0] + (s1[0, 1] * s2[0, 0] * s1[1, 0]) / denominator
            s12 = (s1[0, 1] * s2[0, 1]) / denominator
            s21 = (s2[1, 0] * s1[1, 1]) / denominator
            s22 = s2[1, 1] + (s2[1, 0] * s1[1, 1] * s2[0, 1]) / denominator
            
            return np.array([[s11, s12], [s21, s22]], dtype=complex)
            
        except Exception as e:
            print(f"Warning: S-parameter cascading failed: {e}")
            return s2  # Fallback to second matrix
            
    def calculate_group_delay(self, input_component: str, input_port: int,
                            output_component: str, output_port: int) -> np.ndarray:
        """Calculate group delay from phase response."""
        _, phase = self.analyze_frequency_response(
            input_component, input_port, output_component, output_port
        )
        
        # Unwrap phase
        phase_unwrapped = np.unwrap(phase)
        
        # Calculate group delay: τ_g = -dφ/dω
        domega = 2 * np.pi * (self.frequencies[1] - self.frequencies[0])
        group_delay = -np.gradient(phase_unwrapped, domega)
        
        return group_delay
        
    def find_resonances(self, input_component: str, input_port: int,
                       output_component: str, output_port: int,
                       threshold: float = 0.7) -> List[float]:
        """Find resonance frequencies."""
        transmission, _ = self.analyze_frequency_response(
            input_component, input_port, output_component, output_port
        )
        
        # Find peaks above threshold
        resonances = []
        for i in range(1, len(transmission) - 1):
            if (transmission[i] > threshold and 
                transmission[i] > transmission[i-1] and 
                transmission[i] > transmission[i+1]):
                resonances.append(self.frequencies[i])
                
        return resonances
        
    def sensitivity_analysis(self, parameter_variations: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform sensitivity analysis on circuit parameters."""
        baseline_responses = {}
        sensitivity_results = {}
        
        # Get baseline responses for all component pairs
        for input_comp in self.components:
            for output_comp in self.components:
                if input_comp != output_comp:
                    key = f"{input_comp}_to_{output_comp}"
                    try:
                        transmission, phase = self.analyze_frequency_response(
                            input_comp, 0, output_comp, 0
                        )
                        baseline_responses[key] = {
                            'transmission': transmission,
                            'phase': phase
                        }
                    except Exception as e:
                        print(f"Warning: Baseline analysis failed for {key}: {e}")
                        continue
                        
        # Vary parameters and measure changes
        for comp_name, variations in parameter_variations.items():
            if comp_name not in self.components:
                continue
                
            sensitivity_results[comp_name] = {}
            component = self.components[comp_name]
            
            for param_name, delta_percent in variations.items():
                if hasattr(component, param_name):
                    # Store original value
                    original_value = getattr(component, param_name)
                    
                    # Apply variation
                    delta_value = original_value * delta_percent / 100
                    setattr(component, param_name, original_value + delta_value)
                    
                    # Recalculate S-parameters
                    if hasattr(component, 'get_s_parameters'):
                        try:
                            self.s_parameters[comp_name] = component.get_s_parameters(self.frequencies)
                        except:
                            pass
                            
                    # Measure response changes
                    param_sensitivity = {}
                    for response_key, baseline in baseline_responses.items():
                        if comp_name in response_key:
                            try:
                                input_comp, output_comp = response_key.split('_to_')
                                new_transmission, new_phase = self.analyze_frequency_response(
                                    input_comp, 0, output_comp, 0
                                )
                                
                                # Calculate sensitivity metric
                                transmission_change = np.mean(np.abs(
                                    new_transmission - baseline['transmission']
                                ))
                                phase_change = np.mean(np.abs(
                                    new_phase - baseline['phase']
                                ))
                                
                                param_sensitivity[response_key] = {
                                    'transmission_sensitivity': transmission_change / delta_percent,
                                    'phase_sensitivity': phase_change / delta_percent
                                }
                                
                            except Exception as e:
                                print(f"Warning: Sensitivity calculation failed: {e}")
                                continue
                                
                    sensitivity_results[comp_name][param_name] = param_sensitivity
                    
                    # Restore original value
                    setattr(component, param_name, original_value)
                    
        return sensitivity_results
        
    def optimize_design(self, target_response: np.ndarray, 
                       optimization_params: Dict[str, Tuple[float, float]],
                       method: str = "least_squares") -> Dict[str, float]:
        """Optimize circuit design to match target response."""
        from scipy.optimize import minimize
        
        # Define objective function
        def objective(params_array):
            # Apply parameters
            param_idx = 0
            for comp_name, param_dict in optimization_params.items():
                if comp_name in self.components:
                    component = self.components[comp_name]
                    for param_name, (min_val, max_val) in param_dict.items():
                        if hasattr(component, param_name):
                            # Scale parameter from [0,1] to [min_val, max_val]
                            param_value = min_val + params_array[param_idx] * (max_val - min_val)
                            setattr(component, param_name, param_value)
                            param_idx += 1
                            
            # Calculate current response (simplified - single path)
            try:
                comp_names = list(self.components.keys())
                if len(comp_names) >= 2:
                    current_response, _ = self.analyze_frequency_response(
                        comp_names[0], 0, comp_names[-1], 0
                    )
                    
                    # Calculate error
                    if len(current_response) == len(target_response):
                        error = np.sum((current_response - target_response)**2)
                        return error
                        
            except Exception as e:
                print(f"Warning: Objective function evaluation failed: {e}")
                
            return 1e6  # Large error if failed
            
        # Count total parameters
        n_params = sum(len(param_dict) for param_dict in optimization_params.values())
        
        # Initial guess (middle of parameter ranges)
        x0 = np.ones(n_params) * 0.5
        
        # Bounds (all parameters normalized to [0,1])
        bounds = [(0, 1) for _ in range(n_params)]
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        # Extract optimized parameters
        optimized_params = {}
        param_idx = 0
        for comp_name, param_dict in optimization_params.items():
            optimized_params[comp_name] = {}
            for param_name, (min_val, max_val) in param_dict.items():
                param_value = min_val + result.x[param_idx] * (max_val - min_val)
                optimized_params[comp_name][param_name] = param_value
                param_idx += 1
                
        return optimized_params
        
    def export_netlist(self) -> str:
        """Export circuit as SPICE-like netlist."""
        netlist = "* Photonic Circuit Netlist\n"
        netlist += f"* Generated by CircuitLevelSimulator\n\n"
        
        # Components
        for name, component in self.components.items():
            try:
                netlist_dict = component.to_netlist()
                comp_type = netlist_dict.get('type', 'unknown')
                netlist += f"X{name} {comp_type} "
                
                # Add parameters
                for key, value in netlist_dict.items():
                    if key != 'type':
                        netlist += f"{key}={value} "
                netlist += "\n"
                
            except Exception as e:
                print(f"Warning: Could not export {name}: {e}")
                netlist += f"* ERROR: Could not export {name}\n"
                
        # Connections
        netlist += "\n* Connections\n"
        for comp1, comp2, edge_data in self.graph.edges(data=True):
            length = edge_data.get('length', 0)
            loss = edge_data.get('loss', 1.0)
            netlist += f"W{comp1}_{comp2} {comp1} {comp2} length={length} loss={loss}\n"
            
        netlist += "\n.END\n"
        return netlist