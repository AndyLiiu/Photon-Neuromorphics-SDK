"""
ONNX to photonic neural network compiler.
"""

import torch
import torch.nn as nn
import onnx
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

from ..networks import PhotonicMLP, MZIMesh, MicroringArray
from ..core import PhotonicComponent


class ONNXParser:
    """Parser for ONNX models to photonic implementations."""
    
    def __init__(self, backend: str = "mach_zehnder", wavelength: float = 1550e-9):
        self.backend = backend
        self.wavelength = wavelength
        self.supported_ops = {
            'MatMul', 'Gemm', 'Add', 'Sub', 'Mul', 'Div',
            'Relu', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
            'Conv', 'MaxPool', 'AvgPool', 'GlobalAveragePool',
            'Reshape', 'Transpose', 'Flatten', 'Concat', 'Split',
            'BatchNormalization', 'LayerNormalization',
            'Dropout', 'Identity'
        }
        self.layer_mapping = {}
        
    def parse_model(self, onnx_model: Union[str, onnx.ModelProto]) -> Dict[str, Any]:
        """Parse ONNX model and extract layer information."""
        if isinstance(onnx_model, str):
            model = onnx.load(onnx_model)
        else:
            model = onnx_model
            
        # Validate model
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            warnings.warn(f"ONNX model validation failed: {e}")
            
        # Extract graph information
        graph = model.graph
        layers = []
        
        for i, node in enumerate(graph.node):
            if node.op_type in self.supported_ops:
                layer_info = self._parse_node(node, graph)
                if layer_info:
                    layers.append(layer_info)
            else:
                warnings.warn(f"Unsupported operation: {node.op_type} in node {node.name}")
                
        return {
            'layers': layers,
            'input_shape': self._get_input_shape(graph),
            'output_shape': self._get_output_shape(graph),
            'model_info': {
                'producer_name': model.producer_name,
                'producer_version': model.producer_version,
                'ir_version': model.ir_version
            }
        }
        
    def _parse_node(self, node, graph) -> Optional[Dict[str, Any]]:
        """Parse individual ONNX node."""
        try:
            layer_info = {
                'name': node.name or f"layer_{len(self.layer_mapping)}",
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {}
            }
            
            # Parse attributes
            for attr in node.attribute:
                layer_info['attributes'][attr.name] = self._parse_attribute(attr)
                
            # Get weights if available
            weights = self._get_node_weights(node, graph)
            if weights:
                layer_info['weights'] = weights
                
            return layer_info
            
        except Exception as e:
            warnings.warn(f"Failed to parse node {node.name}: {e}")
            return None
            
    def _parse_attribute(self, attr) -> Any:
        """Parse ONNX attribute."""
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        else:
            return None
            
    def _get_node_weights(self, node, graph) -> Optional[Dict[str, np.ndarray]]:
        """Extract weights for a node."""
        weights = {}
        
        # Look for initializers that match node inputs
        initializers = {init.name: init for init in graph.initializer}
        
        for input_name in node.input:
            if input_name in initializers:
                init = initializers[input_name]
                # Convert ONNX tensor to numpy array
                weights[input_name] = onnx.numpy_helper.to_array(init)
                
        return weights if weights else None
        
    def _get_input_shape(self, graph) -> List[int]:
        """Get model input shape."""
        if graph.input:
            input_tensor = graph.input[0]
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            return shape
        return []
        
    def _get_output_shape(self, graph) -> List[int]:
        """Get model output shape."""
        if graph.output:
            output_tensor = graph.output[0]
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)
            return shape
        return []
        
    def get_photonic_mapping(self, op_type: str) -> Dict[str, Any]:
        """Get photonic implementation mapping for ONNX operator."""
        mappings = {
            # Linear operations - map to MZI meshes
            'MatMul': {'type': 'mzi_mesh', 'supports_complex': True},
            'Gemm': {'type': 'mzi_mesh', 'supports_complex': True, 'has_bias': True},
            
            # Element-wise operations - map to nonlinear elements
            'Add': {'type': 'optical_combiner', 'operation': 'add'},
            'Sub': {'type': 'optical_combiner', 'operation': 'subtract'},
            'Mul': {'type': 'optical_mixer', 'operation': 'multiply'},
            'Div': {'type': 'optical_mixer', 'operation': 'divide'},
            
            # Activation functions - map to nonlinear optical elements
            'Relu': {'type': 'saturable_absorber', 'activation': 'relu'},
            'Sigmoid': {'type': 'saturable_absorber', 'activation': 'sigmoid'},
            'Tanh': {'type': 'saturable_absorber', 'activation': 'tanh'},
            'Softmax': {'type': 'optical_softmax', 'normalization': 'l1'},
            
            # Convolution - map to optical convolution (wavelength-division multiplexing)
            'Conv': {'type': 'optical_conv', 'implementation': 'wdm_based'},
            
            # Pooling operations - map to optical averaging/selection
            'MaxPool': {'type': 'optical_pooling', 'operation': 'max'},
            'AvgPool': {'type': 'optical_pooling', 'operation': 'average'},
            'GlobalAveragePool': {'type': 'optical_pooling', 'operation': 'global_average'},
            
            # Structural operations
            'Reshape': {'type': 'optical_router', 'operation': 'reshape'},
            'Transpose': {'type': 'optical_router', 'operation': 'transpose'},
            'Flatten': {'type': 'optical_router', 'operation': 'flatten'},
            'Concat': {'type': 'optical_combiner', 'operation': 'concat'},
            'Split': {'type': 'optical_splitter', 'operation': 'split'},
            
            # Normalization - map to optical normalization circuits
            'BatchNormalization': {'type': 'optical_normalizer', 'method': 'batch'},
            'LayerNormalization': {'type': 'optical_normalizer', 'method': 'layer'},
            
            # Regularization
            'Dropout': {'type': 'variable_attenuator', 'operation': 'dropout'},
            'Identity': {'type': 'waveguide', 'operation': 'passthrough'},
        }
        
        return mappings.get(op_type, {'type': 'unsupported', 'operation': 'unknown'})
        
    def estimate_photonic_complexity(self, layers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Estimate photonic circuit complexity."""
        complexity = {
            'mzi_count': 0,
            'phase_shifters': 0,
            'nonlinear_elements': 0,
            'detectors': 0,
            'modulators': 0,
            'total_area_mm2': 0.0
        }
        
        for layer in layers:
            op_type = layer['op_type']
            mapping = self.get_photonic_mapping(op_type)
            
            if mapping['type'] == 'mzi_mesh':
                # Estimate MZI count from weight matrix size
                weights = layer.get('weights', {})
                for weight_array in weights.values():
                    if len(weight_array.shape) == 2:
                        rows, cols = weight_array.shape
                        # Each MZI can implement a 2x2 unitary
                        complexity['mzi_count'] += (rows * cols) // 4
                        complexity['phase_shifters'] += rows * cols * 2  # 2 phases per MZI
                        
            elif mapping['type'] in ['saturable_absorber', 'optical_softmax']:
                complexity['nonlinear_elements'] += 1
                
            elif mapping['type'] in ['optical_combiner', 'optical_mixer']:
                complexity['modulators'] += 2  # Input modulators
                
        # Estimate physical area (very rough)
        complexity['total_area_mm2'] = (
            complexity['mzi_count'] * 0.01 +  # 10 µm² per MZI
            complexity['phase_shifters'] * 0.001 +  # 1 µm² per phase shifter
            complexity['nonlinear_elements'] * 0.1  # 100 µm² per nonlinear element
        )
        
        return complexity


def compile_to_photonic(model: Union[nn.Module, str], backend: str = "mach_zehnder",
                        wavelength: float = 1550e-9, loss_db_per_cm: float = 0.1,
                        optimization_level: int = 2) -> PhotonicComponent:
    """Compile PyTorch model or ONNX file to photonic implementation."""
    
    if isinstance(model, str):
        # ONNX file path
        return _compile_onnx_to_photonic(model, backend, wavelength, loss_db_per_cm, optimization_level)
    elif isinstance(model, nn.Module):
        # PyTorch model
        return _compile_pytorch_to_photonic(model, backend, wavelength, loss_db_per_cm, optimization_level)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def _compile_onnx_to_photonic(onnx_path: str, backend: str, wavelength: float,
                             loss_db_per_cm: float, optimization_level: int) -> PhotonicComponent:
    """Compile ONNX model to photonic implementation."""
    
    # Parse ONNX model
    parser = ONNXParser(backend, wavelength)
    model_info = parser.parse_model(onnx_path)
    
    # Extract layer topology
    layer_sizes = _extract_layer_sizes(model_info['layers'])
    
    if not layer_sizes:
        raise ValueError("Could not extract layer topology from ONNX model")
        
    # Create photonic network based on backend
    if backend == "mach_zehnder":
        photonic_model = PhotonicMLP(layer_sizes, activation="relu")
    elif backend == "microring":
        photonic_model = _create_microring_network(layer_sizes)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
        
    # Transfer weights from ONNX model
    _transfer_weights(model_info['layers'], photonic_model)
    
    # Apply optimizations
    if optimization_level > 0:
        photonic_model = _optimize_photonic_model(photonic_model, optimization_level)
        
    return photonic_model


def _compile_pytorch_to_photonic(pytorch_model: nn.Module, backend: str, wavelength: float,
                                loss_db_per_cm: float, optimization_level: int) -> PhotonicComponent:
    """Compile PyTorch model to photonic implementation."""
    
    # Extract layer information from PyTorch model
    layer_sizes = []
    
    for module in pytorch_model.modules():
        if isinstance(module, nn.Linear):
            if not layer_sizes:
                layer_sizes.append(module.in_features)
            layer_sizes.append(module.out_features)
            
    if not layer_sizes:
        raise ValueError("No linear layers found in PyTorch model")
        
    # Create photonic network
    if backend == "mach_zehnder":
        photonic_model = PhotonicMLP(layer_sizes, activation="relu")
    elif backend == "microring":
        photonic_model = _create_microring_network(layer_sizes)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
        
    # Transfer weights from PyTorch model
    _transfer_pytorch_weights(pytorch_model, photonic_model)
    
    # Apply optimizations
    if optimization_level > 0:
        photonic_model = _optimize_photonic_model(photonic_model, optimization_level)
        
    return photonic_model


def _extract_layer_sizes(layers: List[Dict[str, Any]]) -> List[int]:
    """Extract layer sizes from parsed ONNX layers."""
    layer_sizes = []
    
    for layer in layers:
        if layer['op_type'] in ['MatMul', 'Gemm']:
            weights = layer.get('weights', {})
            for weight_name, weight_array in weights.items():
                if len(weight_array.shape) == 2:
                    if not layer_sizes:
                        layer_sizes.append(weight_array.shape[1])  # Input size
                    layer_sizes.append(weight_array.shape[0])      # Output size
                    break
                    
    return layer_sizes


def _create_microring_network(layer_sizes: List[int]) -> PhotonicComponent:
    """Create microring-based photonic network."""
    
    class MicroringNetwork(PhotonicComponent):
        def __init__(self, topology: List[int]):
            super().__init__()
            self.topology = topology
            self.layers = nn.ModuleList()
            
            # Create microring arrays for each layer
            for i in range(len(topology) - 1):
                n_rings = topology[i] * topology[i+1]  # Full connectivity
                ring_array = MicroringArray(
                    n_rings=n_rings,
                    quality_factor=10000,
                    tuning="thermal"
                )
                self.layers.append(ring_array)
                
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            current = x
            for layer in self.layers:
                current = layer.forward(current)
            return current
            
        def to_netlist(self) -> Dict[str, Any]:
            return {
                "type": "microring_network",
                "topology": self.topology,
                "layers": [layer.to_netlist() for layer in self.layers]
            }
            
    return MicroringNetwork(layer_sizes)


def _transfer_weights(onnx_layers: List[Dict[str, Any]], photonic_model: PhotonicComponent):
    """Transfer weights from ONNX layers to photonic model."""
    
    layer_idx = 0
    
    for onnx_layer in onnx_layers:
        if onnx_layer['op_type'] in ['MatMul', 'Gemm'] and 'weights' in onnx_layer:
            weights_dict = onnx_layer['weights']
            
            # Find weight matrix
            weight_matrix = None
            for weight_name, weight_array in weights_dict.items():
                if len(weight_array.shape) == 2:
                    weight_matrix = weight_array
                    break
                    
            if weight_matrix is not None and hasattr(photonic_model, 'layers'):
                if layer_idx < len(photonic_model.layers):
                    photonic_layer = photonic_model.layers[layer_idx]
                    
                    # Convert weights to phase settings
                    if hasattr(photonic_layer, 'decompose'):
                        # MZI mesh - decompose unitary matrix
                        try:
                            # SVD decomposition for non-square matrices
                            U, s, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
                            unitary_matrix = U @ Vt
                            
                            # Convert to torch tensor and decompose
                            unitary_tensor = torch.tensor(unitary_matrix, dtype=torch.complex64)
                            phases = photonic_layer.decompose(unitary_tensor)
                            photonic_layer.set_phases(phases)
                            
                        except Exception as e:
                            warnings.warn(f"Weight transfer failed for layer {layer_idx}: {e}")
                            
                    elif hasattr(photonic_layer, 'encode_weights'):
                        # Microring array
                        photonic_layer.encode_weights(weight_matrix)
                        
                layer_idx += 1


def _transfer_pytorch_weights(pytorch_model: nn.Module, photonic_model: PhotonicComponent):
    """Transfer weights from PyTorch model to photonic model."""
    
    layer_idx = 0
    
    for module in pytorch_model.modules():
        if isinstance(module, nn.Linear) and hasattr(photonic_model, 'layers'):
            if layer_idx < len(photonic_model.layers):
                photonic_layer = photonic_model.layers[layer_idx]
                weight_matrix = module.weight.detach().numpy()
                
                # Transfer weights similar to ONNX case
                if hasattr(photonic_layer, 'decompose'):
                    try:
                        U, s, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
                        unitary_matrix = U @ Vt
                        unitary_tensor = torch.tensor(unitary_matrix, dtype=torch.complex64)
                        phases = photonic_layer.decompose(unitary_tensor)
                        photonic_layer.set_phases(phases)
                    except Exception as e:
                        warnings.warn(f"Weight transfer failed for layer {layer_idx}: {e}")
                        
                elif hasattr(photonic_layer, 'encode_weights'):
                    photonic_layer.encode_weights(weight_matrix)
                    
                layer_idx += 1


def _optimize_photonic_model(photonic_model: PhotonicComponent, optimization_level: int) -> PhotonicComponent:
    """Apply optimizations to photonic model."""
    
    if optimization_level >= 1:
        # Level 1: Basic optimizations
        photonic_model = _apply_loss_compensation(photonic_model)
        
    if optimization_level >= 2:
        # Level 2: Advanced optimizations
        photonic_model = _optimize_phase_settings(photonic_model)
        photonic_model = _minimize_crosstalk(photonic_model)
        
    if optimization_level >= 3:
        # Level 3: Aggressive optimizations
        photonic_model = _layout_optimization(photonic_model)
        
    return photonic_model


def _apply_loss_compensation(photonic_model: PhotonicComponent) -> PhotonicComponent:
    """Apply loss compensation to maintain signal levels."""
    
    # Add gain elements or adjust input power to compensate for losses
    if hasattr(photonic_model, 'layers'):
        total_loss_db = 0
        
        for layer in photonic_model.layers:
            if hasattr(layer, '_losses_db'):
                layer_loss = sum(layer._losses_db.values())
                total_loss_db += layer_loss
                
        # Adjust input power to compensate
        if hasattr(photonic_model, 'input_power_compensation'):
            photonic_model.input_power_compensation = total_loss_db
            
    return photonic_model


def _optimize_phase_settings(photonic_model: PhotonicComponent) -> PhotonicComponent:
    """Optimize phase shifter settings for minimum power consumption."""
    
    if hasattr(photonic_model, 'layers'):
        for layer in photonic_model.layers:
            if hasattr(layer, 'phases'):
                # Minimize phase range to reduce power consumption
                current_phases = layer.phases.data
                
                # Wrap phases to [-π, π] range
                optimized_phases = torch.remainder(current_phases + np.pi, 2*np.pi) - np.pi
                layer.phases.data = optimized_phases
                
    return photonic_model


def _minimize_crosstalk(photonic_model: PhotonicComponent) -> PhotonicComponent:
    """Apply techniques to minimize optical crosstalk."""
    
    # This would involve spacing optimization, isolation improvements, etc.
    # For now, just add crosstalk awareness to the model
    
    if hasattr(photonic_model, 'crosstalk_mitigation'):
        photonic_model.crosstalk_mitigation = True
    else:
        setattr(photonic_model, 'crosstalk_mitigation', True)
        
    return photonic_model


def _layout_optimization(photonic_model: PhotonicComponent) -> PhotonicComponent:
    """Optimize physical layout for minimum area and maximum performance."""
    
    # This would involve detailed physical layout optimization
    # For now, just mark the model as layout-optimized
    
    if hasattr(photonic_model, 'layout_optimized'):
        photonic_model.layout_optimized = True
    else:
        setattr(photonic_model, 'layout_optimized', True)
        
    return photonic_model