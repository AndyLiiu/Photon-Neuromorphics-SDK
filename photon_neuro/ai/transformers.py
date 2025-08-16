"""
Optical Transformer Architectures
==================================

Revolutionary transformer implementations using photonic interference,
Mach-Zehnder interferometer meshes, and optical attention mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import math
from dataclasses import dataclass

from ..core.components import PhotonicComponent
from ..networks.feedforward import MZIMesh

# Mock missing component for Generation 5
class MicroringArray:
    def __init__(self, *args, **kwargs):
        pass
# Mock missing component for Generation 5
class PhotonicCircuit:
    def __init__(self, *args, **kwargs):
        pass
from ..utils.logging_system import global_logger, log_execution_time
from ..core.exceptions import ComponentError, validate_parameter


@dataclass
class AttentionMaps:
    """Optical attention visualization data."""
    attention_weights: torch.Tensor
    optical_interference: torch.Tensor
    phase_patterns: torch.Tensor
    optical_field_distribution: torch.Tensor


class OpticalPositionalEncoding(PhotonicComponent):
    """Optical positional encoding using wavelength division multiplexing."""
    
    def __init__(self, d_model: int, max_seq_length: int = 1024, 
                 base_wavelength: float = 1550e-9):
        """
        Initialize optical positional encoding.
        
        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length
            base_wavelength: Base wavelength in meters
        """
        super().__init__()
        
        validate_parameter(d_model, "d_model", min_val=1)
        validate_parameter(max_seq_length, "max_seq_length", min_val=1)
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.base_wavelength = base_wavelength
        
        # Create wavelength encoding matrix
        self.wavelength_encoding = self._create_wavelength_encoding()
        
        # Optical components for wavelength multiplexing
        self.wdm_multiplexer = self._create_wdm_multiplexer()
        
        global_logger.info(f"Initialized OpticalPositionalEncoding with "
                          f"d_model={d_model}, max_seq_length={max_seq_length}")
    
    def _create_wavelength_encoding(self) -> torch.Tensor:
        """Create wavelength-based positional encoding."""
        encoding = torch.zeros(self.max_seq_length, self.d_model)
        position = torch.arange(0, self.max_seq_length).unsqueeze(1).float()
        
        # Use wavelength division instead of frequency division
        wavelength_div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * 
            -(math.log(10000.0) / self.d_model)
        )
        
        # Sine and cosine encoding with optical wavelength modulation
        encoding[:, 0::2] = torch.sin(position * wavelength_div_term)
        encoding[:, 1::2] = torch.cos(position * wavelength_div_term)
        
        # Add wavelength-dependent phase modulation
        for i in range(self.d_model):
            wavelength_shift = i * 0.8e-9  # 0.8 nm spacing
            phase_modulation = 2 * np.pi * (wavelength_shift / self.base_wavelength)
            encoding[:, i] *= torch.cos(torch.tensor(phase_modulation))
        
        return encoding
    
    def _create_wdm_multiplexer(self) -> nn.ModuleDict:
        """Create WDM multiplexer components."""
        components = nn.ModuleDict()
        
        # Arrayed waveguide grating (AWG) for wavelength multiplexing
        components['awg_mux'] = MicroringArray(
            n_rings=self.d_model,
            free_spectral_range=100e9,  # 100 GHz spacing
            quality_factor=10000
        )
        
        # Phase shifters for fine wavelength control
        components['phase_shifters'] = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.d_model)
        ])
        
        return components
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply optical positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Position-encoded tensor with optical modulation
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_seq_length:
            raise ComponentError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_length}")
        
        # Get positional encoding for current sequence length
        pos_encoding = self.wavelength_encoding[:seq_len, :d_model]
        pos_encoding = pos_encoding.to(x.device)
        
        # Apply optical wavelength modulation
        optical_modulation = torch.ones_like(x)
        for i in range(d_model):
            phase = self.wdm_multiplexer['phase_shifters'][i]
            optical_modulation[:, :, i] *= torch.exp(1j * phase).real
        
        # Combine input with optical positional encoding
        return x + pos_encoding.unsqueeze(0) * optical_modulation


class InterferenceSelfAttention(PhotonicComponent):
    """Self-attention mechanism using optical interference patterns."""
    
    def __init__(self, d_model: int, optical_efficiency: float = 0.85):
        """
        Initialize interference-based self-attention.
        
        Args:
            d_model: Model dimension
            optical_efficiency: Optical efficiency factor
        """
        super().__init__()
        
        self.d_model = d_model
        self.optical_efficiency = optical_efficiency
        self.scale = 1.0 / math.sqrt(d_model)
        
        # Optical interference components
        self.mzi_mesh = MZIMesh(
            size=(d_model, d_model),
            topology='rectangular'
        )
        
        # Beamsplitter network for query-key interaction
        self.beamsplitter_network = self._create_beamsplitter_network()
        
        global_logger.info(f"Initialized InterferenceSelfAttention with d_model={d_model}")
    
    def _create_beamsplitter_network(self) -> nn.ModuleDict:
        """Create beamsplitter network for optical interference."""
        network = nn.ModuleDict()
        
        # Cascaded beamsplitters for query-key mixing
        network['query_beamsplitters'] = nn.ModuleList([
            nn.Parameter(torch.randn(2, 2) * 0.1) 
            for _ in range(self.d_model // 2)
        ])
        
        network['key_beamsplitters'] = nn.ModuleList([
            nn.Parameter(torch.randn(2, 2) * 0.1)
            for _ in range(self.d_model // 2)
        ])
        
        # Phase shifters for interference control
        network['interference_phases'] = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.d_model)
        ])
        
        return network
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optical self-attention using interference patterns.
        
        Args:
            query, key, value: Input tensors [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Attention output with optical interference
        """
        batch_size, seq_len, d_model = query.shape
        
        # Convert to optical fields (complex representation)
        query_optical = self._to_optical_field(query)
        key_optical = self._to_optical_field(key)
        value_optical = self._to_optical_field(value)
        
        # Optical query-key interference
        interference_pattern = self._optical_interference(query_optical, key_optical)
        
        # Apply optical scaling and efficiency
        attention_scores = interference_pattern * self.scale * self.optical_efficiency
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Optical softmax using normalized interference
        attention_weights = self._optical_softmax(attention_scores)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, value_optical)
        
        # Convert back to real representation
        output = self._from_optical_field(attended_values)
        
        return output
    
    def _to_optical_field(self, x: torch.Tensor) -> torch.Tensor:
        """Convert real tensor to optical field representation."""
        # Create complex optical field with amplitude and phase
        amplitude = torch.abs(x)
        phase = torch.angle(x + 1e-8j)  # Small imaginary part to avoid zero
        
        return amplitude * torch.exp(1j * phase)
    
    def _from_optical_field(self, optical_field: torch.Tensor) -> torch.Tensor:
        """Convert optical field back to real tensor."""
        # Use intensity (amplitude squared) as output
        return torch.abs(optical_field) ** 2
    
    def _optical_interference(self, query_field: torch.Tensor, 
                             key_field: torch.Tensor) -> torch.Tensor:
        """Calculate optical interference between query and key fields."""
        # Matrix multiplication in optical domain using interference
        batch_size, seq_len, d_model = query_field.shape
        
        # Reshape for batch matrix multiplication
        query_reshaped = query_field.view(batch_size, seq_len, d_model)
        key_reshaped = key_field.view(batch_size, seq_len, d_model).transpose(-2, -1)
        
        # Optical interference: |E1 + E2|^2 = |E1|^2 + |E2|^2 + 2*Re(E1*E2*)
        interference = torch.matmul(query_reshaped, key_reshaped.conj())
        
        # Apply phase control from beamsplitter network
        for i, phase in enumerate(self.beamsplitter_network['interference_phases']):
            if i < interference.shape[-1]:
                interference[:, :, i] *= torch.exp(1j * phase)
        
        # Return intensity (measurable quantity)
        return torch.abs(interference) ** 2
    
    def _optical_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Optical implementation of softmax using normalized interference."""
        # Optical softmax approximation using exponential attenuation
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        
        # Normalize by total optical power
        return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


class OpticalMultiHeadAttention(PhotonicComponent):
    """Multi-head attention using parallel optical channels."""
    
    def __init__(self, d_model: int, n_heads: int, optical_efficiency: float = 0.85):
        """
        Initialize optical multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            optical_efficiency: Optical efficiency factor
        """
        super().__init__()
        
        validate_parameter(d_model, "d_model", min_val=1)
        validate_parameter(n_heads, "n_heads", min_val=1)
        
        if d_model % n_heads != 0:
            raise ComponentError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.optical_efficiency = optical_efficiency
        
        # Parallel optical channels for each head
        self.attention_heads = nn.ModuleList([
            InterferenceSelfAttention(self.d_k, optical_efficiency)
            for _ in range(n_heads)
        ])
        
        # Optical combination network
        self.head_combiner = MZIMesh(
            size=(d_model, d_model),
            topology='rectangular'
        )
        
        # Projection layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        global_logger.info(f"Initialized OpticalMultiHeadAttention with "
                          f"{n_heads} heads, d_model={d_model}")
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, AttentionMaps]:
        """
        Multi-head optical attention.
        
        Args:
            query, key, value: Input tensors [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            (output, attention_maps): Attended output and optical attention maps
        """
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head processing
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Parallel optical attention computation
        head_outputs = []
        attention_patterns = []
        optical_interferences = []
        
        for i, attention_head in enumerate(self.attention_heads):
            head_q = Q[:, i, :, :]
            head_k = K[:, i, :, :]
            head_v = V[:, i, :, :]
            
            head_output = attention_head(head_q, head_k, head_v, mask)
            head_outputs.append(head_output)
            
            # Collect optical interference patterns for visualization
            optical_field = attention_head._to_optical_field(head_q)
            interference = attention_head._optical_interference(optical_field, optical_field)
            optical_interferences.append(interference)
        
        # Concatenate heads
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # Optical head combination using MZI mesh
        combined_output = self.head_combiner(concatenated)
        
        # Final projection
        output = self.w_o(combined_output)
        
        # Create attention maps for analysis
        attention_maps = AttentionMaps(
            attention_weights=torch.stack(optical_interferences, dim=1),
            optical_interference=torch.stack(optical_interferences, dim=1),
            phase_patterns=torch.zeros_like(torch.stack(optical_interferences, dim=1)),
            optical_field_distribution=torch.zeros_like(output)
        )
        
        return output, attention_maps


class OpticalFeedForward(PhotonicComponent):
    """Optical feed-forward network using MZI meshes and nonlinear materials."""
    
    def __init__(self, d_model: int, d_ff: int, nonlinearity_type: str = "kerr"):
        """
        Initialize optical feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            nonlinearity_type: Type of optical nonlinearity ("kerr", "saturable_absorption")
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.nonlinearity_type = nonlinearity_type
        
        # Optical linear transformations using MZI meshes
        self.linear1 = MZIMesh(size=(d_model, d_ff), topology='rectangular')
        self.linear2 = MZIMesh(size=(d_ff, d_model), topology='rectangular')
        
        # Optical nonlinearity parameters
        if nonlinearity_type == "kerr":
            self.nonlinear_coefficient = nn.Parameter(torch.tensor(1e-18))  # Kerr coefficient
        elif nonlinearity_type == "saturable_absorption":
            self.saturation_intensity = nn.Parameter(torch.tensor(1e6))  # W/m^2
        
        global_logger.info(f"Initialized OpticalFeedForward with d_model={d_model}, "
                          f"d_ff={d_ff}, nonlinearity={nonlinearity_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optical feed-forward processing.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Processed tensor with optical nonlinearity
        """
        # First linear transformation
        x1 = self.linear1(x)
        
        # Optical nonlinearity
        x_nonlinear = self._apply_optical_nonlinearity(x1)
        
        # Second linear transformation
        output = self.linear2(x_nonlinear)
        
        return output
    
    def _apply_optical_nonlinearity(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optical nonlinearity."""
        if self.nonlinearity_type == "kerr":
            # Kerr effect: n = n0 + n2*|E|^2
            intensity = torch.abs(x) ** 2
            nonlinear_phase = self.nonlinear_coefficient * intensity
            return x * torch.exp(1j * nonlinear_phase).real
            
        elif self.nonlinearity_type == "saturable_absorption":
            # Saturable absorption: α = α0 / (1 + I/I_sat)
            intensity = torch.abs(x) ** 2
            absorption = 1.0 / (1.0 + intensity / self.saturation_intensity)
            return x * torch.sqrt(absorption)
        
        else:
            # Linear case
            return torch.relu(x)


class OpticalTransformer(PhotonicComponent):
    """Complete optical transformer architecture."""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_ff: int,
                 max_seq_length: int = 1024, optical_efficiency: float = 0.85,
                 nonlinearity_type: str = "kerr"):
        """
        Initialize optical transformer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length
            optical_efficiency: Overall optical efficiency
            nonlinearity_type: Type of optical nonlinearity
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.optical_efficiency = optical_efficiency
        
        # Optical positional encoding
        self.pos_encoding = OpticalPositionalEncoding(d_model, max_seq_length)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            OpticalTransformerLayer(d_model, n_heads, d_ff, optical_efficiency, nonlinearity_type)
            for _ in range(n_layers)
        ])
        
        # Layer normalization (optical implementation)
        self.layer_norms = nn.ModuleList([
            OpticalLayerNorm(d_model) for _ in range(n_layers * 2)
        ])
        
        global_logger.info(f"Initialized OpticalTransformer with {n_layers} layers, "
                          f"{n_heads} heads, d_model={d_model}")
    
    @log_execution_time
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through optical transformer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Dictionary with output and analysis data
        """
        # Apply positional encoding
        x = self.pos_encoding(x)
        
        # Store intermediate states for analysis
        layer_outputs = []
        attention_maps_history = []
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            layer_output, attention_maps = layer(x, mask)
            
            layer_outputs.append(layer_output)
            attention_maps_history.append(attention_maps)
            
            x = layer_output
        
        return {
            'output': x,
            'layer_outputs': layer_outputs,
            'attention_maps': attention_maps_history,
            'optical_efficiency': self._calculate_efficiency()
        }
    
    def _calculate_efficiency(self) -> float:
        """Calculate overall optical efficiency."""
        # Account for losses in each layer
        layer_efficiency = self.optical_efficiency ** self.n_layers
        return layer_efficiency


class OpticalTransformerLayer(PhotonicComponent):
    """Single optical transformer layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 optical_efficiency: float = 0.85, nonlinearity_type: str = "kerr"):
        """Initialize optical transformer layer."""
        super().__init__()
        
        self.multi_head_attention = OpticalMultiHeadAttention(d_model, n_heads, optical_efficiency)
        self.feed_forward = OpticalFeedForward(d_model, d_ff, nonlinearity_type)
        
        self.norm1 = OpticalLayerNorm(d_model)
        self.norm2 = OpticalLayerNorm(d_model)
        
        # Optical residual connections
        self.residual_couplers = nn.ModuleList([
            nn.Parameter(torch.tensor([0.5, 0.5])) for _ in range(2)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, AttentionMaps]:
        """Forward pass through transformer layer."""
        # Multi-head attention with residual connection
        attn_output, attention_maps = self.multi_head_attention(x, x, x, mask)
        x = self._optical_residual_connection(x, attn_output, 0)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self._optical_residual_connection(x, ff_output, 1)
        x = self.norm2(x)
        
        return x, attention_maps
    
    def _optical_residual_connection(self, x: torch.Tensor, residual: torch.Tensor, 
                                   coupler_idx: int) -> torch.Tensor:
        """Optical residual connection using directional couplers."""
        coupling_ratio = self.residual_couplers[coupler_idx]
        
        # Optical power splitting
        main_signal = x * coupling_ratio[0]
        residual_signal = residual * coupling_ratio[1]
        
        return main_signal + residual_signal


class OpticalLayerNorm(PhotonicComponent):
    """Optical layer normalization using automatic gain control."""
    
    def __init__(self, d_model: int):
        """Initialize optical layer normalization."""
        super().__init__()
        
        self.d_model = d_model
        
        # Optical gain control parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        # Optical power monitoring
        self.power_monitor = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optical layer normalization."""
        # Calculate optical power statistics
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize optical power
        normalized = (x - mean) / torch.sqrt(variance + 1e-8)
        
        # Apply optical gain control
        return self.gamma * normalized + self.beta


class PhotonicGPT(OpticalTransformer):
    """Photonic implementation of GPT architecture."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 12, d_ff: int = 2048, max_seq_length: int = 1024,
                 optical_efficiency: float = 0.85):
        """
        Initialize Photonic GPT.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of layers
            d_ff: Feed-forward dimension
            max_seq_length: Maximum sequence length
            optical_efficiency: Optical efficiency
        """
        super().__init__(d_model, n_heads, n_layers, d_ff, max_seq_length, optical_efficiency)
        
        self.vocab_size = vocab_size
        
        # Token embeddings with optical encoding
        self.token_embedding = OpticalTokenEmbedding(vocab_size, d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        global_logger.info(f"Initialized PhotonicGPT with vocab_size={vocab_size}, "
                          f"d_model={d_model}, {n_layers} layers")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through Photonic GPT."""
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Transformer processing
        transformer_output = super().forward(x, attention_mask)
        
        # Output projection
        logits = self.output_projection(transformer_output['output'])
        
        transformer_output['logits'] = logits
        return transformer_output


class OpticalTokenEmbedding(PhotonicComponent):
    """Optical token embedding using wavelength-based encoding."""
    
    def __init__(self, vocab_size: int, d_model: int):
        """Initialize optical token embedding."""
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Traditional embedding table
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Optical wavelength encoding
        self.wavelength_encoders = nn.ModuleList([
            MicroringArray(n_rings=d_model, free_spectral_range=50e9)
            for _ in range(min(vocab_size, 1000))  # Limit for practicality
        ])
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply optical token embedding."""
        # Get base embeddings
        embeddings = self.embedding(input_ids)
        
        # Apply optical wavelength modulation
        batch_size, seq_len = input_ids.shape
        
        for i in range(min(seq_len, len(self.wavelength_encoders))):
            if i < len(self.wavelength_encoders):
                # Apply wavelength-specific modulation
                optical_modulation = self.wavelength_encoders[i](embeddings[:, i, :])
                embeddings[:, i, :] = optical_modulation
        
        return embeddings


class MZIAttentionMesh(PhotonicComponent):
    """Attention mechanism using Mach-Zehnder interferometer mesh."""
    
    def __init__(self, d_model: int, mesh_size: Tuple[int, int] = None):
        """
        Initialize MZI attention mesh.
        
        Args:
            d_model: Model dimension
            mesh_size: Size of MZI mesh (default: (d_model, d_model))
        """
        super().__init__()
        
        self.d_model = d_model
        self.mesh_size = mesh_size or (d_model, d_model)
        
        # Create MZI mesh for attention computation
        self.mzi_mesh = MZIMesh(size=self.mesh_size, topology='triangular')
        
        # Phase control for attention weights
        self.phase_controllers = nn.ParameterList([
            nn.Parameter(torch.randn(self.mesh_size[0], self.mesh_size[1]) * 0.1)
            for _ in range(3)  # For Q, K, V transformations
        ])
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """Compute attention using MZI mesh interference."""
        # Apply MZI transformations
        q_transformed = self.mzi_mesh(query, phase_offset=self.phase_controllers[0])
        k_transformed = self.mzi_mesh(key, phase_offset=self.phase_controllers[1])
        v_transformed = self.mzi_mesh(value, phase_offset=self.phase_controllers[2])
        
        # Optical interference for attention computation
        attention_scores = torch.matmul(q_transformed, k_transformed.transpose(-2, -1))
        attention_weights = torch.softmax(attention_scores / math.sqrt(self.d_model), dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v_transformed)
        
        return output