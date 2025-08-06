"""
Spiking photonic neural networks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from ..core import PhotonicComponent, MicroringModulator, Photodetector
from ..core.registry import register_component


@register_component
class PhotonicLIFNeuron(PhotonicComponent):
    """Photonic leaky integrate-and-fire neuron."""
    
    def __init__(self, threshold: float = 1.0, leak_rate: float = 0.1,
                 refractory_period: float = 1e-9):
        super().__init__()
        self.threshold = threshold
        self.leak_rate = leak_rate
        self.refractory_period = refractory_period
        
        # Photonic components
        self.integrator_ring = MicroringModulator(radius=10e-6, quality_factor=5000)
        self.detector = Photodetector(responsivity=1.0, bandwidth=10e9)
        self.spike_generator = MicroringModulator(radius=5e-6)
        
        # State variables
        self.register_buffer('membrane_potential', torch.zeros(1))
        self.register_buffer('last_spike_time', torch.zeros(1))
        self.register_buffer('time_step', torch.zeros(1))
        
    def forward(self, input_current: torch.Tensor, dt: float = 1e-12) -> torch.Tensor:
        """Forward pass for single time step."""
        batch_size = input_current.shape[0]
        
        # Initialize states for batch
        if self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(batch_size, device=input_current.device)
            self.last_spike_time = torch.zeros(batch_size, device=input_current.device)
            
        # Check refractory period
        time_since_spike = self.time_step - self.last_spike_time
        refractory_mask = time_since_spike > self.refractory_period
        
        # Leak membrane potential
        self.membrane_potential *= torch.exp(-dt * self.leak_rate)
        
        # Integrate input (only if not in refractory period)
        self.membrane_potential += input_current * dt * refractory_mask
        
        # Check for spikes
        spike_mask = self.membrane_potential >= self.threshold
        spikes = spike_mask.float()
        
        # Reset membrane potential after spike
        self.membrane_potential[spike_mask] = 0.0
        self.last_spike_time[spike_mask] = self.time_step.expand_as(self.last_spike_time)[spike_mask]
        
        # Update time
        self.time_step += dt
        
        # Generate optical spikes
        optical_spikes = self._generate_optical_spikes(spikes)
        
        return optical_spikes
        
    def _generate_optical_spikes(self, electrical_spikes: torch.Tensor) -> torch.Tensor:
        """Convert electrical spikes to optical pulses."""
        # Drive spike generator modulator
        spike_amplitude = 1.0  # Normalized
        optical_field = torch.zeros_like(electrical_spikes, dtype=torch.complex64)
        
        # Generate optical pulses for each spike
        for i, spike in enumerate(electrical_spikes):
            if spike > 0:
                # Short optical pulse (simplified)
                pulse_duration = 10e-12  # 10 ps
                pulse_power = spike_amplitude
                
                # Gaussian pulse shape
                t_pulse = torch.linspace(-pulse_duration, pulse_duration, 100)
                pulse_shape = torch.exp(-t_pulse**2 / (2 * (pulse_duration/4)**2))
                
                # Take peak value for simplification
                optical_field[i] = pulse_power * torch.max(pulse_shape)
                
        return optical_field
        
    def reset_state(self):
        """Reset neuron state."""
        self.membrane_potential.zero_()
        self.last_spike_time.zero_()
        self.time_step.zero_()
        
    def to_netlist(self) -> dict:
        return {
            "type": "photonic_lif_neuron",
            "threshold": self.threshold,
            "leak_rate": self.leak_rate,
            "refractory_period": self.refractory_period
        }


@register_component
class PhotonicSNN(PhotonicComponent):
    """Photonic spiking neural network."""
    
    def __init__(self, topology: List[int], neuron_model: str = "photonic_lif",
                 synapse_type: str = "microring", timestep: float = 1e-12):
        super().__init__()
        self.topology = topology
        self.neuron_model = neuron_model
        self.synapse_type = synapse_type
        self.timestep = timestep
        
        # Create layers of photonic neurons
        self.layers = nn.ModuleList()
        for i in range(1, len(topology)):
            layer = nn.ModuleList([
                PhotonicLIFNeuron() for _ in range(topology[i])
            ])
            self.layers.append(layer)
            
        # Synaptic weights (microring modulators)
        self.synapses = nn.ModuleList()
        for i in range(len(topology) - 1):
            synapse_layer = nn.ModuleList()
            for j in range(topology[i+1]):
                # Each post-synaptic neuron has modulators for all pre-synaptic connections
                neuron_synapses = nn.ModuleList([
                    MicroringModulator(radius=8e-6, coupling_coefficient=0.05)
                    for _ in range(topology[i])
                ])
                synapse_layer.append(neuron_synapses)
            self.synapses.append(synapse_layer)
            
        # Weight matrices for controlling synapses
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(topology[i+1], topology[i]) * 0.1)
            for i in range(len(topology) - 1)
        ])
        
    def configure_optics(self, waveguide_width: float = 450e-9,
                        microring_radius: float = 5e-6,
                        coupling_gap: float = 200e-9,
                        phase_shifter: str = "thermal"):
        """Configure optical parameters."""
        self.waveguide_width = waveguide_width
        self.microring_radius = microring_radius
        self.coupling_gap = coupling_gap
        self.phase_shifter_type = phase_shifter
        
        # Update all microrings
        for synapse_layer in self.synapses:
            for neuron_synapses in synapse_layer:
                for synapse in neuron_synapses:
                    synapse.radius = microring_radius
                    
    def forward(self, input_spikes: torch.Tensor, n_timesteps: int = 100) -> torch.Tensor:
        """Forward pass through spiking network."""
        batch_size, n_inputs = input_spikes.shape
        
        if n_inputs != self.topology[0]:
            raise ValueError(f"Expected {self.topology[0]} inputs, got {n_inputs}")
            
        # Reset all neuron states
        for layer in self.layers:
            for neuron in layer:
                neuron.reset_state()
                
        # Simulate over time
        output_spikes = []
        current_input = input_spikes
        
        for t in range(n_timesteps):
            layer_outputs = [current_input]
            
            # Process each layer
            for layer_idx, layer in enumerate(self.layers):
                layer_input = layer_outputs[layer_idx]
                layer_output = []
                
                # Process each neuron in layer
                for neuron_idx, neuron in enumerate(layer):
                    # Calculate synaptic input
                    synaptic_current = self._calculate_synaptic_input(
                        layer_input, layer_idx, neuron_idx
                    )
                    
                    # Generate spikes
                    neuron_output = neuron.forward(synaptic_current, self.timestep)
                    layer_output.append(neuron_output)
                    
                layer_outputs.append(torch.stack(layer_output, dim=1))
                
            output_spikes.append(layer_outputs[-1])
            
        # Return spike trains over time
        return torch.stack(output_spikes, dim=2)  # [batch, neurons, time]
        
    def _calculate_synaptic_input(self, pre_spikes: torch.Tensor, 
                                 layer_idx: int, neuron_idx: int) -> torch.Tensor:
        """Calculate synaptic input current for a neuron."""
        batch_size = pre_spikes.shape[0]
        synaptic_current = torch.zeros(batch_size, device=pre_spikes.device)
        
        # Get synapses for this neuron
        neuron_synapses = self.synapses[layer_idx][neuron_idx]
        weights = self.weights[layer_idx][neuron_idx]
        
        # Process each pre-synaptic connection
        for pre_idx, (synapse, weight) in enumerate(zip(neuron_synapses, weights)):
            # Set synapse weight (voltage control)
            synapse.set_drive_voltage(weight.item())
            
            # Apply synaptic transmission
            pre_spike = pre_spikes[:, pre_idx].unsqueeze(-1)
            if pre_spike.dtype != torch.complex64:
                pre_spike = pre_spike.to(torch.complex64)
                
            transmitted = synapse.forward(pre_spike)
            
            # Convert to current (simplified)
            current = torch.abs(transmitted).squeeze()
            synaptic_current += current
            
        return synaptic_current
        
    def train_stdp(self, input_spikes: torch.Tensor, target_spikes: torch.Tensor,
                   learning_rate: float = 0.01, stdp_window: float = 20e-12):
        """Train using spike-timing dependent plasticity."""
        # Forward pass to get actual spikes
        actual_spikes = self.forward(input_spikes)
        
        # Calculate STDP updates for each synapse
        with torch.no_grad():
            for layer_idx, weight_matrix in enumerate(self.weights):
                # Get pre and post spike times
                if layer_idx == 0:
                    pre_spikes = input_spikes
                else:
                    # Previous layer outputs (simplified)
                    pre_spikes = actual_spikes[:, :self.topology[layer_idx], :].mean(dim=2)
                    
                post_spikes = actual_spikes[:, :self.topology[layer_idx+1], :].mean(dim=2)
                
                # STDP rule: strengthen if pre before post, weaken otherwise
                for i in range(weight_matrix.shape[0]):  # post neurons
                    for j in range(weight_matrix.shape[1]):  # pre neurons
                        # Simplified STDP calculation
                        pre_time = pre_spikes[:, j].mean()
                        post_time = post_spikes[:, i].mean()
                        
                        dt = post_time - pre_time
                        
                        if dt > 0 and dt < stdp_window:
                            # Potentiation
                            delta_w = learning_rate * torch.exp(-dt / stdp_window)
                        elif dt < 0 and abs(dt) < stdp_window:
                            # Depression  
                            delta_w = -learning_rate * torch.exp(dt / stdp_window)
                        else:
                            delta_w = 0
                            
                        weight_matrix[i, j] += delta_w
                        
        return torch.mean((actual_spikes - target_spikes)**2).item()
        
    @property 
    def efficiency(self) -> float:
        """Calculate optical power efficiency."""
        total_input_power = 1.0  # Normalized
        total_loss = 0.0
        
        for synapse_layer in self.synapses:
            for neuron_synapses in synapse_layer:
                for synapse in neuron_synapses:
                    total_loss += synapse.get_loss_linear()
                    
        return 1.0 / (1.0 + total_loss)
        
    @property
    def latency_ps(self) -> float:
        """Calculate network latency in picoseconds."""
        # Photonic propagation is very fast
        return len(self.layers) * 1.0  # 1 ps per layer
        
    def to_netlist(self) -> dict:
        return {
            "type": "photonic_snn", 
            "topology": self.topology,
            "neuron_model": self.neuron_model,
            "synapse_type": self.synapse_type,
            "timestep": self.timestep,
            "weights": [w.detach().numpy().tolist() for w in self.weights]
        }