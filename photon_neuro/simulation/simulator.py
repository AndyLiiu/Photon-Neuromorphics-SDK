"""
Core photonic simulation engines.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..core import PhotonicComponent
import matplotlib.pyplot as plt


class PhotonicSimulator:
    """Main photonic device and circuit simulator."""
    
    def __init__(self, timestep: float = 1e-15, backend: str = "torch"):
        self.timestep = timestep
        self.backend = backend
        self.components: List[PhotonicComponent] = []
        self.connections: List[Tuple[int, int]] = []
        self.simulation_data = {}
        
    def add_component(self, component: PhotonicComponent) -> int:
        """Add a photonic component to the simulation."""
        self.components.append(component)
        return len(self.components) - 1
        
    def connect(self, output_idx: int, input_idx: int):
        """Connect output of one component to input of another."""
        self.connections.append((output_idx, input_idx))
        
    def run_simulation(self, input_signals: Dict[int, torch.Tensor], 
                      duration: float = 1e-9) -> Dict[int, torch.Tensor]:
        """Run time-domain simulation."""
        n_timesteps = int(duration / self.timestep)
        
        # Initialize signal arrays
        signals = {}
        for i, component in enumerate(self.components):
            signals[i] = torch.zeros(n_timesteps, dtype=torch.complex64)
            
        # Set input signals
        for component_idx, signal in input_signals.items():
            if len(signal) == n_timesteps:
                signals[component_idx] = signal
            else:
                # Pad or truncate to match timesteps
                signals[component_idx][:len(signal)] = signal[:n_timesteps]
                
        # Time evolution
        for t in range(n_timesteps):
            # Process each component
            for i, component in enumerate(self.components):
                # Get input from connections or external input
                if i in input_signals:
                    input_field = signals[i][t:t+1]
                else:
                    # Get input from connected components
                    input_field = torch.zeros(1, dtype=torch.complex64)
                    for output_idx, input_idx in self.connections:
                        if input_idx == i and t > 0:
                            input_field += signals[output_idx][t-1]
                            
                # Process through component
                if torch.abs(input_field).sum() > 0:
                    output_field = component.forward(input_field)
                    signals[i][t] = output_field.squeeze()
                    
        self.simulation_data = signals
        return signals
        
    def frequency_sweep(self, component: PhotonicComponent, 
                       frequency_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform frequency domain sweep of component."""
        wavelengths = 3e8 / frequency_range
        transmission = []
        phase = []
        
        for wavelength in wavelengths:
            # Create monochromatic input
            input_field = torch.ones(1, dtype=torch.complex64)
            
            # Set component wavelength-dependent parameters
            if hasattr(component, 'wavelength'):
                component.wavelength = wavelength
                
            # Get transfer function
            output_field = component.forward(input_field)
            
            transmission.append(torch.abs(output_field).item())
            phase.append(torch.angle(output_field).item())
            
        return np.array(transmission), np.array(phase)
        
    def calculate_group_delay(self, component: PhotonicComponent,
                            frequency_range: np.ndarray) -> np.ndarray:
        """Calculate group delay from phase response."""
        _, phase = self.frequency_sweep(component, frequency_range)
        
        # Unwrap phase
        phase_unwrapped = np.unwrap(phase)
        
        # Calculate group delay: τ_g = -dφ/dω
        df = frequency_range[1] - frequency_range[0]
        group_delay = -np.gradient(phase_unwrapped, df)
        
        return group_delay
        
    def plot_results(self, component_idx: int = 0, plot_type: str = "time"):
        """Plot simulation results."""
        if not self.simulation_data:
            raise ValueError("No simulation data available. Run simulation first.")
            
        signal = self.simulation_data[component_idx]
        time = np.arange(len(signal)) * self.timestep
        
        plt.figure(figsize=(10, 6))
        
        if plot_type == "time":
            plt.subplot(2, 1, 1)
            plt.plot(time * 1e12, torch.abs(signal).numpy(), 'b-', label='Amplitude')
            plt.xlabel('Time (ps)')
            plt.ylabel('Field Amplitude')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2) 
            plt.plot(time * 1e12, torch.angle(signal).numpy(), 'r-', label='Phase')
            plt.xlabel('Time (ps)')
            plt.ylabel('Phase (rad)')
            plt.legend()
            plt.grid(True)
            
        elif plot_type == "frequency":
            # FFT for frequency domain
            fft_signal = torch.fft.fft(signal)
            frequencies = torch.fft.fftfreq(len(signal), self.timestep)
            
            plt.semilogy(frequencies[:len(frequencies)//2], 
                        torch.abs(fft_signal[:len(fft_signal)//2]).numpy())
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()


class FDTDSolver:
    """Finite-difference time-domain electromagnetic solver."""
    
    def __init__(self, grid_size: Tuple[int, int, int], 
                 cell_size: float = 50e-9, dt: float = None):
        self.grid_size = grid_size
        self.cell_size = cell_size
        
        # Courant stability condition
        c = 3e8  # Speed of light
        if dt is None:
            self.dt = cell_size / (c * np.sqrt(3))  # 3D stability
        else:
            self.dt = dt
            
        # Initialize field arrays
        self.Ex = np.zeros(grid_size)
        self.Ey = np.zeros(grid_size) 
        self.Ez = np.zeros(grid_size)
        self.Hx = np.zeros(grid_size)
        self.Hy = np.zeros(grid_size)
        self.Hz = np.zeros(grid_size)
        
        # Material parameters
        self.epsilon = np.ones(grid_size)  # Relative permittivity
        self.mu = np.ones(grid_size)       # Relative permeability
        self.sigma = np.zeros(grid_size)   # Conductivity
        
        # Boundary conditions
        self.boundary_type = "pml"  # Perfectly Matched Layer
        self.pml_thickness = 10     # PML layer thickness
        self._setup_pml()
        
    def set_material(self, region: Tuple[slice, slice, slice], 
                    epsilon_r: float, mu_r: float = 1.0, sigma: float = 0.0):
        """Set material properties in a region."""
        self.epsilon[region] = epsilon_r
        self.mu[region] = mu_r
        self.sigma[region] = sigma
        
    def add_waveguide(self, start: Tuple[int, int, int], 
                     end: Tuple[int, int, int], width: int,
                     epsilon_core: float = 12.25):  # Silicon n=3.5
        """Add a waveguide structure."""
        x1, y1, z1 = start
        x2, y2, z2 = end
        
        # Create waveguide path (simplified rectangular)
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(max(0, y1 - width//2), min(self.grid_size[1], y1 + width//2)):
                for z in range(max(0, z1 - width//2), min(self.grid_size[2], z1 + width//2)):
                    if 0 <= x < self.grid_size[0]:
                        self.epsilon[x, y, z] = epsilon_core
                        
    def add_source(self, position: Tuple[int, int, int], 
                  amplitude: float, frequency: float, pulse_width: float = None):
        """Add electromagnetic source."""
        self.source_pos = position
        self.source_amplitude = amplitude
        self.source_frequency = frequency
        self.source_pulse_width = pulse_width
        
    def step(self, time_step: int):
        """Single FDTD time step."""
        # Update H field (curl of E)
        self._update_H_field()
        
        # Apply boundary conditions to H field
        self._apply_boundary_conditions()
        
        # Update E field (curl of H)
        self._update_E_field()
        
        # Add sources
        if hasattr(self, 'source_pos'):
            self._add_source(time_step)
            
        # Apply boundary conditions to E field
        self._apply_boundary_conditions()
            
    def _update_H_field(self):
        """Update magnetic field components using proper FDTD equations."""
        mu_0 = 4e-7 * np.pi
        dt_over_mu = self.dt / mu_0
        
        # Update Hx: ∂Hx/∂t = (1/μ)[∂Ey/∂z - ∂Ez/∂y]
        self.Hx[:, :-1, :-1] += dt_over_mu * (
            (self.Ey[:, :-1, 1:] - self.Ey[:, :-1, :-1]) / self.cell_size -
            (self.Ez[:, 1:, :-1] - self.Ez[:, :-1, :-1]) / self.cell_size
        ) / self.mu[:, :-1, :-1]
        
        # Update Hy: ∂Hy/∂t = (1/μ)[∂Ez/∂x - ∂Ex/∂z]
        self.Hy[:-1, :, :-1] += dt_over_mu * (
            (self.Ez[1:, :, :-1] - self.Ez[:-1, :, :-1]) / self.cell_size -
            (self.Ex[:-1, :, 1:] - self.Ex[:-1, :, :-1]) / self.cell_size
        ) / self.mu[:-1, :, :-1]
        
        # Update Hz: ∂Hz/∂t = (1/μ)[∂Ex/∂y - ∂Ey/∂x]
        self.Hz[:-1, :-1, :] += dt_over_mu * (
            (self.Ex[:-1, 1:, :] - self.Ex[:-1, :-1, :]) / self.cell_size -
            (self.Ey[1:, :-1, :] - self.Ey[:-1, :-1, :]) / self.cell_size
        ) / self.mu[:-1, :-1, :]
        
    def _update_E_field(self):
        """Update electric field components using proper FDTD equations."""
        epsilon_0 = 8.854e-12
        dt_over_epsilon = self.dt / epsilon_0
        
        # Update Ex: ∂Ex/∂t = (1/ε)[∂Hz/∂y - ∂Hy/∂z] - σ*Ex/ε
        self.Ex[1:-1, :, 1:-1] += dt_over_epsilon * (
            (self.Hz[1:-1, 1:, 1:-1] - self.Hz[1:-1, :-1, 1:-1]) / self.cell_size -
            (self.Hy[1:-1, :, 1:-1] - self.Hy[1:-1, :, :-2]) / self.cell_size
        ) / self.epsilon[1:-1, :, 1:-1] - \
        self.sigma[1:-1, :, 1:-1] * self.Ex[1:-1, :, 1:-1] * dt_over_epsilon / self.epsilon[1:-1, :, 1:-1]
        
        # Update Ey: ∂Ey/∂t = (1/ε)[∂Hx/∂z - ∂Hz/∂x] - σ*Ey/ε
        self.Ey[:, 1:-1, 1:-1] += dt_over_epsilon * (
            (self.Hx[:, 1:-1, 1:-1] - self.Hx[:, 1:-1, :-2]) / self.cell_size -
            (self.Hz[1:, 1:-1, 1:-1] - self.Hz[:-1, 1:-1, 1:-1]) / self.cell_size
        ) / self.epsilon[:, 1:-1, 1:-1] - \
        self.sigma[:, 1:-1, 1:-1] * self.Ey[:, 1:-1, 1:-1] * dt_over_epsilon / self.epsilon[:, 1:-1, 1:-1]
        
        # Update Ez: ∂Ez/∂t = (1/ε)[∂Hy/∂x - ∂Hx/∂y] - σ*Ez/ε
        self.Ez[1:-1, 1:-1, :] += dt_over_epsilon * (
            (self.Hy[1:-1, 1:-1, :] - self.Hy[:-2, 1:-1, :]) / self.cell_size -
            (self.Hx[1:-1, 1:-1, :] - self.Hx[1:-1, :-2, :]) / self.cell_size
        ) / self.epsilon[1:-1, 1:-1, :] - \
        self.sigma[1:-1, 1:-1, :] * self.Ez[1:-1, 1:-1, :] * dt_over_epsilon / self.epsilon[1:-1, 1:-1, :]
        
    def _add_source(self, time_step: int):
        """Add electromagnetic source at specified position."""
        x, y, z = self.source_pos
        t = time_step * self.dt
        
        if self.source_pulse_width:
            # Gaussian pulse
            pulse = np.exp(-((t - 2*self.source_pulse_width)**2) / 
                          (2*self.source_pulse_width**2))
        else:
            # Continuous wave
            pulse = 1.0
            
        # Add sinusoidal source
        source_value = self.source_amplitude * pulse * np.sin(2 * np.pi * self.source_frequency * t)
        
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
            self.Ez[x, y, z] += source_value
            
    def run_simulation(self, n_timesteps: int) -> Dict[str, np.ndarray]:
        """Run full FDTD simulation."""
        # Storage for field evolution
        field_evolution = {
            'Ex': np.zeros((n_timesteps,) + self.grid_size),
            'Ey': np.zeros((n_timesteps,) + self.grid_size), 
            'Ez': np.zeros((n_timesteps,) + self.grid_size),
        }
        
        for t in range(n_timesteps):
            self.step(t)
            
            # Store fields
            field_evolution['Ex'][t] = self.Ex.copy()
            field_evolution['Ey'][t] = self.Ey.copy()
            field_evolution['Ez'][t] = self.Ez.copy()
            
        return field_evolution
        
    def calculate_transmission(self, monitor_positions: List[Tuple[int, int, int]]) -> List[float]:
        """Calculate transmission at monitor positions."""
        transmissions = []
        
        for pos in monitor_positions:
            x, y, z = pos
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
                # Power flow (Poynting vector magnitude)
                power = (self.Ey[x, y, z]**2 + self.Ez[x, y, z]**2) / 2
                transmissions.append(power)
            else:
                transmissions.append(0.0)
                
        return transmissions
        
    def _setup_pml(self):
        """Setup Perfectly Matched Layer boundary conditions."""
        # PML absorption coefficient
        sigma_max = 0.8 * 3 / (self.cell_size * np.sqrt(1.0))  # Optimal PML conductivity
        
        # Create PML conductivity profiles
        self.sigma_pml_x = np.zeros(self.grid_size)
        self.sigma_pml_y = np.zeros(self.grid_size)
        self.sigma_pml_z = np.zeros(self.grid_size)
        
        # Apply PML to boundaries
        for i in range(self.pml_thickness):
            # Left and right boundaries (x-direction)
            sigma_val = sigma_max * ((self.pml_thickness - i) / self.pml_thickness) ** 3
            self.sigma_pml_x[i, :, :] = sigma_val
            self.sigma_pml_x[-1-i, :, :] = sigma_val
            
            # Front and back boundaries (y-direction)  
            self.sigma_pml_y[:, i, :] = sigma_val
            self.sigma_pml_y[:, -1-i, :] = sigma_val
            
            # Top and bottom boundaries (z-direction)
            self.sigma_pml_z[:, :, i] = sigma_val
            self.sigma_pml_z[:, :, -1-i] = sigma_val
            
    def _apply_boundary_conditions(self):
        """Apply boundary conditions to electromagnetic fields."""
        if self.boundary_type == "pml":
            # Apply PML absorption to fields
            self._apply_pml_absorption()
        elif self.boundary_type == "periodic":
            # Periodic boundary conditions
            self.Ex[0, :, :] = self.Ex[-1, :, :]
            self.Ey[:, 0, :] = self.Ey[:, -1, :]
            self.Ez[:, :, 0] = self.Ez[:, :, -1]
        elif self.boundary_type == "absorbing":
            # Simple absorbing boundaries (Mur condition)
            self._apply_mur_boundary()
        else:
            # Default: zero boundaries
            self.Ex[0, :, :] = 0; self.Ex[-1, :, :] = 0
            self.Ey[:, 0, :] = 0; self.Ey[:, -1, :] = 0
            self.Ez[:, :, 0] = 0; self.Ez[:, :, -1] = 0
            
    def _apply_pml_absorption(self):
        """Apply PML absorption to fields."""
        # Apply exponential decay based on PML conductivity
        decay_factor_x = np.exp(-self.sigma_pml_x * self.dt / (8.854e-12))
        decay_factor_y = np.exp(-self.sigma_pml_y * self.dt / (8.854e-12))
        decay_factor_z = np.exp(-self.sigma_pml_z * self.dt / (8.854e-12))
        
        # Apply to E fields
        self.Ex *= decay_factor_x
        self.Ey *= decay_factor_y
        self.Ez *= decay_factor_z
        
        # Apply to H fields with same factors
        self.Hx *= decay_factor_x
        self.Hy *= decay_factor_y
        self.Hz *= decay_factor_z
        
    def _apply_mur_boundary(self):
        """Apply Mur absorbing boundary condition (first-order)."""
        c = 3e8
        mur_coeff = (c * self.dt - self.cell_size) / (c * self.dt + self.cell_size)
        
        # Simple Mur condition for Ez field at x boundaries
        self.Ez[0, 1:-1, 1:-1] = self.Ez[1, 1:-1, 1:-1] + mur_coeff * (
            self.Ez[0, 1:-1, 1:-1] - self.Ez[1, 1:-1, 1:-1]
        )
        self.Ez[-1, 1:-1, 1:-1] = self.Ez[-2, 1:-1, 1:-1] + mur_coeff * (
            self.Ez[-1, 1:-1, 1:-1] - self.Ez[-2, 1:-1, 1:-1]
        )