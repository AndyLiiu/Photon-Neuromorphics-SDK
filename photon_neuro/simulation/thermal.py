"""
Advanced thermal simulation and analysis for photonic systems.

This module provides comprehensive thermal modeling including:
- Dynamic thermal effects
- Heat dissipation modeling
- Cooling system analysis
- Thermal-optical coupling
- Temperature-dependent component behavior
"""

import numpy as np
import torch
import scipy.sparse
import scipy.sparse.linalg
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

from ..core.exceptions import (
    ThermalError, ValidationError, validate_parameter,
    check_tensor_validity, safe_execution, global_error_recovery
)


class ThermalSimulator:
    """Advanced thermal simulator with dynamic effects and cooling."""
    
    def __init__(self, geometry: Dict[str, Any] = None, 
                 material_properties: Dict[str, Dict[str, float]] = None):
        """
        Initialize thermal simulator.
        
        Args:
            geometry: Dictionary defining chip geometry
            material_properties: Material thermal properties
        """
        try:
            # Default geometry (chip dimensions in meters)
            default_geometry = {
                'length': 10e-3,  # 10 mm
                'width': 5e-3,    # 5 mm  
                'thickness': 300e-6,  # 300 μm
                'grid_points': (50, 25, 10)  # Simulation grid
            }
            self.geometry = geometry or default_geometry
            
            # Validate geometry
            self.length = validate_parameter("length", self.geometry['length'],
                                           expected_type=(int, float), valid_range=(1e-3, 100e-3))
            self.width = validate_parameter("width", self.geometry['width'],
                                          expected_type=(int, float), valid_range=(1e-3, 100e-3))
            self.thickness = validate_parameter("thickness", self.geometry['thickness'],
                                              expected_type=(int, float), valid_range=(100e-6, 1e-3))
            
            # Default material properties
            default_materials = {
                'silicon': {
                    'thermal_conductivity': 150.0,  # W/(m·K)
                    'specific_heat': 703.0,         # J/(kg·K)
                    'density': 2330.0,              # kg/m³
                    'thermal_expansion': 2.6e-6     # 1/K
                },
                'silicon_dioxide': {
                    'thermal_conductivity': 1.4,    # W/(m·K)
                    'specific_heat': 745.0,         # J/(kg·K)
                    'density': 2203.0,              # kg/m³
                    'thermal_expansion': 0.5e-6     # 1/K
                },
                'air': {
                    'thermal_conductivity': 0.026,  # W/(m·K)
                    'specific_heat': 1005.0,        # J/(kg·K)
                    'density': 1.225,               # kg/m³
                    'thermal_expansion': 3.4e-3     # 1/K
                }
            }
            
            self.materials = material_properties or default_materials
            
            # Initialize thermal grid
            self.nx, self.ny, self.nz = self.geometry['grid_points']
            self.dx = self.length / self.nx
            self.dy = self.width / self.ny
            self.dz = self.thickness / self.nz
            
            # Initialize temperature field
            self.ambient_temperature = 25.0  # Celsius
            self.temperature_field = np.full((self.nx, self.ny, self.nz), 
                                           self.ambient_temperature, dtype=np.float64)
            
            # Power dissipation sources
            self.heat_sources = {}
            
            # Cooling system
            self.cooling_system = CoolingSystem()
            
            # Boundary conditions
            self.boundary_conditions = self._setup_boundary_conditions()
            
            # Thermal properties grid
            self.thermal_conductivity = np.full((self.nx, self.ny, self.nz), 
                                              self.materials['silicon']['thermal_conductivity'])
            self.heat_capacity = np.full((self.nx, self.ny, self.nz), 
                                       self.materials['silicon']['specific_heat'] * 
                                       self.materials['silicon']['density'])
            
        except ValidationError as e:
            raise ThermalError(f"Invalid thermal simulator parameters: {e}")
    
    def _setup_boundary_conditions(self) -> Dict[str, Any]:
        """Setup default boundary conditions."""
        return {
            'top': {'type': 'convection', 'h': 10.0, 'T_inf': self.ambient_temperature},
            'bottom': {'type': 'convection', 'h': 1000.0, 'T_inf': self.ambient_temperature},  # Good heat sink
            'sides': {'type': 'convection', 'h': 5.0, 'T_inf': self.ambient_temperature}
        }
    
    def add_heat_source(self, location: Tuple[float, float, float], 
                       power: float, size: Tuple[float, float, float] = None):
        """Add a heat source at specified location."""
        try:
            x, y, z = location
            x = validate_parameter("x", x, expected_type=(int, float), valid_range=(0, self.length))
            y = validate_parameter("y", y, expected_type=(int, float), valid_range=(0, self.width))
            z = validate_parameter("z", z, expected_type=(int, float), valid_range=(0, self.thickness))
            power = validate_parameter("power", power, expected_type=(int, float), valid_range=(0, 10.0))
            
            # Default size
            if size is None:
                size = (self.dx, self.dy, self.dz)
                
            # Convert to grid indices
            i = int(x / self.dx)
            j = int(y / self.dy)
            k = int(z / self.dz)
            
            # Calculate power density (W/m³)
            volume = size[0] * size[1] * size[2]
            power_density = power / volume
            
            self.heat_sources[(i, j, k)] = {
                'power': power,
                'power_density': power_density,
                'size': size,
                'location': location
            }
            
        except ValidationError as e:
            raise ThermalError(f"Invalid heat source parameters: {e}")
    
    @safe_execution(fallback_value=None)
    def solve_steady_state(self) -> np.ndarray:
        """Solve steady-state thermal equation using finite differences."""
        try:
            # Build thermal conductance matrix
            A, b = self._build_thermal_system()
            
            # Solve linear system Ax = b
            temperature_vector = scipy.sparse.linalg.spsolve(A, b)
            
            # Reshape back to 3D grid
            self.temperature_field = temperature_vector.reshape((self.nx, self.ny, self.nz))
            
            # Apply cooling system effects
            if self.cooling_system.enabled:
                self.temperature_field = self.cooling_system.apply_cooling(self.temperature_field)
            
            return self.temperature_field
            
        except Exception as e:
            raise ThermalError(f"Steady-state thermal solution failed: {e}")
    
    def _build_thermal_system(self) -> Tuple[scipy.sparse.csr_matrix, np.ndarray]:
        """Build finite difference matrix for thermal system."""
        try:
            n_points = self.nx * self.ny * self.nz
            
            # Initialize sparse matrix
            A = scipy.sparse.lil_matrix((n_points, n_points))
            b = np.zeros(n_points)
            
            # Build finite difference stencil
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        idx = self._get_linear_index(i, j, k)
                        
                        # Interior points
                        if (0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1):
                            # 7-point stencil for 3D heat equation
                            kappa = self.thermal_conductivity[i, j, k]
                            
                            # Central coefficient
                            A[idx, idx] = -2 * kappa * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2)
                            
                            # Neighboring points
                            A[idx, self._get_linear_index(i+1, j, k)] = kappa / self.dx**2
                            A[idx, self._get_linear_index(i-1, j, k)] = kappa / self.dx**2
                            A[idx, self._get_linear_index(i, j+1, k)] = kappa / self.dy**2
                            A[idx, self._get_linear_index(i, j-1, k)] = kappa / self.dy**2
                            A[idx, self._get_linear_index(i, j, k+1)] = kappa / self.dz**2
                            A[idx, self._get_linear_index(i, j, k-1)] = kappa / self.dz**2
                            
                            # Heat source term
                            if (i, j, k) in self.heat_sources:
                                b[idx] = -self.heat_sources[(i, j, k)]['power_density']
                        
                        # Boundary conditions
                        else:
                            self._apply_boundary_condition(A, b, i, j, k, idx)
            
            return A.tocsr(), b
            
        except Exception as e:
            raise ThermalError(f"Failed to build thermal system: {e}")
    
    def _get_linear_index(self, i: int, j: int, k: int) -> int:
        """Convert 3D indices to linear index."""
        return i * self.ny * self.nz + j * self.nz + k
    
    def _apply_boundary_condition(self, A: scipy.sparse.lil_matrix, b: np.ndarray,
                                i: int, j: int, k: int, idx: int):
        """Apply boundary conditions to the system."""
        try:
            # Determine which boundary
            if k == 0:  # Bottom surface
                bc = self.boundary_conditions['bottom']
            elif k == self.nz - 1:  # Top surface
                bc = self.boundary_conditions['top']
            else:  # Side surfaces
                bc = self.boundary_conditions['sides']
            
            if bc['type'] == 'convection':
                # Convective boundary condition: -k*dT/dn = h*(T - T_inf)
                h = bc['h']
                T_inf = bc['T_inf']
                kappa = self.thermal_conductivity[i, j, k]
                
                if k == 0:  # Bottom
                    A[idx, idx] = kappa/self.dz + h
                    A[idx, self._get_linear_index(i, j, k+1)] = -kappa/self.dz
                    b[idx] = h * T_inf
                elif k == self.nz - 1:  # Top
                    A[idx, idx] = kappa/self.dz + h
                    A[idx, self._get_linear_index(i, j, k-1)] = -kappa/self.dz
                    b[idx] = h * T_inf
                else:  # Sides
                    A[idx, idx] = h
                    b[idx] = h * T_inf
                    
            elif bc['type'] == 'fixed':
                # Fixed temperature boundary condition
                A[idx, idx] = 1.0
                b[idx] = bc['temperature']
                
        except Exception as e:
            raise ThermalError(f"Failed to apply boundary condition: {e}")
    
    @safe_execution(fallback_value=None)
    def solve_transient(self, time_steps: int, dt: float) -> np.ndarray:
        """Solve transient thermal equation."""
        try:
            dt = validate_parameter("dt", dt, expected_type=(int, float), valid_range=(1e-9, 1.0))
            time_steps = validate_parameter("time_steps", time_steps, 
                                          expected_type=int, valid_range=(1, 100000))
            
            # Store temperature evolution
            temperature_history = np.zeros((time_steps, self.nx, self.ny, self.nz))
            
            # Build system matrices for implicit method
            A_steady, b_steady = self._build_thermal_system()
            
            # Mass matrix for transient term
            M = self._build_mass_matrix()
            
            # Combined matrix for implicit Euler: (M/dt - A) * T^{n+1} = M/dt * T^n + b
            A_transient = M/dt - A_steady
            
            for step in range(time_steps):
                # Right-hand side
                T_current = self.temperature_field.flatten()
                rhs = M.dot(T_current)/dt + b_steady
                
                # Solve for next time step
                T_next = scipy.sparse.linalg.spsolve(A_transient, rhs)
                
                # Update temperature field
                self.temperature_field = T_next.reshape((self.nx, self.ny, self.nz))
                
                # Apply cooling system
                if self.cooling_system.enabled:
                    self.temperature_field = self.cooling_system.apply_cooling(self.temperature_field)
                
                # Store result
                temperature_history[step] = self.temperature_field.copy()
            
            return temperature_history
            
        except Exception as e:
            raise ThermalError(f"Transient thermal solution failed: {e}")
    
    def _build_mass_matrix(self) -> scipy.sparse.csr_matrix:
        """Build mass matrix for transient analysis."""
        n_points = self.nx * self.ny * self.nz
        M = scipy.sparse.diags([self.heat_capacity.flatten()], [0], shape=(n_points, n_points))
        return M.tocsr()
    
    def get_component_temperature(self, component_location: Tuple[float, float, float]) -> float:
        """Get temperature at component location."""
        try:
            x, y, z = component_location
            
            # Convert to grid indices
            i = int(x / self.dx)
            j = int(y / self.dy)
            k = int(z / self.dz)
            
            # Bounds checking
            i = max(0, min(i, self.nx - 1))
            j = max(0, min(j, self.ny - 1))
            k = max(0, min(k, self.nz - 1))
            
            return float(self.temperature_field[i, j, k])
            
        except Exception as e:
            raise ThermalError(f"Failed to get component temperature: {e}")
    
    def calculate_thermal_resistance(self, location1: Tuple[float, float, float],
                                   location2: Tuple[float, float, float]) -> float:
        """Calculate thermal resistance between two locations."""
        try:
            x1, y1, z1 = location1
            x2, y2, z2 = location2
            
            # Distance between points
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            
            # Average thermal conductivity along path (simplified)
            kappa_avg = np.mean([
                self.thermal_conductivity[int(x1/self.dx), int(y1/self.dy), int(z1/self.dz)],
                self.thermal_conductivity[int(x2/self.dx), int(y2/self.dy), int(z2/self.dz)]
            ])
            
            # Cross-sectional area (simplified)
            area = min(self.dx * self.dy, self.dy * self.dz, self.dx * self.dz)
            
            # Thermal resistance
            R_th = distance / (kappa_avg * area)
            
            return R_th
            
        except Exception as e:
            raise ThermalError(f"Thermal resistance calculation failed: {e}")
    
    def analyze_hotspots(self, threshold_temperature: float = 85.0) -> List[Dict[str, Any]]:
        """Identify thermal hotspots above threshold temperature."""
        try:
            threshold_temperature = validate_parameter("threshold_temperature", threshold_temperature,
                                                     expected_type=(int, float), valid_range=(0, 200))
            
            hotspots = []
            
            # Find points above threshold
            hot_indices = np.where(self.temperature_field > threshold_temperature)
            
            for idx in range(len(hot_indices[0])):
                i, j, k = hot_indices[0][idx], hot_indices[1][idx], hot_indices[2][idx]
                
                hotspot = {
                    'location': (i * self.dx, j * self.dy, k * self.dz),
                    'temperature': float(self.temperature_field[i, j, k]),
                    'grid_indices': (i, j, k),
                    'severity': 'critical' if self.temperature_field[i, j, k] > 100 else 'warning'
                }
                
                hotspots.append(hotspot)
            
            # Sort by temperature (highest first)
            hotspots.sort(key=lambda x: x['temperature'], reverse=True)
            
            return hotspots
            
        except Exception as e:
            raise ThermalError(f"Hotspot analysis failed: {e}")
    
    def plot_temperature_field(self, slice_plane: str = 'xy', slice_index: int = None,
                             save_path: str = None):
        """Plot temperature field visualization."""
        try:
            if slice_index is None:
                if slice_plane == 'xy':
                    slice_index = self.nz // 2
                elif slice_plane == 'xz':
                    slice_index = self.ny // 2
                elif slice_plane == 'yz':
                    slice_index = self.nx // 2
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if slice_plane == 'xy':
                temp_slice = self.temperature_field[:, :, slice_index].T
                extent = [0, self.length*1000, 0, self.width*1000]  # Convert to mm
                ax.set_xlabel('Length (mm)')
                ax.set_ylabel('Width (mm)')
            elif slice_plane == 'xz':
                temp_slice = self.temperature_field[:, slice_index, :].T
                extent = [0, self.length*1000, 0, self.thickness*1000]
                ax.set_xlabel('Length (mm)')
                ax.set_ylabel('Thickness (mm)')
            elif slice_plane == 'yz':
                temp_slice = self.temperature_field[slice_index, :, :].T
                extent = [0, self.width*1000, 0, self.thickness*1000]
                ax.set_xlabel('Width (mm)')
                ax.set_ylabel('Thickness (mm)')
            
            # Create temperature map
            im = ax.imshow(temp_slice, extent=extent, origin='lower', 
                          cmap='hot', aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Temperature (°C)')
            
            # Add contour lines
            x = np.linspace(extent[0], extent[1], temp_slice.shape[1])
            y = np.linspace(extent[2], extent[3], temp_slice.shape[0])
            X, Y = np.meshgrid(x, y)
            
            contours = ax.contour(X, Y, temp_slice, colors='black', alpha=0.5, linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8)
            
            ax.set_title(f'Temperature Field ({slice_plane} plane, index {slice_index})')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            raise ThermalError(f"Temperature field plotting failed: {e}")


class CoolingSystem:
    """Model for active cooling systems."""
    
    def __init__(self):
        self.enabled = False
        self.cooling_type = "air"  # "air", "liquid", "thermoelectric"
        self.cooling_power = 0.0   # Watts
        self.target_temperature = 25.0  # Celsius
        self.controller_gain = 1.0
        
    def set_air_cooling(self, fan_power: float, air_flow_rate: float):
        """Configure air cooling system."""
        try:
            self.cooling_type = "air"
            self.enabled = True
            
            fan_power = validate_parameter("fan_power", fan_power,
                                         expected_type=(int, float), valid_range=(0, 100))
            air_flow_rate = validate_parameter("air_flow_rate", air_flow_rate,
                                             expected_type=(int, float), valid_range=(0, 1.0))
            
            # Simplified air cooling model
            self.cooling_power = fan_power * 0.1  # 10% efficiency
            self.convection_coefficient = 10 + 50 * air_flow_rate  # W/(m²·K)
            
        except ValidationError as e:
            raise ThermalError(f"Invalid air cooling parameters: {e}")
    
    def set_liquid_cooling(self, pump_power: float, flow_rate: float, 
                          coolant_temperature: float = 20.0):
        """Configure liquid cooling system."""
        try:
            self.cooling_type = "liquid"
            self.enabled = True
            
            pump_power = validate_parameter("pump_power", pump_power,
                                          expected_type=(int, float), valid_range=(0, 1000))
            flow_rate = validate_parameter("flow_rate", flow_rate,
                                         expected_type=(int, float), valid_range=(0, 10e-3))
            
            # Simplified liquid cooling model
            self.cooling_power = pump_power * 0.8  # 80% efficiency
            self.coolant_temperature = coolant_temperature
            self.convection_coefficient = 1000 + 5000 * flow_rate  # W/(m²·K)
            
        except ValidationError as e:
            raise ThermalError(f"Invalid liquid cooling parameters: {e}")
    
    def set_thermoelectric_cooling(self, current: float, voltage: float):
        """Configure thermoelectric (Peltier) cooling."""
        try:
            self.cooling_type = "thermoelectric"
            self.enabled = True
            
            current = validate_parameter("current", current,
                                       expected_type=(int, float), valid_range=(0, 10))
            voltage = validate_parameter("voltage", voltage,
                                       expected_type=(int, float), valid_range=(0, 24))
            
            # Simplified Peltier model
            electrical_power = current * voltage
            
            # Peltier cooling power (simplified)
            seebeck_coefficient = 200e-6  # V/K
            cooling_power = seebeck_coefficient * current * 300  # Rough estimate
            
            self.cooling_power = min(cooling_power, electrical_power * 0.6)  # COP limit
            
        except ValidationError as e:
            raise ThermalError(f"Invalid thermoelectric cooling parameters: {e}")
    
    def apply_cooling(self, temperature_field: np.ndarray) -> np.ndarray:
        """Apply cooling effect to temperature field."""
        try:
            if not self.enabled:
                return temperature_field
            
            cooled_field = temperature_field.copy()
            
            if self.cooling_type == "air":
                # Air cooling primarily affects top surface
                cooling_effect = min(10.0, self.cooling_power / 0.1)  # Max 10°C reduction
                cooled_field[:, :, -1] -= cooling_effect
                
            elif self.cooling_type == "liquid":
                # Liquid cooling affects bottom surface primarily
                cooling_effect = min(30.0, self.cooling_power / 1.0)  # Max 30°C reduction
                cooled_field[:, :, 0] = np.minimum(cooled_field[:, :, 0], 
                                                 self.coolant_temperature + 5)
                
            elif self.cooling_type == "thermoelectric":
                # Thermoelectric cooling affects specific regions
                cooling_effect = min(50.0, self.cooling_power / 2.0)  # Max 50°C reduction
                # Apply to center region
                nx, ny, nz = cooled_field.shape
                x_center, y_center = nx//2, ny//2
                region_size = min(nx//4, ny//4)
                
                cooled_field[x_center-region_size:x_center+region_size,
                           y_center-region_size:y_center+region_size,
                           0] -= cooling_effect
            
            # Ensure temperatures don't go below physically reasonable values
            cooled_field = np.maximum(cooled_field, -20.0)  # -20°C minimum
            
            return cooled_field
            
        except Exception as e:
            raise ThermalError(f"Cooling application failed: {e}")


class ThermalOpticalCoupling:
    """Model coupling between thermal and optical effects."""
    
    def __init__(self):
        self.thermo_optic_coefficients = {
            'silicon': 1.8e-4,      # 1/K
            'silicon_nitride': 2.5e-5,
            'silica': 1e-5,
            'polymer': 1e-4
        }
        
    def calculate_refractive_index_shift(self, material: str, 
                                       temperature_change: float,
                                       base_index: float = None) -> float:
        """Calculate refractive index change due to temperature."""
        try:
            material = validate_parameter("material", material, expected_type=str)
            temperature_change = validate_parameter("temperature_change", temperature_change,
                                                  expected_type=(int, float), valid_range=(-100, 200))
            
            if material not in self.thermo_optic_coefficients:
                raise ValidationError(f"Unknown material: {material}")
            
            dn_dt = self.thermo_optic_coefficients[material]
            delta_n = dn_dt * temperature_change
            
            return delta_n
            
        except Exception as e:
            raise ThermalError(f"Refractive index shift calculation failed: {e}")
    
    def calculate_phase_shift(self, material: str, length: float, 
                            temperature_change: float, wavelength: float = 1550e-9) -> float:
        """Calculate optical phase shift due to temperature change."""
        try:
            length = validate_parameter("length", length,
                                      expected_type=(int, float), valid_range=(1e-6, 1e-2))
            wavelength = validate_parameter("wavelength", wavelength,
                                          expected_type=(int, float), valid_range=(1e-6, 10e-6))
            
            delta_n = self.calculate_refractive_index_shift(material, temperature_change)
            
            # Phase shift calculation
            delta_phi = 2 * np.pi * length * delta_n / wavelength
            
            return delta_phi
            
        except Exception as e:
            raise ThermalError(f"Phase shift calculation failed: {e}")
    
    def calculate_resonance_shift(self, material: str, cavity_length: float,
                                temperature_change: float, 
                                base_resonance: float) -> float:
        """Calculate resonance frequency shift due to temperature."""
        try:
            cavity_length = validate_parameter("cavity_length", cavity_length,
                                             expected_type=(int, float), valid_range=(1e-6, 1e-2))
            base_resonance = validate_parameter("base_resonance", base_resonance,
                                              expected_type=(int, float), valid_range=(1e14, 1e15))
            
            # Relative index change
            dn_dt = self.thermo_optic_coefficients[material]
            relative_change = dn_dt * temperature_change
            
            # Resonance shift
            delta_f = -base_resonance * relative_change
            
            return delta_f
            
        except Exception as e:
            raise ThermalError(f"Resonance shift calculation failed: {e}")


# Thermal analysis utilities
def calculate_thermal_time_constant(volume: float, surface_area: float, 
                                  thermal_conductivity: float, heat_capacity: float) -> float:
    """Calculate thermal time constant for a component."""
    try:
        volume = validate_parameter("volume", volume,
                                  expected_type=(int, float), valid_range=(1e-12, 1e-3))
        surface_area = validate_parameter("surface_area", surface_area,
                                        expected_type=(int, float), valid_range=(1e-8, 1e-2))
        
        thermal_mass = heat_capacity * volume
        thermal_conductance = thermal_conductivity * surface_area / (volume**(1/3))
        
        time_constant = thermal_mass / thermal_conductance
        
        return time_constant
        
    except Exception as e:
        raise ThermalError(f"Thermal time constant calculation failed: {e}")


def estimate_junction_temperature(ambient_temp: float, power_dissipation: float,
                                thermal_resistance: float, 
                                cooling_effectiveness: float = 1.0) -> float:
    """Estimate junction temperature for a component."""
    try:
        ambient_temp = validate_parameter("ambient_temp", ambient_temp,
                                        expected_type=(int, float), valid_range=(-50, 200))
        power_dissipation = validate_parameter("power_dissipation", power_dissipation,
                                             expected_type=(int, float), valid_range=(0, 100))
        thermal_resistance = validate_parameter("thermal_resistance", thermal_resistance,
                                              expected_type=(int, float), valid_range=(1, 10000))
        
        # Junction temperature calculation
        temperature_rise = power_dissipation * thermal_resistance / cooling_effectiveness
        junction_temp = ambient_temp + temperature_rise
        
        return junction_temp
        
    except Exception as e:
        raise ThermalError(f"Junction temperature calculation failed: {e}")