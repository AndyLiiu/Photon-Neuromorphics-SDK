"""
Power budget analysis for photonic systems.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..core import PhotonicComponent


class PowerBudgetAnalyzer:
    """Comprehensive power analysis for photonic neural networks."""
    
    def __init__(self, photonic_model: PhotonicComponent):
        self.model = photonic_model
        self.power_breakdown = {}
        self.thermal_model = ThermalModel()
        
    def analyze(self, input_power_dbm: float = 0, include_thermal: bool = True,
               include_electrical: bool = True) -> 'PowerReport':
        """Perform comprehensive power budget analysis."""
        
        # Convert input power
        input_power_w = 1e-3 * 10**(input_power_dbm / 10)
        
        # Analyze optical power flow
        optical_powers = self._analyze_optical_power(input_power_w)
        
        # Analyze electrical power consumption
        electrical_powers = {}
        if include_electrical:
            electrical_powers = self._analyze_electrical_power()
            
        # Analyze thermal effects
        thermal_effects = {}
        if include_thermal:
            thermal_effects = self._analyze_thermal_effects(electrical_powers)
            
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency(optical_powers, electrical_powers)
        
        return PowerReport(
            optical_powers=optical_powers,
            electrical_powers=electrical_powers,
            thermal_effects=thermal_effects,
            efficiency_metrics=efficiency_metrics
        )
        
    def _analyze_optical_power(self, input_power_w: float) -> Dict[str, float]:
        """Analyze optical power distribution."""
        powers = {
            'input_power': input_power_w,
            'transmitted_power': 0.0,
            'absorbed_power': 0.0,
            'scattered_power': 0.0,
            'total_loss': 0.0
        }
        
        try:
            # Create input field
            input_field = torch.sqrt(torch.tensor(input_power_w, dtype=torch.float32))
            input_field = input_field.to(torch.complex64).unsqueeze(0)
            
            # Process through model
            output_field = self.model.forward(input_field)
            output_power = torch.abs(output_field)**2
            
            powers['transmitted_power'] = output_power.sum().item()
            
            # Calculate losses
            powers['total_loss'] = input_power_w - powers['transmitted_power']
            
            # Estimate loss breakdown (simplified)
            if hasattr(self.model, '_losses_db'):
                total_loss_db = sum(self.model._losses_db.values())
                powers['insertion_loss'] = input_power_w * (1 - 10**(-total_loss_db/10))
            else:
                powers['insertion_loss'] = powers['total_loss'] * 0.8
                
            powers['scattering_loss'] = powers['total_loss'] * 0.1
            powers['absorption_loss'] = powers['total_loss'] * 0.1
            
        except Exception as e:
            print(f"Warning: Optical power analysis failed: {e}")
            
        return powers
        
    def _analyze_electrical_power(self) -> Dict[str, float]:
        """Analyze electrical power consumption."""
        powers = {
            'total_electrical': 0.0,
            'modulator_power': 0.0,
            'detector_power': 0.0,
            'control_power': 0.0,
            'thermal_tuning': 0.0
        }
        
        try:
            # Get power from individual components
            if hasattr(self.model, 'calculate_power_consumption'):
                powers['total_electrical'] = self.model.calculate_power_consumption()
                
            # Break down by component type
            component_powers = self._get_component_powers()
            powers.update(component_powers)
            
        except Exception as e:
            print(f"Warning: Electrical power analysis failed: {e}")
            
        return powers
        
    def _get_component_powers(self) -> Dict[str, float]:
        """Get power consumption by component type."""
        powers = {}
        
        # Traverse model structure to find components
        if hasattr(self.model, 'modules'):
            for name, module in self.model.named_modules():
                if hasattr(module, 'calculate_power_consumption'):
                    try:
                        power = module.calculate_power_consumption()
                        
                        # Categorize by component type
                        if 'modulator' in name.lower():
                            powers.setdefault('modulator_power', 0.0)
                            powers['modulator_power'] += power
                        elif 'detector' in name.lower():
                            powers.setdefault('detector_power', 0.0)
                            powers['detector_power'] += power
                        elif 'phase' in name.lower() or 'thermal' in name.lower():
                            powers.setdefault('thermal_tuning', 0.0)
                            powers['thermal_tuning'] += power
                        else:
                            powers.setdefault('other_power', 0.0)
                            powers['other_power'] += power
                            
                    except Exception as e:
                        print(f"Warning: Failed to get power for {name}: {e}")
                        continue
                        
        return powers
        
    def _analyze_thermal_effects(self, electrical_powers: Dict[str, float]) -> Dict[str, Any]:
        """Analyze thermal effects from power dissipation."""
        effects = {
            'temperature_rise': 0.0,
            'thermal_resistance': {},
            'hotspots': [],
            'thermal_crosstalk': {}
        }
        
        try:
            total_power = electrical_powers.get('total_electrical', 0.0)
            
            # Simple thermal model
            ambient_temp = 25  # °C
            thermal_resistance = 100  # °C/W (chip-to-ambient)
            
            effects['temperature_rise'] = total_power * thermal_resistance
            effects['max_temperature'] = ambient_temp + effects['temperature_rise']
            
            # Identify thermal hotspots
            if electrical_powers.get('thermal_tuning', 0) > 1e-3:  # > 1 mW
                effects['hotspots'].append({
                    'location': 'thermal_phase_shifters',
                    'power': electrical_powers['thermal_tuning'],
                    'temperature': ambient_temp + electrical_powers['thermal_tuning'] * 200
                })
                
            # Thermal crosstalk estimation
            if len(effects['hotspots']) > 1:
                effects['thermal_crosstalk']['coupling_strength'] = 0.1
                effects['thermal_crosstalk']['affected_components'] = ['adjacent_rings']
                
        except Exception as e:
            print(f"Warning: Thermal analysis failed: {e}")
            
        return effects
        
    def _calculate_efficiency(self, optical_powers: Dict[str, float], 
                            electrical_powers: Dict[str, float]) -> Dict[str, float]:
        """Calculate various efficiency metrics."""
        metrics = {}
        
        try:
            input_power = optical_powers.get('input_power', 0.0)
            output_power = optical_powers.get('transmitted_power', 0.0)
            electrical_power = electrical_powers.get('total_electrical', 0.0)
            
            # Optical efficiency (transmission)
            if input_power > 0:
                metrics['optical_efficiency'] = output_power / input_power
                metrics['insertion_loss_db'] = -10 * np.log10(metrics['optical_efficiency'])
            else:
                metrics['optical_efficiency'] = 0.0
                metrics['insertion_loss_db'] = np.inf
                
            # Wall-plug efficiency
            total_input_power = input_power + electrical_power
            if total_input_power > 0:
                metrics['wall_plug_efficiency'] = output_power / total_input_power
            else:
                metrics['wall_plug_efficiency'] = 0.0
                
            # Energy per operation (for neural networks)
            if hasattr(self.model, 'topology') and electrical_power > 0:
                # Estimate operations per second
                clock_freq = 1e9  # 1 GHz
                ops_per_second = clock_freq * sum(self.model.topology[:-1])
                metrics['energy_per_op'] = electrical_power / ops_per_second
                metrics['energy_per_mac'] = metrics['energy_per_op']  # Multiply-accumulate
            
        except Exception as e:
            print(f"Warning: Efficiency calculation failed: {e}")
            
        return metrics
        
    def plot_sankey_diagram(self, power_report: 'PowerReport'):
        """Plot Sankey diagram showing power flow."""
        try:
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Define power flows
            optical = power_report.optical_powers
            electrical = power_report.electrical_powers
            
            input_power = optical.get('input_power', 0) * 1000  # Convert to mW
            output_power = optical.get('transmitted_power', 0) * 1000
            loss_power = optical.get('total_loss', 0) * 1000
            elec_power = electrical.get('total_electrical', 0) * 1000
            
            # Draw boxes for power stages
            boxes = [
                {'name': 'Input\nOptical', 'power': input_power, 'pos': (1, 0.7), 'color': 'lightblue'},
                {'name': 'Device', 'power': input_power, 'pos': (3, 0.7), 'color': 'lightgreen'},
                {'name': 'Output\nOptical', 'power': output_power, 'pos': (5, 0.8), 'color': 'lightblue'},
                {'name': 'Loss\n(Heat)', 'power': loss_power, 'pos': (5, 0.6), 'color': 'salmon'},
                {'name': 'Electrical\nInput', 'power': elec_power, 'pos': (3, 0.3), 'color': 'yellow'},
            ]
            
            # Draw boxes and labels
            for box in boxes:
                if box['power'] > 0:
                    width = 0.3
                    height = box['power'] / max(input_power, 1) * 0.3  # Scale height
                    
                    rect = patches.Rectangle((box['pos'][0] - width/2, box['pos'][1] - height/2),
                                           width, height, linewidth=1, edgecolor='black',
                                           facecolor=box['color'])
                    ax.add_patch(rect)
                    
                    # Add text
                    ax.text(box['pos'][0], box['pos'][1], 
                           f"{box['name']}\n{box['power']:.1f} mW",
                           ha='center', va='center', fontsize=10)
            
            # Draw arrows for power flow
            if input_power > 0:
                ax.arrow(1.3, 0.7, 1.4, 0, head_width=0.02, head_length=0.1, 
                        fc='blue', ec='blue', alpha=0.7)
                        
            if output_power > 0:
                ax.arrow(3.3, 0.75, 1.4, 0.05, head_width=0.02, head_length=0.1,
                        fc='green', ec='green', alpha=0.7)
                        
            if loss_power > 0:
                ax.arrow(3.3, 0.65, 1.4, -0.05, head_width=0.02, head_length=0.1,
                        fc='red', ec='red', alpha=0.7)
                        
            if elec_power > 0:
                ax.arrow(3, 0.45, 0, 0.15, head_width=0.05, head_length=0.02,
                        fc='orange', ec='orange', alpha=0.7)
            
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 1.2)
            ax.set_aspect('equal')
            ax.set_title('Power Flow Diagram', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Warning: Could not create Sankey diagram: {e}")
            self._plot_simple_power_breakdown(power_report)
            
    def _plot_simple_power_breakdown(self, power_report: 'PowerReport'):
        """Simple bar chart of power breakdown."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Optical powers
        opt_labels = []
        opt_values = []
        for key, value in power_report.optical_powers.items():
            if value > 0:
                opt_labels.append(key.replace('_', '\n'))
                opt_values.append(value * 1000)  # Convert to mW
                
        if opt_values:
            ax1.bar(opt_labels, opt_values, color='lightblue', edgecolor='navy')
            ax1.set_ylabel('Optical Power (mW)')
            ax1.set_title('Optical Power Breakdown')
            ax1.tick_params(axis='x', rotation=45)
        
        # Electrical powers  
        elec_labels = []
        elec_values = []
        for key, value in power_report.electrical_powers.items():
            if value > 0:
                elec_labels.append(key.replace('_', '\n'))
                elec_values.append(value * 1000)  # Convert to mW
                
        if elec_values:
            ax2.bar(elec_labels, elec_values, color='yellow', edgecolor='orange')
            ax2.set_ylabel('Electrical Power (mW)')
            ax2.set_title('Electrical Power Breakdown')
            ax2.tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()


class ThermalModel:
    """Simple thermal model for photonic devices."""
    
    def __init__(self):
        self.ambient_temperature = 25  # °C
        self.thermal_conductivity = 150  # W/(m·K) for Silicon
        self.specific_heat = 703  # J/(kg·K) for Silicon
        self.density = 2330  # kg/m³ for Silicon
        
    def calculate_temperature_rise(self, power: float, thermal_resistance: float) -> float:
        """Calculate steady-state temperature rise."""
        return power * thermal_resistance
        
    def calculate_thermal_time_constant(self, volume: float, surface_area: float) -> float:
        """Calculate thermal time constant."""
        mass = volume * self.density
        heat_capacity = mass * self.specific_heat
        thermal_conductance = self.thermal_conductivity * surface_area
        
        return heat_capacity / thermal_conductance


class PowerReport:
    """Container for power analysis results."""
    
    def __init__(self, optical_powers: Dict[str, float] = None,
                 electrical_powers: Dict[str, float] = None,
                 thermal_effects: Dict[str, Any] = None,
                 efficiency_metrics: Dict[str, float] = None):
        self.optical_powers = optical_powers or {}
        self.electrical_powers = electrical_powers or {}
        self.thermal_effects = thermal_effects or {}
        self.efficiency_metrics = efficiency_metrics or {}
        
    @property
    def total_mw(self) -> float:
        """Total power consumption in mW."""
        optical_total = sum(self.optical_powers.values()) * 1000
        electrical_total = sum(self.electrical_powers.values()) * 1000
        return optical_total + electrical_total
        
    @property
    def optical_efficiency(self) -> float:
        """Optical transmission efficiency."""
        return self.efficiency_metrics.get('optical_efficiency', 0.0)
        
    @property
    def wall_plug_efficiency(self) -> float:
        """Overall wall-plug efficiency."""
        return self.efficiency_metrics.get('wall_plug_efficiency', 0.0)
        
    def summary(self) -> str:
        """Generate summary report."""
        summary = f"""
Power Budget Analysis Summary
============================
Total Power: {self.total_mw:.1f} mW
Optical Efficiency: {self.optical_efficiency:.1%}
Wall-plug Efficiency: {self.wall_plug_efficiency:.1%}

Optical Powers:
"""
        for key, value in self.optical_powers.items():
            summary += f"  {key}: {value*1000:.2f} mW\n"
            
        summary += "\nElectrical Powers:\n"
        for key, value in self.electrical_powers.items():
            summary += f"  {key}: {value*1000:.2f} mW\n"
            
        if 'max_temperature' in self.thermal_effects:
            summary += f"\nMax Temperature: {self.thermal_effects['max_temperature']:.1f} °C\n"
            
        return summary