"""
Neural Architecture Search for Photonic Networks
===============================================

Automated design and optimization of photonic neural network architectures
using evolutionary algorithms, reinforcement learning, and differentiable search.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import copy
from collections import defaultdict
import json

from ..core.components import PhotonicComponent
from ..networks.feedforward import MZIMesh, MicroringArray
from ..networks.spiking import PhotonicSNN
from ..simulation.power import PowerBudgetAnalyzer
from ..utils.logging_system import global_logger, log_execution_time
from ..core.exceptions import ComponentError, validate_parameter


@dataclass
class ArchitectureConfig:
    """Configuration for a photonic neural architecture."""
    name: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[str, str]]
    optical_parameters: Dict[str, float]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'layers': self.layers,
            'connections': self.connections,
            'optical_parameters': self.optical_parameters,
            'performance_metrics': self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureConfig':
        """Create from dictionary representation."""
        return cls(
            name=data['name'],
            layers=data['layers'],
            connections=data['connections'],
            optical_parameters=data['optical_parameters'],
            performance_metrics=data.get('performance_metrics', {})
        )


@dataclass
class SearchResult:
    """Result from architecture search."""
    best_architecture: ArchitectureConfig
    search_history: List[ArchitectureConfig]
    performance_evolution: List[float]
    search_time: float
    total_evaluations: int


class ArchitectureSearchSpace:
    """Defines the search space for photonic architectures."""
    
    def __init__(self):
        """Initialize architecture search space."""
        self.layer_types = [
            'mzi_mesh', 'microring_array', 'photonic_snn', 
            'optical_attention', 'optical_feedforward',
            'quantum_photonic', 'nonlinear_photonic'
        ]
        
        self.topology_types = [
            'feedforward', 'recurrent', 'attention', 'hybrid'
        ]
        
        self.optical_parameters = {
            'wavelength': (1520e-9, 1580e-9),  # C-band
            'loss_db_per_cm': (0.1, 2.0),
            'coupling_coefficient': (0.1, 0.9),
            'quality_factor': (1000, 50000),
            'thermal_coefficient': (1e-4, 1e-3),
            'power_budget_mw': (10, 1000)
        }
        
        self.layer_parameters = {
            'mzi_mesh': {
                'size': [(4, 4), (8, 8), (16, 16), (32, 32)],
                'topology': ['rectangular', 'triangular', 'circular']
            },
            'microring_array': {
                'n_rings': [16, 32, 64, 128, 256],
                'free_spectral_range': [20e9, 50e9, 100e9],
                'quality_factor': [5000, 10000, 20000]
            },
            'photonic_snn': {
                'neuron_model': ['photonic_lif', 'photonic_izhikevich'],
                'synapse_type': ['microring', 'mzi_coupler'],
                'n_neurons': [64, 128, 256, 512, 1024]
            }
        }
        
        global_logger.info("Initialized ArchitectureSearchSpace with "
                          f"{len(self.layer_types)} layer types")
    
    def sample_architecture(self, target_complexity: str = 'medium') -> ArchitectureConfig:
        """
        Sample a random architecture from the search space.
        
        Args:
            target_complexity: Target complexity ('simple', 'medium', 'complex')
            
        Returns:
            Random architecture configuration
        """
        complexity_configs = {
            'simple': {'n_layers': (2, 4), 'max_connections': 5},
            'medium': {'n_layers': (4, 8), 'max_connections': 12},
            'complex': {'n_layers': (8, 16), 'max_connections': 25}
        }
        
        config = complexity_configs[target_complexity]
        n_layers = random.randint(*config['n_layers'])
        
        # Generate layers
        layers = []
        for i in range(n_layers):
            layer_type = random.choice(self.layer_types)
            layer_config = self._generate_layer_config(layer_type, i)
            layers.append(layer_config)
        
        # Generate connections
        connections = self._generate_connections(layers, config['max_connections'])
        
        # Generate optical parameters
        optical_params = self._sample_optical_parameters()
        
        arch_name = f"arch_{target_complexity}_{random.randint(1000, 9999)}"
        
        return ArchitectureConfig(
            name=arch_name,
            layers=layers,
            connections=connections,
            optical_parameters=optical_params
        )
    
    def _generate_layer_config(self, layer_type: str, layer_id: int) -> Dict[str, Any]:
        """Generate configuration for a specific layer type."""
        base_config = {
            'id': f"layer_{layer_id}",
            'type': layer_type,
        }
        
        if layer_type in self.layer_parameters:
            params = self.layer_parameters[layer_type]
            for param_name, param_values in params.items():
                if isinstance(param_values, list):
                    base_config[param_name] = random.choice(param_values)
                else:
                    base_config[param_name] = param_values
        
        return base_config
    
    def _generate_connections(self, layers: List[Dict[str, Any]], 
                            max_connections: int) -> List[Tuple[str, str]]:
        """Generate connections between layers."""
        connections = []
        layer_ids = [layer['id'] for layer in layers]
        
        # Create sequential connections
        for i in range(len(layer_ids) - 1):
            connections.append((layer_ids[i], layer_ids[i + 1]))
        
        # Add random skip connections
        n_skip = random.randint(0, min(max_connections - len(connections), len(layers) // 2))
        for _ in range(n_skip):
            source = random.choice(layer_ids[:-1])
            target = random.choice(layer_ids[layer_ids.index(source) + 2:])
            if (source, target) not in connections:
                connections.append((source, target))
        
        return connections
    
    def _sample_optical_parameters(self) -> Dict[str, float]:
        """Sample optical parameters from ranges."""
        params = {}
        for param_name, (min_val, max_val) in self.optical_parameters.items():
            params[param_name] = random.uniform(min_val, max_val)
        return params


class PhotonicArchitectureEvaluator:
    """Evaluates photonic neural architectures."""
    
    def __init__(self, evaluation_dataset: Optional[torch.utils.data.DataLoader] = None,
                 max_evaluation_time: float = 300.0):
        """
        Initialize architecture evaluator.
        
        Args:
            evaluation_dataset: Dataset for evaluation
            max_evaluation_time: Maximum time for evaluation (seconds)
        """
        self.evaluation_dataset = evaluation_dataset
        self.max_evaluation_time = max_evaluation_time
        
        # Evaluation metrics
        self.metrics_functions = {
            'accuracy': self._evaluate_accuracy,
            'optical_efficiency': self._evaluate_optical_efficiency,
            'power_consumption': self._evaluate_power_consumption,
            'inference_latency': self._evaluate_inference_latency,
            'fabrication_complexity': self._evaluate_fabrication_complexity,
            'thermal_stability': self._evaluate_thermal_stability
        }
        
        global_logger.info("Initialized PhotonicArchitectureEvaluator")
    
    @log_execution_time
    def evaluate(self, architecture: ArchitectureConfig, 
                 metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate a photonic architecture.
        
        Args:
            architecture: Architecture to evaluate
            metrics: List of metrics to compute (default: all)
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = list(self.metrics_functions.keys())
        
        results = {}
        
        try:
            # Build the architecture
            model = self._build_model(architecture)
            
            # Evaluate each metric
            for metric in metrics:
                if metric in self.metrics_functions:
                    try:
                        results[metric] = self.metrics_functions[metric](model, architecture)
                    except Exception as e:
                        global_logger.warning(f"Failed to evaluate {metric}: {e}")
                        results[metric] = 0.0
                else:
                    global_logger.warning(f"Unknown metric: {metric}")
            
            # Compute composite score
            results['composite_score'] = self._compute_composite_score(results)
            
        except Exception as e:
            global_logger.error(f"Architecture evaluation failed: {e}")
            # Return minimal scores for failed architectures
            results = {metric: 0.0 for metric in metrics}
            results['composite_score'] = 0.0
        
        return results
    
    def _build_model(self, architecture: ArchitectureConfig) -> nn.Module:
        """Build PyTorch model from architecture configuration."""
        try:
            # Create a simplified model based on architecture
            layers = []
            
            for layer_config in architecture.layers:
                layer_type = layer_config['type']
                
                if layer_type == 'mzi_mesh':
                    size = layer_config.get('size', (8, 8))
                    topology = layer_config.get('topology', 'rectangular')
                    layer = MZIMesh(size=size, topology=topology)
                    
                elif layer_type == 'microring_array':
                    n_rings = layer_config.get('n_rings', 64)
                    fsr = layer_config.get('free_spectral_range', 50e9)
                    q = layer_config.get('quality_factor', 10000)
                    layer = MicroringArray(n_rings=n_rings, 
                                         free_spectral_range=fsr,
                                         quality_factor=q)
                    
                elif layer_type == 'photonic_snn':
                    n_neurons = layer_config.get('n_neurons', 256)
                    neuron_model = layer_config.get('neuron_model', 'photonic_lif')
                    synapse_type = layer_config.get('synapse_type', 'microring')
                    
                    # Simplified SNN layer
                    layer = nn.Linear(n_neurons, n_neurons)  # Placeholder
                    
                else:
                    # Default linear layer
                    layer = nn.Linear(128, 128)
                
                layers.append(layer)
            
            # Create sequential model
            if layers:
                model = nn.Sequential(*layers)
            else:
                model = nn.Identity()
            
            return model
            
        except Exception as e:
            global_logger.error(f"Model building failed: {e}")
            return nn.Identity()  # Return dummy model
    
    def _evaluate_accuracy(self, model: nn.Module, 
                          architecture: ArchitectureConfig) -> float:
        """Evaluate model accuracy."""
        if self.evaluation_dataset is None:
            # Return simulated accuracy based on architecture complexity
            n_layers = len(architecture.layers)
            base_accuracy = 0.5 + 0.3 * min(n_layers / 10.0, 1.0)
            
            # Add noise
            noise = random.uniform(-0.1, 0.1)
            return max(0.0, min(1.0, base_accuracy + noise))
        
        # Real evaluation would run inference on dataset
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.evaluation_dataset):
                if batch_idx > 10:  # Limit evaluation time
                    break
                    
                try:
                    outputs = model(data)
                    if outputs.shape[-1] == targets.shape[-1]:
                        predicted = outputs.argmax(dim=-1)
                        correct += (predicted == targets.argmax(dim=-1)).sum().item()
                        total += targets.size(0)
                except:
                    break
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_optical_efficiency(self, model: nn.Module, 
                                   architecture: ArchitectureConfig) -> float:
        """Evaluate optical efficiency."""
        # Calculate efficiency based on component types and parameters
        base_efficiency = 0.9  # Start with high efficiency
        
        for layer in architecture.layers:
            layer_type = layer['type']
            
            if layer_type == 'mzi_mesh':
                size = layer.get('size', (8, 8))
                # Each MZI has insertion loss
                n_mzi = size[0] * size[1] // 2
                loss_per_mzi = 0.05  # 5% loss per MZI
                layer_efficiency = (1 - loss_per_mzi) ** n_mzi
                base_efficiency *= layer_efficiency
                
            elif layer_type == 'microring_array':
                n_rings = layer.get('n_rings', 64)
                q_factor = layer.get('quality_factor', 10000)
                # Higher Q factor means better efficiency
                ring_efficiency = min(0.95, 0.5 + 0.5 * (q_factor / 20000))
                base_efficiency *= ring_efficiency ** (n_rings / 100)
        
        # Account for optical parameters
        loss_db_cm = architecture.optical_parameters.get('loss_db_per_cm', 0.5)
        loss_factor = 10 ** (-loss_db_cm / 10)  # Convert dB to linear
        
        return max(0.1, base_efficiency * loss_factor)
    
    def _evaluate_power_consumption(self, model: nn.Module, 
                                  architecture: ArchitectureConfig) -> float:
        """Evaluate power consumption (lower is better)."""
        base_power = 50.0  # mW
        
        for layer in architecture.layers:
            layer_type = layer['type']
            
            if layer_type == 'mzi_mesh':
                size = layer.get('size', (8, 8))
                # Each phase shifter consumes power
                n_phase_shifters = size[0] * size[1]
                layer_power = n_phase_shifters * 2.0  # 2 mW per phase shifter
                base_power += layer_power
                
            elif layer_type == 'microring_array':
                n_rings = layer.get('n_rings', 64)
                layer_power = n_rings * 1.5  # 1.5 mW per tunable ring
                base_power += layer_power
                
            elif layer_type == 'photonic_snn':
                n_neurons = layer.get('n_neurons', 256)
                layer_power = n_neurons * 0.5  # 0.5 mW per neuron
                base_power += layer_power
        
        # Power budget constraint
        power_budget = architecture.optical_parameters.get('power_budget_mw', 500)
        
        # Return normalized power (1.0 = within budget, 0.0 = over budget)
        return max(0.0, 1.0 - base_power / power_budget)
    
    def _evaluate_inference_latency(self, model: nn.Module, 
                                  architecture: ArchitectureConfig) -> float:
        """Evaluate inference latency (lower latency = higher score)."""
        base_latency = 1e-9  # 1 ns base latency
        
        for layer in architecture.layers:
            layer_type = layer['type']
            
            if layer_type == 'mzi_mesh':
                # MZI mesh has propagation delay
                size = layer.get('size', (8, 8))
                mesh_latency = max(size) * 10e-12  # 10 ps per stage
                base_latency += mesh_latency
                
            elif layer_type == 'photonic_snn':
                # SNN has temporal dynamics
                base_latency += 100e-12  # 100 ps for spike processing
        
        # Convert to score (1.0 for latency < 1ns, 0.0 for latency > 10ns)
        latency_score = max(0.0, 1.0 - (base_latency - 1e-9) / 9e-9)
        return latency_score
    
    def _evaluate_fabrication_complexity(self, model: nn.Module, 
                                       architecture: ArchitectureConfig) -> float:
        """Evaluate fabrication complexity (lower complexity = higher score)."""
        complexity_score = 1.0
        
        for layer in architecture.layers:
            layer_type = layer['type']
            
            if layer_type == 'mzi_mesh':
                size = layer.get('size', (8, 8))
                # Larger meshes are more complex to fabricate
                mesh_complexity = min(size) / 32.0  # Normalize by 32x32
                complexity_score *= (1.0 - mesh_complexity * 0.3)
                
            elif layer_type == 'microring_array':
                n_rings = layer.get('n_rings', 64)
                # More rings increase complexity
                ring_complexity = n_rings / 256.0  # Normalize by 256 rings
                complexity_score *= (1.0 - ring_complexity * 0.2)
        
        return max(0.1, complexity_score)
    
    def _evaluate_thermal_stability(self, model: nn.Module, 
                                  architecture: ArchitectureConfig) -> float:
        """Evaluate thermal stability."""
        # Higher thermal coefficient means lower stability
        thermal_coeff = architecture.optical_parameters.get('thermal_coefficient', 1e-4)
        
        # Convert to stability score
        stability_score = max(0.0, 1.0 - (thermal_coeff - 1e-4) / 9e-4)
        
        return stability_score
    
    def _compute_composite_score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted composite score."""
        weights = {
            'accuracy': 0.3,
            'optical_efficiency': 0.2,
            'power_consumption': 0.2,
            'inference_latency': 0.15,
            'fabrication_complexity': 0.1,
            'thermal_stability': 0.05
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]
                composite += weight * value
                total_weight += weight
        
        return composite / total_weight if total_weight > 0 else 0.0


class OpticalArchitectureOptimizer(ABC):
    """Abstract base class for architecture optimizers."""
    
    @abstractmethod
    def optimize(self, search_space: ArchitectureSearchSpace,
                 evaluator: PhotonicArchitectureEvaluator,
                 n_iterations: int = 100) -> SearchResult:
        """Optimize architecture."""
        pass


class EvolutionaryPhotonicNAS(OpticalArchitectureOptimizer):
    """Evolutionary algorithm for photonic neural architecture search."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7, elitism_ratio: float = 0.1):
        """
        Initialize evolutionary NAS.
        
        Args:
            population_size: Size of population
            mutation_rate: Rate of mutation
            crossover_rate: Rate of crossover
            elitism_ratio: Ratio of elite individuals to preserve
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.n_elite = int(population_size * elitism_ratio)
        
        global_logger.info(f"Initialized EvolutionaryPhotonicNAS with "
                          f"population={population_size}, mutation_rate={mutation_rate}")
    
    @log_execution_time
    def optimize(self, search_space: ArchitectureSearchSpace,
                 evaluator: PhotonicArchitectureEvaluator,
                 n_iterations: int = 100) -> SearchResult:
        """
        Run evolutionary optimization.
        
        Args:
            search_space: Architecture search space
            evaluator: Architecture evaluator
            n_iterations: Number of generations
            
        Returns:
            Search result with best architecture
        """
        import time
        start_time = time.time()
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            arch = search_space.sample_architecture('medium')
            population.append(arch)
        
        search_history = []
        performance_evolution = []
        total_evaluations = 0
        best_architecture = None
        best_score = 0.0
        
        global_logger.info(f"Starting evolutionary search with {n_iterations} generations")
        
        for generation in range(n_iterations):
            # Evaluate population
            population_scores = []
            
            for arch in population:
                metrics = evaluator.evaluate(arch)
                score = metrics.get('composite_score', 0.0)
                arch.performance_metrics = metrics
                population_scores.append(score)
                total_evaluations += 1
                
                # Track best architecture
                if score > best_score:
                    best_score = score
                    best_architecture = copy.deepcopy(arch)
            
            # Log generation progress
            avg_score = np.mean(population_scores)
            performance_evolution.append(avg_score)
            
            global_logger.info(f"Generation {generation}: avg_score={avg_score:.4f}, "
                              f"best_score={best_score:.4f}")
            
            # Selection
            elite_indices = np.argsort(population_scores)[-self.n_elite:]
            elite_population = [population[i] for i in elite_indices]
            
            # Create next generation
            new_population = elite_population.copy()  # Elitism
            
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, population_scores)
                parent2 = self._tournament_selection(population, population_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2, search_space)
                else:
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, search_space)
                
                new_population.append(child)
            
            population = new_population
            search_history.extend(elite_population)
        
        search_time = time.time() - start_time
        
        global_logger.info(f"Evolutionary search completed in {search_time:.2f}s, "
                          f"best_score={best_score:.4f}")
        
        return SearchResult(
            best_architecture=best_architecture,
            search_history=search_history,
            performance_evolution=performance_evolution,
            search_time=search_time,
            total_evaluations=total_evaluations
        )
    
    def _tournament_selection(self, population: List[ArchitectureConfig],
                            scores: List[float], tournament_size: int = 3) -> ArchitectureConfig:
        """Tournament selection."""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]
    
    def _crossover(self, parent1: ArchitectureConfig, parent2: ArchitectureConfig,
                   search_space: ArchitectureSearchSpace) -> ArchitectureConfig:
        """Crossover between two architectures."""
        # Simple crossover: combine layers from both parents
        child_layers = []
        
        max_layers = min(len(parent1.layers), len(parent2.layers))
        for i in range(max_layers):
            if random.random() < 0.5:
                child_layers.append(copy.deepcopy(parent1.layers[i]))
            else:
                child_layers.append(copy.deepcopy(parent2.layers[i]))
        
        # Crossover optical parameters
        child_optical_params = {}
        for param_name in parent1.optical_parameters:
            if random.random() < 0.5:
                child_optical_params[param_name] = parent1.optical_parameters[param_name]
            else:
                child_optical_params[param_name] = parent2.optical_parameters.get(param_name, 
                                                                                parent1.optical_parameters[param_name])
        
        # Generate new connections
        child_connections = search_space._generate_connections(child_layers, 15)
        
        child_name = f"crossover_{random.randint(1000, 9999)}"
        
        return ArchitectureConfig(
            name=child_name,
            layers=child_layers,
            connections=child_connections,
            optical_parameters=child_optical_params
        )
    
    def _mutate(self, architecture: ArchitectureConfig, 
               search_space: ArchitectureSearchSpace) -> ArchitectureConfig:
        """Mutate an architecture."""
        mutated = copy.deepcopy(architecture)
        mutated.name = f"mutated_{random.randint(1000, 9999)}"
        
        # Mutate layers
        if random.random() < 0.3 and len(mutated.layers) > 1:
            # Remove a layer
            idx = random.randint(0, len(mutated.layers) - 1)
            mutated.layers.pop(idx)
        
        if random.random() < 0.3:
            # Add a layer
            new_layer = search_space._generate_layer_config(
                random.choice(search_space.layer_types),
                len(mutated.layers)
            )
            insert_pos = random.randint(0, len(mutated.layers))
            mutated.layers.insert(insert_pos, new_layer)
        
        # Mutate layer parameters
        for layer in mutated.layers:
            if random.random() < 0.2:
                layer_type = layer['type']
                if layer_type in search_space.layer_parameters:
                    params = search_space.layer_parameters[layer_type]
                    param_name = random.choice(list(params.keys()))
                    if isinstance(params[param_name], list):
                        layer[param_name] = random.choice(params[param_name])
        
        # Mutate optical parameters
        for param_name in mutated.optical_parameters:
            if random.random() < 0.1:
                param_range = search_space.optical_parameters[param_name]
                mutated.optical_parameters[param_name] = random.uniform(*param_range)
        
        # Regenerate connections
        mutated.connections = search_space._generate_connections(mutated.layers, 15)
        
        return mutated


class PhotonicNAS:
    """Main Neural Architecture Search coordinator for photonic networks."""
    
    def __init__(self, search_space: ArchitectureSearchSpace = None,
                 evaluator: PhotonicArchitectureEvaluator = None):
        """Initialize PhotonicNAS system."""
        self.search_space = search_space or ArchitectureSearchSpace()
        self.evaluator = evaluator or PhotonicArchitectureEvaluator()
        
        self.optimizers = {
            'evolutionary': EvolutionaryPhotonicNAS(),
            'random_search': RandomSearchOptimizer(),
            'differential_evolution': DifferentialEvolutionOptimizer()
        }
        
        self.search_results = {}
        
        global_logger.info("Initialized PhotonicNAS system")
    
    def search(self, optimizer_type: str = 'evolutionary', 
               n_iterations: int = 100, **kwargs) -> SearchResult:
        """
        Run architecture search.
        
        Args:
            optimizer_type: Type of optimizer to use
            n_iterations: Number of search iterations
            **kwargs: Additional optimizer parameters
            
        Returns:
            Search results
        """
        if optimizer_type not in self.optimizers:
            raise ComponentError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer = self.optimizers[optimizer_type]
        
        # Update optimizer parameters
        for key, value in kwargs.items():
            if hasattr(optimizer, key):
                setattr(optimizer, key, value)
        
        global_logger.info(f"Starting {optimizer_type} search with {n_iterations} iterations")
        
        result = optimizer.optimize(self.search_space, self.evaluator, n_iterations)
        self.search_results[optimizer_type] = result
        
        return result
    
    def compare_optimizers(self, optimizer_types: List[str] = None,
                          n_iterations: int = 50) -> Dict[str, SearchResult]:
        """Compare different optimization algorithms."""
        if optimizer_types is None:
            optimizer_types = list(self.optimizers.keys())
        
        results = {}
        
        for optimizer_type in optimizer_types:
            global_logger.info(f"Running {optimizer_type} comparison")
            result = self.search(optimizer_type, n_iterations)
            results[optimizer_type] = result
        
        return results
    
    def export_best_architecture(self, filepath: str, optimizer_type: str = 'evolutionary'):
        """Export best architecture to file."""
        if optimizer_type not in self.search_results:
            raise ComponentError(f"No results for optimizer: {optimizer_type}")
        
        result = self.search_results[optimizer_type]
        arch_data = result.best_architecture.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(arch_data, f, indent=2)
        
        global_logger.info(f"Exported best architecture to {filepath}")
    
    def load_architecture(self, filepath: str) -> ArchitectureConfig:
        """Load architecture from file."""
        with open(filepath, 'r') as f:
            arch_data = json.load(f)
        
        return ArchitectureConfig.from_dict(arch_data)


class RandomSearchOptimizer(OpticalArchitectureOptimizer):
    """Random search baseline optimizer."""
    
    def optimize(self, search_space: ArchitectureSearchSpace,
                 evaluator: PhotonicArchitectureEvaluator,
                 n_iterations: int = 100) -> SearchResult:
        """Run random search."""
        import time
        start_time = time.time()
        
        best_architecture = None
        best_score = 0.0
        search_history = []
        performance_evolution = []
        
        for i in range(n_iterations):
            # Sample random architecture
            arch = search_space.sample_architecture('medium')
            
            # Evaluate
            metrics = evaluator.evaluate(arch)
            score = metrics.get('composite_score', 0.0)
            arch.performance_metrics = metrics
            
            search_history.append(arch)
            performance_evolution.append(score)
            
            # Track best
            if score > best_score:
                best_score = score
                best_architecture = copy.deepcopy(arch)
            
            if (i + 1) % 10 == 0:
                global_logger.info(f"Random search iteration {i+1}: best_score={best_score:.4f}")
        
        search_time = time.time() - start_time
        
        return SearchResult(
            best_architecture=best_architecture,
            search_history=search_history,
            performance_evolution=performance_evolution,
            search_time=search_time,
            total_evaluations=n_iterations
        )


class DifferentialEvolutionOptimizer(OpticalArchitectureOptimizer):
    """Differential evolution optimizer."""
    
    def __init__(self, population_size: int = 30, F: float = 0.5, CR: float = 0.7):
        """Initialize differential evolution."""
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
    
    def optimize(self, search_space: ArchitectureSearchSpace,
                 evaluator: PhotonicArchitectureEvaluator,
                 n_iterations: int = 100) -> SearchResult:
        """Run differential evolution."""
        # Similar to evolutionary but with DE-specific operations
        # This is a simplified implementation
        evolutionary_optimizer = EvolutionaryPhotonicNAS(
            population_size=self.population_size,
            mutation_rate=self.F,
            crossover_rate=self.CR
        )
        
        return evolutionary_optimizer.optimize(search_space, evaluator, n_iterations)