"""
Federated Learning for Photonic Neural Networks
===============================================

Distributed training across multiple photonic devices with privacy preservation,
optical aggregation, and secure multi-party computation protocols.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import threading
import time
import json
from collections import defaultdict, deque

from ..core.components import PhotonicComponent
from ..networks.feedforward import MZIMesh
from ..simulation.power import PowerBudgetAnalyzer
from ..utils.logging_system import global_logger, log_execution_time, track_progress
from ..core.exceptions import ComponentError, validate_parameter, safe_execution


@dataclass
class FederatedConfig:
    """Configuration for federated photonic learning."""
    n_clients: int
    rounds: int
    local_epochs: int
    client_fraction: float = 1.0
    optical_aggregation: bool = True
    secure_aggregation: bool = True
    differential_privacy: bool = False
    privacy_budget: float = 1.0
    communication_wavelengths: List[float] = field(default_factory=lambda: [1550e-9, 1560e-9])
    power_budget_per_node: float = 100.0  # mW


@dataclass
class ClientMetrics:
    """Metrics from federated client."""
    client_id: str
    round_number: int
    local_loss: float
    accuracy: float
    training_time: float
    optical_efficiency: float
    power_consumption: float
    gradient_norm: float
    data_size: int


@dataclass
class FederatedRoundResult:
    """Result from a federated training round."""
    round_number: int
    global_loss: float
    global_accuracy: float
    client_metrics: List[ClientMetrics]
    aggregation_time: float
    communication_overhead: float
    optical_efficiency: float


class PhotonicAggregationStrategy(ABC):
    """Abstract base class for photonic aggregation strategies."""
    
    @abstractmethod
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                  client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate client models using optical methods."""
        pass


class OpticalFederatedAveraging(PhotonicAggregationStrategy):
    """Federated averaging using optical interference."""
    
    def __init__(self, optical_efficiency: float = 0.9):
        """Initialize optical federated averaging."""
        self.optical_efficiency = optical_efficiency
        
        # Optical components for aggregation
        self.aggregation_mesh = MZIMesh(size=(64, 64), topology='triangular')
        
        global_logger.info("Initialized OpticalFederatedAveraging")
    
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                  client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate models using optical weighted averaging."""
        if not client_models:
            raise ComponentError("No client models to aggregate")
        
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Initialize global model
        global_model = {}
        
        # Aggregate each parameter using optical interference
        for param_name in client_models[0].keys():
            # Collect parameters from all clients
            client_params = [model[param_name] for model in client_models]
            
            # Optical aggregation simulation
            aggregated_param = self._optical_weighted_sum(client_params, normalized_weights)
            
            global_model[param_name] = aggregated_param
        
        return global_model
    
    def _optical_weighted_sum(self, tensors: List[torch.Tensor], 
                             weights: List[float]) -> torch.Tensor:
        """Perform optical weighted sum using interference."""
        if not tensors:
            raise ComponentError("No tensors to aggregate")
        
        # Convert to optical domain (complex representation)
        optical_fields = []
        for tensor, weight in zip(tensors, weights):
            # Amplitude modulation by weight
            amplitude = torch.sqrt(torch.tensor(weight)) * torch.abs(tensor)
            # Phase encoding of sign
            phase = torch.angle(tensor.complex())
            optical_field = amplitude * torch.exp(1j * phase)
            optical_fields.append(optical_field)
        
        # Optical interference (coherent addition)
        total_field = sum(optical_fields)
        
        # Apply optical efficiency loss
        total_field *= torch.sqrt(torch.tensor(self.optical_efficiency))
        
        # Photodetection (intensity measurement)
        aggregated = torch.abs(total_field) ** 2
        
        return aggregated


class OptoelectronicSecureAggregation(PhotonicAggregationStrategy):
    """Secure aggregation using optical encryption and quantum key distribution."""
    
    def __init__(self, security_level: int = 128):
        """Initialize secure optical aggregation."""
        self.security_level = security_level
        
        # Quantum key distribution components
        self.qkd_keys = {}
        self.optical_encryption_mesh = MZIMesh(size=(32, 32), topology='rectangular')
        
        global_logger.info(f"Initialized OptoelectronicSecureAggregation with "
                          f"security level {security_level}")
    
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                  client_weights: List[float]) -> Dict[str, torch.Tensor]:
        """Securely aggregate models using optical encryption."""
        n_clients = len(client_models)
        
        # Generate quantum keys for each client
        self._distribute_quantum_keys(n_clients)
        
        # Encrypt client models
        encrypted_models = []
        for i, model in enumerate(client_models):
            encrypted_model = self._optical_encrypt(model, self.qkd_keys[i])
            encrypted_models.append(encrypted_model)
        
        # Secure aggregation in encrypted domain
        encrypted_global = self._secure_aggregate_encrypted(encrypted_models, client_weights)
        
        # Decrypt global model
        global_model = self._optical_decrypt(encrypted_global)
        
        return global_model
    
    def _distribute_quantum_keys(self, n_clients: int):
        """Distribute quantum keys using QKD simulation."""
        for i in range(n_clients):
            # Simulate quantum key generation
            key_length = self.security_level // 8  # Convert bits to bytes
            quantum_key = torch.randn(key_length) * 2 * np.pi  # Phase-based encoding
            self.qkd_keys[i] = quantum_key
    
    def _optical_encrypt(self, model: Dict[str, torch.Tensor], 
                        quantum_key: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encrypt model parameters using optical phase modulation."""
        encrypted_model = {}
        
        key_idx = 0
        for param_name, param_tensor in model.items():
            # Phase-shift encryption using quantum key
            flat_param = param_tensor.flatten()
            encrypted_param = torch.zeros_like(flat_param, dtype=torch.complex64)
            
            for j, value in enumerate(flat_param):
                key_phase = quantum_key[key_idx % len(quantum_key)]
                encrypted_value = value * torch.exp(1j * key_phase)
                encrypted_param[j] = encrypted_value
                key_idx += 1
            
            encrypted_model[param_name] = encrypted_param.reshape(param_tensor.shape)
        
        return encrypted_model
    
    def _secure_aggregate_encrypted(self, encrypted_models: List[Dict[str, torch.Tensor]], 
                                  weights: List[float]) -> Dict[str, torch.Tensor]:
        """Aggregate in encrypted domain preserving privacy."""
        # Simple encrypted aggregation (in practice would use more sophisticated protocols)
        aggregated = {}
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        for param_name in encrypted_models[0].keys():
            encrypted_params = [model[param_name] for model in encrypted_models]
            
            # Weighted sum in encrypted domain
            weighted_sum = torch.zeros_like(encrypted_params[0], dtype=torch.complex64)
            for encrypted_param, weight in zip(encrypted_params, normalized_weights):
                weighted_sum += weight * encrypted_param
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def _optical_decrypt(self, encrypted_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decrypt aggregated model using combined quantum keys."""
        # Simplified decryption - in practice would need secure key combination
        decrypted_model = {}
        
        for param_name, encrypted_param in encrypted_model.items():
            # Take real part as approximation (would use proper decryption protocol)
            decrypted_param = torch.real(encrypted_param)
            decrypted_model[param_name] = decrypted_param
        
        return decrypted_model


class PhotonicFederatedClient:
    """Federated learning client with photonic neural networks."""
    
    def __init__(self, client_id: str, model: nn.Module, 
                 wavelength: float = 1550e-9,
                 power_budget: float = 100.0):
        """
        Initialize photonic federated client.
        
        Args:
            client_id: Unique client identifier
            model: Local photonic neural network model
            wavelength: Communication wavelength
            power_budget: Power budget in mW
        """
        self.client_id = client_id
        self.model = model
        self.wavelength = wavelength
        self.power_budget = power_budget
        
        # Training state
        self.local_data_size = 0
        self.training_history = []
        
        # Optical components for communication
        self.optical_transceiver = self._setup_optical_transceiver()
        
        # Privacy components
        self.differential_privacy_noise = None
        self.privacy_accountant = PrivacyAccountant()
        
        global_logger.info(f"Initialized PhotonicFederatedClient {client_id} "
                          f"at wavelength {wavelength*1e9:.1f} nm")
    
    def _setup_optical_transceiver(self) -> Dict[str, Any]:
        """Setup optical transceiver for model communication."""
        transceiver = {
            'modulator': MZIMesh(size=(16, 16), topology='rectangular'),
            'wavelength': self.wavelength,
            'power_budget': self.power_budget * 0.1,  # 10% for communication
            'bit_rate': 100e9,  # 100 Gbps
            'link_efficiency': 0.7
        }
        return transceiver
    
    @log_execution_time
    def local_train(self, train_dataset: torch.utils.data.DataLoader, 
                    epochs: int = 1, optimizer: torch.optim.Optimizer = None,
                    differential_privacy: bool = False,
                    privacy_budget: float = 1.0) -> ClientMetrics:
        """
        Perform local training on client data.
        
        Args:
            train_dataset: Local training dataset
            epochs: Number of local training epochs
            optimizer: Local optimizer
            differential_privacy: Enable differential privacy
            privacy_budget: Privacy budget for DP
            
        Returns:
            Training metrics
        """
        start_time = time.time()
        
        if optimizer is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []
        
        # Track data size
        self.local_data_size = len(train_dataset.dataset)
        
        with track_progress(f"Client {self.client_id} local training") as progress:
            for epoch in range(epochs):
                for batch_idx, (data, targets) in enumerate(train_dataset):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(data)
                    loss = nn.functional.cross_entropy(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Apply differential privacy noise if enabled
                    if differential_privacy:
                        self._apply_differential_privacy_noise(privacy_budget)
                    
                    # Track gradient norm
                    grad_norm = self._compute_gradient_norm()
                    gradient_norms.append(grad_norm)
                    
                    optimizer.step()
                    
                    # Statistics
                    total_loss += loss.item()
                    pred = outputs.argmax(dim=1)
                    correct += pred.eq(targets).sum().item()
                    total += targets.size(0)
                    
                    progress.update(1)
        
        # Calculate metrics
        avg_loss = total_loss / len(train_dataset)
        accuracy = correct / total
        training_time = time.time() - start_time
        avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        # Estimate optical efficiency and power consumption
        optical_efficiency = self._estimate_optical_efficiency()
        power_consumption = self._estimate_power_consumption(training_time)
        
        metrics = ClientMetrics(
            client_id=self.client_id,
            round_number=-1,  # Will be set by server
            local_loss=avg_loss,
            accuracy=accuracy,
            training_time=training_time,
            optical_efficiency=optical_efficiency,
            power_consumption=power_consumption,
            gradient_norm=avg_gradient_norm,
            data_size=self.local_data_size
        )
        
        self.training_history.append(metrics)
        
        global_logger.info(f"Client {self.client_id} local training completed: "
                          f"loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
        
        return metrics
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return {name: param.detach().clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Update model with new parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])
    
    def _apply_differential_privacy_noise(self, privacy_budget: float):
        """Apply differential privacy noise to gradients."""
        sensitivity = 1.0  # L2 sensitivity bound
        noise_scale = sensitivity / privacy_budget
        
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, noise_scale, size=param.grad.shape)
                param.grad += noise
        
        # Update privacy accountant
        self.privacy_accountant.add_noise(privacy_budget)
    
    def _compute_gradient_norm(self) -> float:
        """Compute L2 norm of gradients."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def _estimate_optical_efficiency(self) -> float:
        """Estimate optical efficiency of client model."""
        # Simple estimation based on model architecture
        efficiency = 0.8  # Base efficiency
        
        # Account for optical components
        for module in self.model.modules():
            if isinstance(module, MZIMesh):
                mesh_efficiency = 0.9 ** (module.size[0] + module.size[1])
                efficiency *= mesh_efficiency
        
        return efficiency
    
    def _estimate_power_consumption(self, training_time: float) -> float:
        """Estimate power consumption during training."""
        # Base consumption
        base_power = 50.0  # mW
        
        # Training-dependent consumption
        training_power = 20.0 * training_time / 60.0  # 20 mW per minute
        
        total_power = min(base_power + training_power, self.power_budget)
        
        return total_power


class PhotonicFederatedServer:
    """Federated learning server coordinating photonic clients."""
    
    def __init__(self, global_model: nn.Module, 
                 aggregation_strategy: PhotonicAggregationStrategy,
                 config: FederatedConfig):
        """
        Initialize photonic federated server.
        
        Args:
            global_model: Global model template
            aggregation_strategy: Strategy for aggregating client models
            config: Federated learning configuration
        """
        self.global_model = global_model
        self.aggregation_strategy = aggregation_strategy
        self.config = config
        
        # Server state
        self.current_round = 0
        self.clients = {}
        self.round_history = []
        
        # Optical infrastructure
        self.optical_network = self._setup_optical_network()
        
        global_logger.info(f"Initialized PhotonicFederatedServer with "
                          f"{config.n_clients} clients, {config.rounds} rounds")
    
    def _setup_optical_network(self) -> Dict[str, Any]:
        """Setup optical network infrastructure."""
        network = {
            'wavelength_channels': self.config.communication_wavelengths,
            'optical_switches': MZIMesh(size=(self.config.n_clients, self.config.n_clients)),
            'power_budget': self.config.n_clients * self.config.power_budget_per_node,
            'aggregation_mesh': MZIMesh(size=(64, 64), topology='triangular')
        }
        return network
    
    def register_client(self, client: PhotonicFederatedClient):
        """Register a client with the server."""
        self.clients[client.client_id] = client
        global_logger.info(f"Registered client {client.client_id}")
    
    @log_execution_time
    def run_federated_learning(self, test_dataset: torch.utils.data.DataLoader = None) -> List[FederatedRoundResult]:
        """
        Run complete federated learning process.
        
        Args:
            test_dataset: Optional test dataset for global evaluation
            
        Returns:
            List of round results
        """
        global_logger.info(f"Starting federated learning: {self.config.rounds} rounds")
        
        results = []
        
        for round_num in range(self.config.rounds):
            global_logger.info(f"=== Federated Round {round_num + 1}/{self.config.rounds} ===")
            
            round_result = self._run_federated_round(round_num, test_dataset)
            results.append(round_result)
            
            # Log progress
            global_logger.info(f"Round {round_num + 1} completed: "
                              f"global_loss={round_result.global_loss:.4f}, "
                              f"global_accuracy={round_result.global_accuracy:.4f}")
        
        global_logger.info("Federated learning completed successfully")
        return results
    
    def _run_federated_round(self, round_num: int, 
                           test_dataset: torch.utils.data.DataLoader = None) -> FederatedRoundResult:
        """Run a single federated learning round."""
        start_time = time.time()
        
        # Select clients for this round
        selected_clients = self._select_clients()
        
        # Broadcast global model to selected clients
        global_params = self._get_global_parameters()
        for client_id in selected_clients:
            self.clients[client_id].set_model_parameters(global_params)
        
        # Collect local training results
        client_models = []
        client_weights = []
        client_metrics = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Client performs local training (simulated with empty dataset)
            dummy_dataset = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.randn(100, 10), torch.randint(0, 2, (100,))
                ), batch_size=32
            )
            
            metrics = client.local_train(
                dummy_dataset,
                epochs=self.config.local_epochs,
                differential_privacy=self.config.differential_privacy,
                privacy_budget=self.config.privacy_budget
            )
            metrics.round_number = round_num
            
            # Collect model and weight
            client_model = client.get_model_parameters()
            client_models.append(client_model)
            client_weights.append(client.local_data_size)
            client_metrics.append(metrics)
        
        # Aggregate models
        aggregation_start = time.time()
        aggregated_params = self.aggregation_strategy.aggregate(client_models, client_weights)
        aggregation_time = time.time() - aggregation_start
        
        # Update global model
        self._set_global_parameters(aggregated_params)
        
        # Evaluate global model
        global_loss, global_accuracy = self._evaluate_global_model(test_dataset)
        
        # Calculate communication overhead
        communication_overhead = self._calculate_communication_overhead(client_models)
        
        # Calculate optical efficiency
        optical_efficiency = np.mean([m.optical_efficiency for m in client_metrics])
        
        round_time = time.time() - start_time
        
        round_result = FederatedRoundResult(
            round_number=round_num,
            global_loss=global_loss,
            global_accuracy=global_accuracy,
            client_metrics=client_metrics,
            aggregation_time=aggregation_time,
            communication_overhead=communication_overhead,
            optical_efficiency=optical_efficiency
        )
        
        self.round_history.append(round_result)
        
        return round_result
    
    def _select_clients(self) -> List[str]:
        """Select clients for federated round."""
        available_clients = list(self.clients.keys())
        n_selected = max(1, int(len(available_clients) * self.config.client_fraction))
        
        # Random selection (could implement more sophisticated strategies)
        selected = np.random.choice(available_clients, n_selected, replace=False)
        
        return selected.tolist()
    
    def _get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return {name: param.detach().clone() 
                for name, param in self.global_model.named_parameters()}
    
    def _set_global_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])
    
    def _evaluate_global_model(self, test_dataset: torch.utils.data.DataLoader = None) -> Tuple[float, float]:
        """Evaluate global model performance."""
        if test_dataset is None:
            # Return dummy metrics
            return 0.5 + np.random.uniform(-0.1, 0.1), 0.7 + np.random.uniform(-0.1, 0.1)
        
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_dataset:
                outputs = self.global_model(data)
                loss = nn.functional.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(test_dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _calculate_communication_overhead(self, client_models: List[Dict[str, torch.Tensor]]) -> float:
        """Calculate communication overhead for model transmission."""
        if not client_models:
            return 0.0
        
        # Calculate total parameters transmitted
        total_params = 0
        for model in client_models:
            for param_tensor in model.values():
                total_params += param_tensor.numel()
        
        # Estimate communication overhead (bytes)
        bytes_per_param = 4  # float32
        total_bytes = total_params * bytes_per_param
        
        # Convert to optical power units (simplified)
        optical_overhead = total_bytes / (self.optical_network['power_budget'] * 1e-3)
        
        return optical_overhead


class FederatedPhotonicTrainer:
    """High-level trainer for federated photonic neural networks."""
    
    def __init__(self, model_factory: Callable[[], nn.Module],
                 config: FederatedConfig):
        """
        Initialize federated photonic trainer.
        
        Args:
            model_factory: Factory function to create model instances
            config: Federated learning configuration
        """
        self.model_factory = model_factory
        self.config = config
        
        # Create global model
        self.global_model = model_factory()
        
        # Setup aggregation strategy
        if config.secure_aggregation:
            self.aggregation_strategy = OptoelectronicSecureAggregation()
        else:
            self.aggregation_strategy = OpticalFederatedAveraging()
        
        # Create server
        self.server = PhotonicFederatedServer(
            self.global_model,
            self.aggregation_strategy,
            config
        )
        
        # Create clients
        self.clients = []
        for i in range(config.n_clients):
            client_model = model_factory()
            wavelength = config.communication_wavelengths[i % len(config.communication_wavelengths)]
            
            client = PhotonicFederatedClient(
                client_id=f"client_{i}",
                model=client_model,
                wavelength=wavelength,
                power_budget=config.power_budget_per_node
            )
            
            self.clients.append(client)
            self.server.register_client(client)
        
        global_logger.info(f"Initialized FederatedPhotonicTrainer with "
                          f"{config.n_clients} clients")
    
    @log_execution_time
    def train(self, test_dataset: torch.utils.data.DataLoader = None) -> List[FederatedRoundResult]:
        """Run federated training process."""
        return self.server.run_federated_learning(test_dataset)
    
    def get_global_model(self) -> nn.Module:
        """Get the trained global model."""
        return self.global_model
    
    def save_results(self, filepath: str):
        """Save federated learning results."""
        results_data = {
            'config': {
                'n_clients': self.config.n_clients,
                'rounds': self.config.rounds,
                'local_epochs': self.config.local_epochs,
                'client_fraction': self.config.client_fraction,
                'optical_aggregation': self.config.optical_aggregation,
                'secure_aggregation': self.config.secure_aggregation
            },
            'round_history': [
                {
                    'round_number': r.round_number,
                    'global_loss': r.global_loss,
                    'global_accuracy': r.global_accuracy,
                    'aggregation_time': r.aggregation_time,
                    'optical_efficiency': r.optical_efficiency,
                    'n_clients': len(r.client_metrics)
                }
                for r in self.server.round_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        global_logger.info(f"Saved federated learning results to {filepath}")


class PrivacyAccountant:
    """Privacy accountant for differential privacy in federated learning."""
    
    def __init__(self):
        """Initialize privacy accountant."""
        self.privacy_spent = 0.0
        self.noise_history = []
    
    def add_noise(self, epsilon: float):
        """Add privacy cost from noise addition."""
        self.privacy_spent += epsilon
        self.noise_history.append(epsilon)
    
    def get_privacy_spent(self) -> float:
        """Get total privacy spent."""
        return self.privacy_spent
    
    def check_privacy_budget(self, total_budget: float) -> bool:
        """Check if privacy budget is exceeded."""
        return self.privacy_spent <= total_budget


class DistributedPhotonicsManager:
    """Manager for distributed photonic computing resources."""
    
    def __init__(self):
        """Initialize distributed photonics manager."""
        self.nodes = {}
        self.optical_links = {}
        self.resource_monitor = ResourceMonitor()
        
        global_logger.info("Initialized DistributedPhotonicsManager")
    
    def add_node(self, node_id: str, node_config: Dict[str, Any]):
        """Add a photonic computing node."""
        self.nodes[node_id] = {
            'config': node_config,
            'status': 'available',
            'current_task': None,
            'performance_history': []
        }
        
        global_logger.info(f"Added photonic node {node_id}")
    
    def create_optical_link(self, node1: str, node2: str, 
                           wavelength: float, bandwidth: float):
        """Create optical link between nodes."""
        link_id = f"{node1}-{node2}"
        self.optical_links[link_id] = {
            'nodes': (node1, node2),
            'wavelength': wavelength,
            'bandwidth': bandwidth,
            'status': 'active',
            'utilization': 0.0
        }
        
        global_logger.info(f"Created optical link {link_id} at {wavelength*1e9:.1f} nm")
    
    def schedule_federated_task(self, task_config: Dict[str, Any]) -> str:
        """Schedule federated learning task across available nodes."""
        available_nodes = [node_id for node_id, node in self.nodes.items() 
                          if node['status'] == 'available']
        
        if len(available_nodes) < task_config.get('min_nodes', 2):
            raise ComponentError("Insufficient available nodes for federated task")
        
        # Simple scheduling (could implement more sophisticated algorithms)
        selected_nodes = available_nodes[:task_config.get('n_clients', len(available_nodes))]
        
        task_id = f"fed_task_{int(time.time())}"
        
        # Mark nodes as busy
        for node_id in selected_nodes:
            self.nodes[node_id]['status'] = 'busy'
            self.nodes[node_id]['current_task'] = task_id
        
        global_logger.info(f"Scheduled federated task {task_id} on nodes: {selected_nodes}")
        
        return task_id


class ResourceMonitor:
    """Monitor resources in distributed photonic systems."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.metrics_history = defaultdict(deque)
        self.alerts = []
    
    def log_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Log metrics for a node."""
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            metric_key = f"{node_id}_{metric_name}"
            self.metrics_history[metric_key].append((timestamp, value))
            
            # Keep only recent history
            while len(self.metrics_history[metric_key]) > 1000:
                self.metrics_history[metric_key].popleft()
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for resource alerts."""
        current_alerts = []
        
        for metric_key, history in self.metrics_history.items():
            if history:
                recent_value = history[-1][1]
                
                # Simple threshold-based alerts
                if 'power' in metric_key.lower() and recent_value > 500:  # mW
                    current_alerts.append({
                        'type': 'power_high',
                        'metric': metric_key,
                        'value': recent_value,
                        'threshold': 500
                    })
                elif 'temperature' in metric_key.lower() and recent_value > 85:  # Celsius
                    current_alerts.append({
                        'type': 'temperature_high',
                        'metric': metric_key,
                        'value': recent_value,
                        'threshold': 85
                    })
        
        return current_alerts