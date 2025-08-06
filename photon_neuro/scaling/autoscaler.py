"""
Auto-scaling and dynamic resource management system.
"""

import asyncio
import docker
import kubernetes
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import psutil
from pathlib import Path

# Optional imports for cloud providers
try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import compute_v1
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.mgmt.compute import ComputeManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"
    MIGRATE = "migrate"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float = 0.0
    storage_usage_percent: float = 0.0
    network_throughput_mbps: float = 0.0
    active_connections: int = 0
    queue_length: int = 0


@dataclass
class ScalingDecision:
    """A scaling decision made by the autoscaler."""
    timestamp: float
    resource_type: ResourceType
    action: ScalingAction
    current_capacity: float
    target_capacity: float
    reason: str
    confidence: float
    estimated_cost_change: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceConfiguration:
    """Configuration for a scalable resource."""
    min_capacity: float
    max_capacity: float
    target_utilization: float
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_seconds: int
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.8


class AutoScaler:
    """Advanced auto-scaling system with predictive capabilities."""
    
    def __init__(self, 
                 monitoring_interval: float = 30.0,
                 enable_predictive_scaling: bool = True,
                 enable_cost_optimization: bool = True):
        
        self.monitoring_interval = monitoring_interval
        self.enable_predictive_scaling = enable_predictive_scaling
        self.enable_cost_optimization = enable_cost_optimization
        
        # Resource configurations
        self.resource_configs = {
            ResourceType.CPU: ResourceConfiguration(
                min_capacity=1.0, max_capacity=16.0, target_utilization=70.0,
                scale_up_threshold=80.0, scale_down_threshold=40.0,
                cooldown_seconds=300
            ),
            ResourceType.MEMORY: ResourceConfiguration(
                min_capacity=1.0, max_capacity=64.0, target_utilization=75.0,
                scale_up_threshold=85.0, scale_down_threshold=45.0,
                cooldown_seconds=300
            ),
            ResourceType.GPU: ResourceConfiguration(
                min_capacity=0.0, max_capacity=8.0, target_utilization=80.0,
                scale_up_threshold=90.0, scale_down_threshold=30.0,
                cooldown_seconds=600
            )
        }
        
        # State tracking
        self.current_resources = {}
        self.metrics_history = []
        self.scaling_history = []
        self.last_scaling_time = {}
        
        # Prediction and optimization
        self.workload_predictor = WorkloadPredictor()
        self.cost_optimizer = CostOptimizer()
        
        # Control flags
        self.is_running = False
        self.scaling_thread = None
        
        # Resource managers
        self.resource_managers = {
            'docker': DockerResourceManager(),
            'kubernetes': KubernetesResourceManager(),
            'cloud_aws': AWSResourceManager() if AWS_AVAILABLE else None,
            'cloud_gcp': GCPResourceManager() if GCP_AVAILABLE else None,
            'cloud_azure': AzureResourceManager() if AZURE_AVAILABLE else None
        }
        
        # Filter out unavailable managers
        self.resource_managers = {k: v for k, v in self.resource_managers.items() if v is not None}
        
        logging.info(f"AutoScaler initialized with managers: {list(self.resource_managers.keys())}")
    
    def start(self):
        """Start the auto-scaling system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logging.info("AutoScaler started")
    
    def stop(self):
        """Stop the auto-scaling system."""
        self.is_running = False
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        logging.info("AutoScaler stopped")
    
    def _scaling_loop(self):
        """Main scaling loop."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history manageable
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Update workload predictor
                if self.enable_predictive_scaling:
                    self.workload_predictor.update(metrics)
                
                # Make scaling decisions
                decisions = self._make_scaling_decisions(metrics)
                
                # Execute scaling decisions
                for decision in decisions:
                    self._execute_scaling_decision(decision)
                    
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Error in scaling loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource utilization metrics."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1.0)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics
        gpu_usage = 0.0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
        except ImportError:
            # Fallback to CUDA if available
            import torch
            if torch.cuda.is_available():
                # Simple estimation based on memory usage
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage = (allocated / total) * 100
        except Exception:
            pass
        
        # Storage metrics
        storage_usage = 0.0
        try:
            disk = psutil.disk_usage('/')
            storage_usage = disk.percent
        except Exception:
            pass
        
        # Network metrics
        network_throughput = 0.0
        try:
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
                bytes_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
                total_bytes = bytes_sent + bytes_recv
                network_throughput = total_bytes / (1024 * 1024) / self.monitoring_interval  # MB/s
            self._last_net_io = net_io
        except Exception:
            pass
        
        return ResourceMetrics(
            timestamp=timestamp,
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage,
            gpu_usage_percent=gpu_usage,
            storage_usage_percent=storage_usage,
            network_throughput_mbps=network_throughput,
            active_connections=len(psutil.net_connections()),
            queue_length=0  # Would be populated by application-specific logic
        )
    
    def _make_scaling_decisions(self, metrics: ResourceMetrics) -> List[ScalingDecision]:
        """Make scaling decisions based on current and predicted metrics."""
        decisions = []
        current_time = time.time()
        
        # Check each resource type
        for resource_type, config in self.resource_configs.items():
            # Skip if in cooldown period
            last_scaling = self.last_scaling_time.get(resource_type, 0)
            if current_time - last_scaling < config.cooldown_seconds:
                continue
            
            # Get current utilization
            current_usage = self._get_resource_usage(metrics, resource_type)
            current_capacity = self.current_resources.get(resource_type.value, config.min_capacity)
            
            # Basic reactive scaling
            action = ScalingAction.NO_CHANGE
            target_capacity = current_capacity
            reason = "No action needed"
            confidence = 0.8
            
            if current_usage > config.scale_up_threshold:
                action = ScalingAction.SCALE_UP
                target_capacity = min(current_capacity * config.scale_up_factor, config.max_capacity)
                reason = f"Current usage {current_usage:.1f}% > threshold {config.scale_up_threshold}%"
                confidence = 0.9
                
            elif current_usage < config.scale_down_threshold:
                action = ScalingAction.SCALE_DOWN  
                target_capacity = max(current_capacity * config.scale_down_factor, config.min_capacity)
                reason = f"Current usage {current_usage:.1f}% < threshold {config.scale_down_threshold}%"
                confidence = 0.8
            
            # Predictive scaling adjustment
            if self.enable_predictive_scaling and action == ScalingAction.NO_CHANGE:
                prediction = self.workload_predictor.predict_usage(resource_type, horizon_minutes=10)
                if prediction and prediction['confidence'] > 0.6:
                    predicted_usage = prediction['predicted_usage']
                    
                    if predicted_usage > config.scale_up_threshold:
                        action = ScalingAction.SCALE_UP
                        target_capacity = min(current_capacity * 1.2, config.max_capacity)  # Conservative
                        reason = f"Predicted usage {predicted_usage:.1f}% > threshold"
                        confidence = prediction['confidence'] * 0.8  # Reduce confidence for predictions
            
            # Cost optimization
            if self.enable_cost_optimization and action != ScalingAction.NO_CHANGE:
                cost_analysis = self.cost_optimizer.analyze_scaling_cost(
                    resource_type, current_capacity, target_capacity
                )
                
                # Adjust decision based on cost
                if cost_analysis['cost_increase_percent'] > 50 and action == ScalingAction.SCALE_UP:
                    # Very expensive scaling - reduce target
                    target_capacity = current_capacity * 1.1
                    confidence *= 0.7
                    reason += " (cost-optimized)"
            
            # Create decision if action is needed
            if action != ScalingAction.NO_CHANGE and target_capacity != current_capacity:
                decision = ScalingDecision(
                    timestamp=current_time,
                    resource_type=resource_type,
                    action=action,
                    current_capacity=current_capacity,
                    target_capacity=target_capacity,
                    reason=reason,
                    confidence=confidence,
                    metadata={
                        'current_usage_percent': current_usage,
                        'config_threshold': config.scale_up_threshold if action == ScalingAction.SCALE_UP else config.scale_down_threshold
                    }
                )
                decisions.append(decision)
        
        return decisions
    
    def _get_resource_usage(self, metrics: ResourceMetrics, resource_type: ResourceType) -> float:
        """Get current usage percentage for a resource type."""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_usage_percent
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_usage_percent
        elif resource_type == ResourceType.GPU:
            return metrics.gpu_usage_percent
        elif resource_type == ResourceType.STORAGE:
            return metrics.storage_usage_percent
        else:
            return 0.0
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision using available resource managers."""
        try:
            success = False
            
            # Try each resource manager
            for manager_name, manager in self.resource_managers.items():
                if manager.can_handle_resource(decision.resource_type):
                    try:
                        success = manager.scale_resource(
                            decision.resource_type,
                            decision.target_capacity,
                            decision.metadata
                        )
                        if success:
                            logging.info(f"Scaled {decision.resource_type.value} to {decision.target_capacity} using {manager_name}")
                            break
                    except Exception as e:
                        logging.error(f"Failed to scale using {manager_name}: {e}")
                        continue
            
            if success:
                # Update tracking
                self.current_resources[decision.resource_type.value] = decision.target_capacity
                self.last_scaling_time[decision.resource_type] = decision.timestamp
                self.scaling_history.append(decision)
                
                # Keep history manageable
                if len(self.scaling_history) > 100:
                    self.scaling_history = self.scaling_history[-50:]
                    
            else:
                logging.error(f"Failed to execute scaling decision: {decision}")
                
        except Exception as e:
            logging.error(f"Error executing scaling decision: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status."""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        recent_decisions = self.scaling_history[-5:] if self.scaling_history else []
        
        return {
            'is_running': self.is_running,
            'current_resources': self.current_resources.copy(),
            'recent_metrics': [
                {
                    'timestamp': m.timestamp,
                    'cpu_usage': m.cpu_usage_percent,
                    'memory_usage': m.memory_usage_percent,
                    'gpu_usage': m.gpu_usage_percent
                } for m in recent_metrics
            ],
            'recent_scaling_decisions': [
                {
                    'timestamp': d.timestamp,
                    'resource_type': d.resource_type.value,
                    'action': d.action.value,
                    'target_capacity': d.target_capacity,
                    'reason': d.reason
                } for d in recent_decisions
            ],
            'resource_managers': list(self.resource_managers.keys())
        }


class WorkloadPredictor:
    """Predicts future workload patterns for proactive scaling."""
    
    def __init__(self, history_size: int = 500):
        self.history_size = history_size
        self.metrics_history = []
        self.prediction_models = {}
    
    def update(self, metrics: ResourceMetrics):
        """Update with new metrics."""
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size//2:]
        
        # Update models periodically
        if len(self.metrics_history) % 50 == 0:
            self._update_prediction_models()
    
    def _update_prediction_models(self):
        """Update prediction models with recent data."""
        if len(self.metrics_history) < 100:
            return
        
        # Simple trend-based prediction for each resource
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]:
            values = []
            
            for metrics in self.metrics_history[-100:]:
                if resource_type == ResourceType.CPU:
                    values.append(metrics.cpu_usage_percent)
                elif resource_type == ResourceType.MEMORY:
                    values.append(metrics.memory_usage_percent)
                elif resource_type == ResourceType.GPU:
                    values.append(metrics.gpu_usage_percent)
            
            # Calculate trend
            x = np.arange(len(values))
            if len(values) > 10:
                slope = np.polyfit(x, values, 1)[0]
                recent_avg = np.mean(values[-10:])
                
                self.prediction_models[resource_type] = {
                    'trend_slope': slope,
                    'recent_average': recent_avg,
                    'variance': np.var(values),
                    'last_update': time.time()
                }
    
    def predict_usage(self, resource_type: ResourceType, horizon_minutes: int = 10) -> Optional[Dict[str, Any]]:
        """Predict resource usage for the given time horizon."""
        if resource_type not in self.prediction_models:
            return None
        
        model = self.prediction_models[resource_type]
        
        # Simple linear prediction
        time_steps = horizon_minutes * 2  # Assuming 30-second intervals
        predicted_usage = model['recent_average'] + model['trend_slope'] * time_steps
        
        # Add some bounds checking
        predicted_usage = max(0, min(100, predicted_usage))
        
        # Calculate confidence based on variance and data age
        age_minutes = (time.time() - model['last_update']) / 60
        confidence = max(0.3, 0.9 - model['variance'] / 1000 - age_minutes / 60)
        
        return {
            'predicted_usage': predicted_usage,
            'confidence': confidence,
            'horizon_minutes': horizon_minutes,
            'trend_slope': model['trend_slope']
        }


class CostOptimizer:
    """Optimizes scaling decisions based on cost considerations."""
    
    def __init__(self):
        # Cost per unit per hour for different resources
        self.resource_costs = {
            ResourceType.CPU: 0.05,      # $0.05 per core-hour
            ResourceType.MEMORY: 0.01,   # $0.01 per GB-hour  
            ResourceType.GPU: 0.90,      # $0.90 per GPU-hour
            ResourceType.STORAGE: 0.001  # $0.001 per GB-hour
        }
        
        self.cost_history = []
    
    def analyze_scaling_cost(self, resource_type: ResourceType, 
                           current_capacity: float, target_capacity: float) -> Dict[str, Any]:
        """Analyze the cost impact of a scaling decision."""
        
        if resource_type not in self.resource_costs:
            return {
                'cost_increase_per_hour': 0.0,
                'cost_increase_percent': 0.0,
                'recommendation': 'proceed'
            }
        
        unit_cost = self.resource_costs[resource_type]
        current_cost_per_hour = current_capacity * unit_cost
        new_cost_per_hour = target_capacity * unit_cost
        
        cost_increase = new_cost_per_hour - current_cost_per_hour
        cost_increase_percent = (cost_increase / max(current_cost_per_hour, 0.001)) * 100
        
        # Make recommendation
        recommendation = 'proceed'
        if cost_increase_percent > 100:
            recommendation = 'review'  # More than 100% increase
        elif cost_increase_percent > 200:
            recommendation = 'block'   # More than 200% increase
        
        return {
            'current_cost_per_hour': current_cost_per_hour,
            'new_cost_per_hour': new_cost_per_hour,
            'cost_increase_per_hour': cost_increase,
            'cost_increase_percent': cost_increase_percent,
            'recommendation': recommendation
        }


class DockerResourceManager:
    """Manages Docker container resources."""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.available = True
        except Exception:
            self.available = False
    
    def can_handle_resource(self, resource_type: ResourceType) -> bool:
        """Check if this manager can handle the resource type."""
        return self.available and resource_type in [ResourceType.CPU, ResourceType.MEMORY]
    
    def scale_resource(self, resource_type: ResourceType, target_capacity: float, 
                      metadata: Dict[str, Any]) -> bool:
        """Scale Docker container resources."""
        if not self.available:
            return False
        
        try:
            # This is a simplified example - real implementation would be more complex
            containers = self.client.containers.list(filters={'label': 'photon-neuro'})
            
            for container in containers:
                if resource_type == ResourceType.CPU:
                    # Update CPU allocation
                    container.update(cpu_count=int(target_capacity))
                elif resource_type == ResourceType.MEMORY:
                    # Update memory allocation
                    container.update(mem_limit=f"{int(target_capacity)}g")
            
            return True
            
        except Exception as e:
            logging.error(f"Docker scaling failed: {e}")
            return False


class KubernetesResourceManager:
    """Manages Kubernetes pod resources."""
    
    def __init__(self):
        try:
            kubernetes.config.load_incluster_config()
            self.v1 = kubernetes.client.CoreV1Api()
            self.apps_v1 = kubernetes.client.AppsV1Api()
            self.available = True
        except Exception:
            try:
                kubernetes.config.load_kube_config()
                self.v1 = kubernetes.client.CoreV1Api()
                self.apps_v1 = kubernetes.client.AppsV1Api()
                self.available = True
            except Exception:
                self.available = False
    
    def can_handle_resource(self, resource_type: ResourceType) -> bool:
        """Check if this manager can handle the resource type."""
        return self.available and resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]
    
    def scale_resource(self, resource_type: ResourceType, target_capacity: float,
                      metadata: Dict[str, Any]) -> bool:
        """Scale Kubernetes deployment resources."""
        if not self.available:
            return False
        
        try:
            namespace = metadata.get('namespace', 'default')
            deployment_name = metadata.get('deployment', 'photon-neuro')
            
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(deployment_name, namespace)
            
            # Update resource requirements
            container = deployment.spec.template.spec.containers[0]
            
            if resource_type == ResourceType.CPU:
                container.resources.requests['cpu'] = f"{target_capacity}"
                container.resources.limits['cpu'] = f"{target_capacity * 1.1}"
            elif resource_type == ResourceType.MEMORY:
                container.resources.requests['memory'] = f"{int(target_capacity)}Gi"
                container.resources.limits['memory'] = f"{int(target_capacity * 1.1)}Gi"
            elif resource_type == ResourceType.GPU:
                container.resources.requests['nvidia.com/gpu'] = str(int(target_capacity))
                container.resources.limits['nvidia.com/gpu'] = str(int(target_capacity))
            
            # Update deployment
            self.apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Kubernetes scaling failed: {e}")
            return False


class AWSResourceManager:
    """Manages AWS EC2 and ECS resources."""
    
    def __init__(self):
        if not AWS_AVAILABLE:
            self.available = False
            return
        
        try:
            self.ec2_client = boto3.client('ec2')
            self.ecs_client = boto3.client('ecs')
            self.available = True
        except Exception:
            self.available = False
    
    def can_handle_resource(self, resource_type: ResourceType) -> bool:
        """Check if this manager can handle the resource type."""
        return self.available and resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]
    
    def scale_resource(self, resource_type: ResourceType, target_capacity: float,
                      metadata: Dict[str, Any]) -> bool:
        """Scale AWS resources."""
        if not self.available:
            return False
        
        # Implementation would depend on specific AWS service being used
        # This is a placeholder for the actual implementation
        logging.info(f"AWS scaling {resource_type.value} to {target_capacity}")
        return True


class GCPResourceManager:
    """Manages Google Cloud Platform resources."""
    
    def __init__(self):
        if not GCP_AVAILABLE:
            self.available = False
            return
        
        try:
            self.compute_client = compute_v1.InstancesClient()
            self.available = True
        except Exception:
            self.available = False
    
    def can_handle_resource(self, resource_type: ResourceType) -> bool:
        """Check if this manager can handle the resource type."""
        return self.available and resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]
    
    def scale_resource(self, resource_type: ResourceType, target_capacity: float,
                      metadata: Dict[str, Any]) -> bool:
        """Scale GCP resources."""
        if not self.available:
            return False
        
        # Implementation would depend on specific GCP service being used
        logging.info(f"GCP scaling {resource_type.value} to {target_capacity}")
        return True


class AzureResourceManager:
    """Manages Microsoft Azure resources."""
    
    def __init__(self):
        if not AZURE_AVAILABLE:
            self.available = False
            return
        
        try:
            # Azure client initialization would go here
            self.available = False  # Placeholder
        except Exception:
            self.available = False
    
    def can_handle_resource(self, resource_type: ResourceType) -> bool:
        """Check if this manager can handle the resource type."""
        return self.available and resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]
    
    def scale_resource(self, resource_type: ResourceType, target_capacity: float,
                      metadata: Dict[str, Any]) -> bool:
        """Scale Azure resources."""
        if not self.available:
            return False
        
        # Implementation would depend on specific Azure service being used
        logging.info(f"Azure scaling {resource_type.value} to {target_capacity}")
        return True


# Global autoscaler instance
_global_autoscaler = None

def get_global_autoscaler() -> AutoScaler:
    """Get or create global autoscaler instance."""
    global _global_autoscaler
    if _global_autoscaler is None:
        _global_autoscaler = AutoScaler()
    return _global_autoscaler

def start_autoscaling():
    """Start global autoscaling."""
    autoscaler = get_global_autoscaler()
    autoscaler.start()

def stop_autoscaling():
    """Stop global autoscaling."""
    autoscaler = get_global_autoscaler()
    autoscaler.stop()