"""
Autonomous Auto-scaling and Load Balancing
==========================================

Intelligent auto-scaling, load balancing, and distributed resource management
for high-performance photonic neural network processing.
"""

import time
import threading
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import psutil

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"

@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    network_io: float = 0.0
    disk_io: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ScalingEvent:
    """Auto-scaling event record."""
    timestamp: float
    direction: ScalingDirection
    resource_type: ResourceType
    trigger_value: float
    threshold: float
    instances_before: int
    instances_after: int
    reason: str

@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""
    node_id: str
    host: str
    port: int
    status: str  # 'active', 'idle', 'busy', 'failed'
    current_load: float
    max_capacity: int
    current_tasks: int
    last_heartbeat: float
    resource_metrics: Optional[ResourceMetrics] = None

class AutoScaler:
    """Autonomous auto-scaling system with predictive capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scaling_enabled = self.config.get("scaling_enabled", True)
        self.min_instances = self.config.get("min_instances", 1)
        self.max_instances = self.config.get("max_instances", 10)
        
        # Scaling thresholds
        self.scale_up_threshold = self.config.get("scale_up_threshold", 80.0)
        self.scale_down_threshold = self.config.get("scale_down_threshold", 30.0)
        self.stability_window = self.config.get("stability_window", 300)  # 5 minutes
        
        # Advanced scaling triggers
        self.queue_length_threshold = self.config.get("queue_length_threshold", 50)
        self.response_time_threshold = self.config.get("response_time_threshold", 2.0)
        self.error_rate_threshold = self.config.get("error_rate_threshold", 0.05)
        self.memory_pressure_threshold = self.config.get("memory_pressure_threshold", 85.0)
        
        # Predictive scaling
        self.enable_predictive_scaling = self.config.get("enable_predictive_scaling", True)
        self.prediction_window = self.config.get("prediction_window", 900)  # 15 minutes
        self.trend_sensitivity = self.config.get("trend_sensitivity", 0.1)
        
        # Current state
        self.current_instances = self.min_instances
        self.scaling_history = deque(maxlen=100)
        self.metrics_history = deque(maxlen=1000)
        self.last_scaling_time = 0
        self.cooldown_period = self.config.get("cooldown_period", 300)  # 5 minutes
        
        # Performance tracking
        self.task_queue_sizes = deque(maxlen=100)
        self.response_times = deque(maxlen=1000)
        self.error_rates = deque(maxlen=100)
        self.scaling_effectiveness = {}
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        self.metrics_callback: Optional[Callable[[], Dict[str, Any]]] = None
        
    def set_scaling_callbacks(self, scale_up: Callable[[int], bool], scale_down: Callable[[int], bool]):
        """Set callbacks for scaling operations."""
        self.scale_up_callback = scale_up
        self.scale_down_callback = scale_down
    
    def start_monitoring(self, interval: float = 30.0):
        """Start monitoring for auto-scaling decisions."""
        if self.monitoring:
            return
            
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self._collect_metrics()
                    self._evaluate_scaling()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Auto-scaling monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Auto-scaling monitoring stopped")
    
    def _collect_metrics(self):
        """Collect current resource metrics."""
        try:
            metrics = ResourceMetrics(
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=psutil.virtual_memory().percent
            )
            
            # Add GPU metrics if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics.gpu_percent = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
            except ImportError:
                pass  # No GPU monitoring available
            
            self.metrics_history.append(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
    
    def _evaluate_scaling(self):
        """Evaluate if scaling is needed using multiple triggers."""
        if not self.scaling_enabled or len(self.metrics_history) < 3:
            return
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.cooldown_period:
            return
        
        # Collect comprehensive metrics
        scaling_signals = self._collect_scaling_signals()
        
        # Evaluate each trigger type
        scale_up_triggers = []
        scale_down_triggers = []
        
        # Resource utilization triggers
        if scaling_signals['avg_cpu'] > self.scale_up_threshold:
            scale_up_triggers.append(f"High CPU: {scaling_signals['avg_cpu']:.1f}%")
        elif scaling_signals['avg_cpu'] < self.scale_down_threshold:
            scale_down_triggers.append(f"Low CPU: {scaling_signals['avg_cpu']:.1f}%")
        
        if scaling_signals['avg_memory'] > self.memory_pressure_threshold:
            scale_up_triggers.append(f"Memory pressure: {scaling_signals['avg_memory']:.1f}%")
        
        # Queue length trigger
        if scaling_signals['queue_length'] > self.queue_length_threshold:
            scale_up_triggers.append(f"Queue backlog: {scaling_signals['queue_length']} tasks")
        
        # Response time trigger
        if scaling_signals['avg_response_time'] > self.response_time_threshold:
            scale_up_triggers.append(f"Slow response: {scaling_signals['avg_response_time']:.2f}s")
        
        # Error rate trigger
        if scaling_signals['error_rate'] > self.error_rate_threshold:
            scale_up_triggers.append(f"High error rate: {scaling_signals['error_rate']:.1%}")
        
        # Predictive scaling
        if self.enable_predictive_scaling:
            prediction_signals = self._evaluate_predictive_scaling()
            if prediction_signals['predicted_load_increase']:
                scale_up_triggers.append("Predicted load increase")
            elif prediction_signals['predicted_load_decrease']:
                scale_down_triggers.append("Predicted load decrease")
        
        # Make scaling decision
        if scale_up_triggers and self.current_instances < self.max_instances:
            primary_resource = self._identify_primary_constraint(
                scaling_signals['avg_cpu'], 
                scaling_signals['avg_memory'], 
                scaling_signals['avg_gpu']
            )
            primary_utilization = max(
                scaling_signals['avg_cpu'], 
                scaling_signals['avg_memory'], 
                scaling_signals['avg_gpu']
            )
            self._scale_up(primary_resource, primary_utilization, scale_up_triggers)
        elif scale_down_triggers and self.current_instances > self.min_instances:
            # Only scale down if ALL metrics are low
            if (scaling_signals['avg_cpu'] < self.scale_down_threshold and
                scaling_signals['avg_memory'] < self.scale_down_threshold and
                scaling_signals['queue_length'] < 10 and
                scaling_signals['error_rate'] < 0.01):
                primary_resource = ResourceType.CPU
                primary_utilization = scaling_signals['avg_cpu']
                self._scale_down(primary_resource, primary_utilization, scale_down_triggers)
    
    def _identify_primary_constraint(self, cpu: float, memory: float, gpu: float) -> ResourceType:
        """Identify the primary resource constraint."""
        max_utilization = max(cpu, memory, gpu)
        
        if cpu == max_utilization:
            return ResourceType.CPU
        elif memory == max_utilization:
            return ResourceType.MEMORY
        else:
            return ResourceType.GPU
    
    def _collect_scaling_signals(self) -> Dict[str, Any]:
        """Collect comprehensive scaling signals from multiple sources."""
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        
        # Calculate averages
        avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics]) if recent_metrics else 0
        avg_memory = statistics.mean([m.memory_percent for m in recent_metrics]) if recent_metrics else 0
        avg_gpu = statistics.mean([m.gpu_percent for m in recent_metrics]) if recent_metrics else 0
        
        # Get application metrics if available
        app_metrics = {}
        if self.metrics_callback:
            try:
                app_metrics = self.metrics_callback()
            except Exception as e:
                logger.warning(f"Failed to collect application metrics: {e}")
        
        # Extract application-specific signals
        queue_length = app_metrics.get('queue_length', 0)
        avg_response_time = app_metrics.get('avg_response_time', 0.0)
        error_rate = app_metrics.get('error_rate', 0.0)
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics[-5:]])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics[-5:]])
        
        return {
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'avg_gpu': avg_gpu,
            'queue_length': queue_length,
            'avg_response_time': avg_response_time,
            'error_rate': error_rate,
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'timestamp': time.time()
        }
    
    def _evaluate_predictive_scaling(self) -> Dict[str, bool]:
        """Evaluate predictive scaling signals."""
        if len(self.metrics_history) < 20:  # Need sufficient history
            return {'predicted_load_increase': False, 'predicted_load_decrease': False}
        
        # Analyze recent trends
        recent_cpu = [m.cpu_percent for m in list(self.metrics_history)[-20:]]
        recent_memory = [m.memory_percent for m in list(self.metrics_history)[-20:]]
        
        cpu_trend = self._calculate_trend(recent_cpu)
        memory_trend = self._calculate_trend(recent_memory)
        
        # Check if trends are accelerating
        cpu_acceleration = self._calculate_acceleration(recent_cpu)
        memory_acceleration = self._calculate_acceleration(recent_memory)
        
        # Predict load increase if strong upward trends
        predicted_load_increase = (
            (cpu_trend > self.trend_sensitivity and cpu_acceleration > 0) or
            (memory_trend > self.trend_sensitivity and memory_acceleration > 0)
        )
        
        # Predict load decrease if strong downward trends and low utilization
        predicted_load_decrease = (
            cpu_trend < -self.trend_sensitivity and 
            memory_trend < -self.trend_sensitivity and
            recent_cpu[-1] < 40 and recent_memory[-1] < 40
        )
        
        return {
            'predicted_load_increase': predicted_load_increase,
            'predicted_load_decrease': predicted_load_decrease,
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'cpu_acceleration': cpu_acceleration,
            'memory_acceleration': memory_acceleration
        }
    
    def _calculate_acceleration(self, values: List[float]) -> float:
        """Calculate acceleration (second derivative) of values."""
        if len(values) < 3:
            return 0.0
        
        # Calculate first derivatives (slopes)
        slopes = []
        for i in range(1, len(values)):
            slope = values[i] - values[i-1]
            slopes.append(slope)
        
        # Calculate second derivative (acceleration)
        if len(slopes) < 2:
            return 0.0
        
        accelerations = []
        for i in range(1, len(slopes)):
            acceleration = slopes[i] - slopes[i-1]
            accelerations.append(acceleration)
        
        return statistics.mean(accelerations) if accelerations else 0.0
    
    def _scale_up(self, resource_type: ResourceType, utilization: float, triggers: List[str] = None):
        """Scale up instances."""
        new_instances = min(self.current_instances + 1, self.max_instances)
        
        if self.scale_up_callback:
            success = self.scale_up_callback(new_instances)
            
            if success:
                trigger_reasons = "; ".join(triggers) if triggers else f"High {resource_type.value} utilization: {utilization:.1f}%"
                
                scaling_event = ScalingEvent(
                    timestamp=time.time(),
                    direction=ScalingDirection.UP,
                    resource_type=resource_type,
                    trigger_value=utilization,
                    threshold=self.scale_up_threshold,
                    instances_before=self.current_instances,
                    instances_after=new_instances,
                    reason=trigger_reasons
                )
                
                self.current_instances = new_instances
                self.last_scaling_time = time.time()
                self.scaling_history.append(scaling_event)
                
                logger.info(f"Scaled up to {new_instances} instances: {trigger_reasons}")
                
                # Track scaling effectiveness
                self._track_scaling_effectiveness(scaling_event)
    
    def _scale_down(self, resource_type: ResourceType, utilization: float, triggers: List[str] = None):
        """Scale down instances."""
        new_instances = max(self.current_instances - 1, self.min_instances)
        
        if self.scale_down_callback:
            success = self.scale_down_callback(new_instances)
            
            if success:
                trigger_reasons = "; ".join(triggers) if triggers else f"Low {resource_type.value} utilization: {utilization:.1f}%"
                
                scaling_event = ScalingEvent(
                    timestamp=time.time(),
                    direction=ScalingDirection.DOWN,
                    resource_type=resource_type,
                    trigger_value=utilization,
                    threshold=self.scale_down_threshold,
                    instances_before=self.current_instances,
                    instances_after=new_instances,
                    reason=trigger_reasons
                )
                
                self.current_instances = new_instances
                self.last_scaling_time = time.time()
                self.scaling_history.append(scaling_event)
                
                logger.info(f"Scaled down to {new_instances} instances: {trigger_reasons}")
                
                # Track scaling effectiveness
                self._track_scaling_effectiveness(scaling_event)
    
    def _track_scaling_effectiveness(self, scaling_event: ScalingEvent):
        """Track the effectiveness of scaling decisions."""
        event_id = f"{scaling_event.timestamp}_{scaling_event.direction.value}"
        self.scaling_effectiveness[event_id] = {
            'event': scaling_event,
            'pre_scaling_metrics': list(self.metrics_history)[-5:] if self.metrics_history else [],
            'post_scaling_metrics': [],  # Will be filled later
            'effectiveness_score': None
        }
        
        # Schedule effectiveness evaluation (would be done by monitoring thread)
        logger.debug(f"Tracking effectiveness for scaling event: {event_id}")
    
    def evaluate_scaling_effectiveness(self):
        """Evaluate the effectiveness of recent scaling decisions."""
        current_time = time.time()
        
        for event_id, tracking_data in list(self.scaling_effectiveness.items()):
            event = tracking_data['event']
            
            # Evaluate effectiveness after sufficient time has passed
            if current_time - event.timestamp > 600:  # 10 minutes
                # Collect post-scaling metrics
                if not tracking_data['post_scaling_metrics'] and self.metrics_history:
                    tracking_data['post_scaling_metrics'] = list(self.metrics_history)[-5:]
                
                # Calculate effectiveness score
                effectiveness = self._calculate_effectiveness_score(tracking_data)
                tracking_data['effectiveness_score'] = effectiveness
                
                logger.info(f"Scaling effectiveness for {event_id}: {effectiveness:.2f}")
                
                # Clean up old tracking data
                if current_time - event.timestamp > 3600:  # 1 hour
                    del self.scaling_effectiveness[event_id]
    
    def _calculate_effectiveness_score(self, tracking_data: Dict[str, Any]) -> float:
        """Calculate effectiveness score for a scaling decision."""
        pre_metrics = tracking_data['pre_scaling_metrics']
        post_metrics = tracking_data['post_scaling_metrics']
        event = tracking_data['event']
        
        if not pre_metrics or not post_metrics:
            return 0.5  # Neutral score if insufficient data
        
        # Calculate average utilization before and after
        pre_cpu = statistics.mean([m.cpu_percent for m in pre_metrics])
        post_cpu = statistics.mean([m.cpu_percent for m in post_metrics])
        
        pre_memory = statistics.mean([m.memory_percent for m in pre_metrics])
        post_memory = statistics.mean([m.memory_percent for m in post_metrics])
        
        if event.direction == ScalingDirection.UP:
            # For scale-up, effectiveness is measured by utilization reduction
            cpu_improvement = max(0, pre_cpu - post_cpu)
            memory_improvement = max(0, pre_memory - post_memory)
            
            # Good if utilization decreased significantly
            effectiveness = min(1.0, (cpu_improvement + memory_improvement) / 40.0)
            
        else:  # Scale down
            # For scale-down, effectiveness is measured by maintaining low utilization
            # without causing spikes
            max_post_cpu = max([m.cpu_percent for m in post_metrics])
            max_post_memory = max([m.memory_percent for m in post_metrics])
            
            # Good if utilization stayed low after scaling down
            if max_post_cpu < 70 and max_post_memory < 70:
                effectiveness = 1.0
            elif max_post_cpu < 85 and max_post_memory < 85:
                effectiveness = 0.7
            else:
                effectiveness = 0.3
        
        return effectiveness
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on current state."""
        if len(self.metrics_history) < 5:
            return {"recommendation": "insufficient_data"}
        
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements
        
        # Calculate statistics
        avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        
        recommendations = {
            "current_instances": self.current_instances,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "cpu_trend": cpu_trend,
            "memory_trend": memory_trend,
            "recommended_action": "none"
        }
        
        # Generate recommendations
        if avg_cpu > 85 or avg_memory > 85:
            recommendations["recommended_action"] = "scale_up"
            recommendations["reason"] = "High resource utilization"
        elif avg_cpu < 25 and avg_memory < 25 and self.current_instances > self.min_instances:
            recommendations["recommended_action"] = "scale_down"
            recommendations["reason"] = "Low resource utilization"
        elif cpu_trend > 0.1 or memory_trend > 0.1:
            recommendations["recommended_action"] = "prepare_scale_up"
            recommendations["reason"] = "Increasing resource trend"
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction for values."""
        if len(values) < 3:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for node health management."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0
    success_count: int = 0
    half_open_time: float = 0

class LoadBalancer:
    """Intelligent load balancer with adaptive routing, circuit breakers, and failover."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.nodes: Dict[str, WorkerNode] = {}
        self.routing_algorithm = self.config.get("algorithm", "weighted_round_robin")
        self.health_check_interval = self.config.get("health_check_interval", 30.0)
        
        # Circuit breaker configuration
        self.circuit_breaker_enabled = self.config.get("circuit_breaker_enabled", True)
        self.failure_threshold = self.config.get("failure_threshold", 5)
        self.timeout_duration = self.config.get("timeout_duration", 60.0)  # seconds
        self.half_open_success_threshold = self.config.get("half_open_success_threshold", 3)
        
        # Load balancing state
        self.current_node_index = 0
        self.request_history = deque(maxlen=1000)
        self.node_performance = defaultdict(lambda: deque(maxlen=100))
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.node_weights = defaultdict(lambda: 1.0)  # Dynamic weights
        self.sticky_sessions = {}  # Session affinity
        
        # Advanced features
        self.enable_geographic_routing = self.config.get("enable_geographic_routing", False)
        self.enable_cost_aware_routing = self.config.get("enable_cost_aware_routing", False)
        self.latency_threshold = self.config.get("latency_threshold", 1.0)  # seconds
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread = None
        
    def register_node(self, node: WorkerNode):
        """Register a new worker node."""
        self.nodes[node.node_id] = node
        self.circuit_breakers[node.node_id] = CircuitBreakerState()
        self.node_weights[node.node_id] = 1.0
        logger.info(f"Registered worker node: {node.node_id} at {node.host}:{node.port}")
    
    def unregister_node(self, node_id: str):
        """Unregister a worker node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.circuit_breakers.pop(node_id, None)
            self.node_weights.pop(node_id, None)
            logger.info(f"Unregistered worker node: {node_id}")
    
    def get_available_nodes(self) -> List[WorkerNode]:
        """Get list of available worker nodes, considering circuit breaker states."""
        available_nodes = []
        current_time = time.time()
        
        for node in self.nodes.values():
            # Basic health checks
            if (node.status not in ['active', 'idle'] or 
                current_time - node.last_heartbeat > 60):
                continue
            
            # Circuit breaker checks
            if self.circuit_breaker_enabled:
                circuit_state = self.circuit_breakers.get(node.node_id)
                if circuit_state and self._is_circuit_breaker_open(node.node_id, current_time):
                    continue
            
            available_nodes.append(node)
        
        return available_nodes
    
    def _is_circuit_breaker_open(self, node_id: str, current_time: float) -> bool:
        """Check if circuit breaker is open for a node."""
        circuit_state = self.circuit_breakers.get(node_id)
        if not circuit_state:
            return False
        
        # Circuit is closed - normal operation
        if not circuit_state.is_open:
            return False
        
        # Circuit is open - check if timeout period has passed
        if current_time - circuit_state.last_failure_time > self.timeout_duration:
            # Transition to half-open state
            circuit_state.is_open = False
            circuit_state.half_open_time = current_time
            circuit_state.success_count = 0
            logger.info(f"Circuit breaker for node {node_id} transitioned to half-open")
            return False
        
        return True
    
    def _update_circuit_breaker(self, node_id: str, success: bool, response_time: float = 0.0):
        """Update circuit breaker state based on request outcome."""
        if not self.circuit_breaker_enabled:
            return
        
        circuit_state = self.circuit_breakers.get(node_id)
        if not circuit_state:
            return
        
        current_time = time.time()
        
        if success and response_time < self.latency_threshold:
            # Successful request
            if circuit_state.is_open:
                # In half-open state, count successes
                circuit_state.success_count += 1
                if circuit_state.success_count >= self.half_open_success_threshold:
                    # Close the circuit
                    circuit_state.is_open = False
                    circuit_state.failure_count = 0
                    circuit_state.success_count = 0
                    logger.info(f"Circuit breaker for node {node_id} closed")
            else:
                # Reset failure count on success
                circuit_state.failure_count = max(0, circuit_state.failure_count - 1)
        else:
            # Failed request or slow response
            circuit_state.failure_count += 1
            circuit_state.last_failure_time = current_time
            
            if circuit_state.failure_count >= self.failure_threshold:
                # Open the circuit
                circuit_state.is_open = True
                logger.warning(f"Circuit breaker opened for node {node_id} after {circuit_state.failure_count} failures")
    
    def _update_node_weights(self, node_id: str, response_time: float, success: bool):
        """Update dynamic node weights based on performance."""
        if node_id not in self.node_weights:
            return
        
        current_weight = self.node_weights[node_id]
        
        if success:
            # Increase weight for fast responses
            if response_time < 0.5:
                new_weight = min(2.0, current_weight * 1.1)
            elif response_time < 1.0:
                new_weight = current_weight
            else:
                new_weight = max(0.5, current_weight * 0.95)
        else:
            # Decrease weight for failures
            new_weight = max(0.1, current_weight * 0.8)
        
        self.node_weights[node_id] = new_weight
    
    def select_node(self, task_requirements: Optional[Dict[str, Any]] = None) -> Optional[WorkerNode]:
        """Select optimal node for task execution."""
        available_nodes = self.get_available_nodes()
        
        if not available_nodes:
            return None
        
        if self.routing_algorithm == "round_robin":
            return self._round_robin_selection(available_nodes)
        elif self.routing_algorithm == "weighted_round_robin":
            return self._weighted_round_robin_selection(available_nodes)
        elif self.routing_algorithm == "least_connections":
            return self._least_connections_selection(available_nodes)
        elif self.routing_algorithm == "least_response_time":
            return self._least_response_time_selection(available_nodes)
        elif self.routing_algorithm == "resource_aware":
            return self._resource_aware_selection(available_nodes, task_requirements)
        else:
            return available_nodes[0]  # Default: first available
    
    def _round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Simple round-robin selection."""
        node = nodes[self.current_node_index % len(nodes)]
        self.current_node_index += 1
        return node
    
    def _weighted_round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin based on node capacity."""
        # Calculate weights based on available capacity
        weights = []
        for node in nodes:
            available_capacity = node.max_capacity - node.current_tasks
            weight = max(1, available_capacity)  # Minimum weight of 1
            weights.append(weight)
        
        # Select based on weighted distribution
        total_weight = sum(weights)
        if total_weight == 0:
            return nodes[0]
        
        # Simple weighted selection
        selection_value = (self.current_node_index % total_weight)
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if selection_value < cumulative_weight:
                self.current_node_index += 1
                return nodes[i]
        
        return nodes[0]
    
    def _least_connections_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least active connections."""
        return min(nodes, key=lambda n: n.current_tasks)
    
    def _least_response_time_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with best response time."""
        best_node = nodes[0]
        best_avg_time = float('inf')
        
        for node in nodes:
            if node.node_id in self.node_performance:
                recent_times = list(self.node_performance[node.node_id])[-10:]
                if recent_times:
                    avg_time = statistics.mean(recent_times)
                    if avg_time < best_avg_time:
                        best_avg_time = avg_time
                        best_node = node
        
        return best_node
    
    def _resource_aware_selection(self, nodes: List[WorkerNode], requirements: Optional[Dict[str, Any]]) -> WorkerNode:
        """Select node based on resource requirements and availability."""
        if not requirements:
            return self._weighted_round_robin_selection(nodes)
        
        # Score nodes based on resource availability vs requirements
        best_node = None
        best_score = -1
        
        for node in nodes:
            if not node.resource_metrics:
                continue
            
            score = self._calculate_resource_score(node, requirements)
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node or nodes[0]
    
    def _calculate_resource_score(self, node: WorkerNode, requirements: Dict[str, Any]) -> float:
        """Calculate resource compatibility score."""
        if not node.resource_metrics:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        # CPU score
        if 'cpu_required' in requirements:
            cpu_available = 100 - node.resource_metrics.cpu_percent
            cpu_required = requirements['cpu_required']
            if cpu_available >= cpu_required:
                score += (cpu_available - cpu_required) * 0.4
            total_weight += 0.4
        
        # Memory score
        if 'memory_required' in requirements:
            memory_available = 100 - node.resource_metrics.memory_percent
            memory_required = requirements['memory_required']
            if memory_available >= memory_required:
                score += (memory_available - memory_required) * 0.4
            total_weight += 0.4
        
        # Task capacity score
        capacity_available = node.max_capacity - node.current_tasks
        if capacity_available > 0:
            score += capacity_available * 0.2
        total_weight += 0.2
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def record_task_completion(self, node_id: str, execution_time: float, success: bool):
        """Record task completion metrics for learning."""
        if success:
            self.node_performance[node_id].append(execution_time)
        
        # Update circuit breaker and node weights
        self._update_circuit_breaker(node_id, success, execution_time)
        self._update_node_weights(node_id, execution_time, success)
        
        # Update node status
        if node_id in self.nodes:
            self.nodes[node_id].current_tasks = max(0, self.nodes[node_id].current_tasks - 1)
        
        # Log performance for monitoring
        status_str = "SUCCESS" if success else "FAILED"
        logger.debug(f"Task completed on {node_id}: {status_str} in {execution_time:.2f}s")
    
    def start_health_monitoring(self):
        """Start health check monitoring for nodes."""
        if self.monitoring:
            return
            
        self.monitoring = True
        
        def health_check_loop():
            while self.monitoring:
                try:
                    self._perform_health_checks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    time.sleep(self.health_check_interval)
        
        self.monitor_thread = threading.Thread(target=health_check_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Load balancer health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop health check monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Load balancer health monitoring stopped")
    
    def _perform_health_checks(self):
        """Perform health checks on all nodes."""
        current_time = time.time()
        
        for node in self.nodes.values():
            # Check if node is still alive based on heartbeat
            if current_time - node.last_heartbeat > 90:  # 1.5 minutes
                if node.status != 'failed':
                    node.status = 'failed'
                    logger.warning(f"Node {node.node_id} marked as failed (no heartbeat)")
            elif node.status == 'failed' and current_time - node.last_heartbeat < 60:
                node.status = 'active'
                logger.info(f"Node {node.node_id} recovered")


class ResourceManager:
    """Intelligent resource management and allocation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.resource_pools = defaultdict(list)
        self.allocations = {}
        self.allocation_history = deque(maxlen=1000)
        
    def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> bool:
        """Allocate resources for a task."""
        required_resources = self._calculate_required_resources(requirements)
        
        if self._check_resource_availability(required_resources):
            allocation = {
                'task_id': task_id,
                'resources': required_resources,
                'timestamp': time.time()
            }
            
            self.allocations[task_id] = allocation
            self.allocation_history.append(allocation)
            
            logger.info(f"Allocated resources for task {task_id}")
            return True
        
        return False
    
    def release_resources(self, task_id: str):
        """Release resources allocated to a task."""
        if task_id in self.allocations:
            del self.allocations[task_id]
            logger.info(f"Released resources for task {task_id}")
    
    def _calculate_required_resources(self, requirements: Dict[str, Any]) -> Dict[str, float]:
        """Calculate actual resource requirements."""
        # Simplified resource calculation
        return {
            'cpu_cores': requirements.get('cpu_cores', 1),
            'memory_gb': requirements.get('memory_gb', 1),
            'gpu_memory_gb': requirements.get('gpu_memory_gb', 0)
        }
    
    def _check_resource_availability(self, required: Dict[str, float]) -> bool:
        """Check if required resources are available."""
        # Simplified availability check
        total_allocated_cpu = sum(
            alloc['resources'].get('cpu_cores', 0) 
            for alloc in self.allocations.values()
        )
        
        total_allocated_memory = sum(
            alloc['resources'].get('memory_gb', 0) 
            for alloc in self.allocations.values()
        )
        
        max_cpu = psutil.cpu_count()
        max_memory = psutil.virtual_memory().total / (1024**3)  # GB
        
        return (total_allocated_cpu + required.get('cpu_cores', 0) <= max_cpu and
                total_allocated_memory + required.get('memory_gb', 0) <= max_memory)


@dataclass
class TaskResult:
    """Result of a distributed task execution."""
    task_id: str
    success: bool
    result_data: Any
    execution_time: float
    node_id: str
    error_message: Optional[str] = None

@dataclass 
class DistributedTask:
    """Represents a distributed task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    requirements: Dict[str, Any]
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    timeout: float = 300.0  # 5 minutes
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable[[TaskResult], None]] = None

class DistributedOrchestrator:
    """Orchestrates distributed processing across multiple nodes with advanced features."""
    
    def __init__(self, autoscaler: AutoScaler, load_balancer: LoadBalancer, 
                 resource_manager: ResourceManager):
        self.autoscaler = autoscaler
        self.load_balancer = load_balancer
        self.resource_manager = resource_manager
        
        # Task management
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.task_queue = deque()
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.failed_tasks: Dict[str, TaskResult] = {}
        
        # Advanced features
        self.enable_task_splitting = True
        self.enable_result_caching = True
        self.enable_priority_queuing = True
        self.enable_dependency_resolution = True
        
        # Monitoring and metrics
        self.task_metrics = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'avg_execution_time': 0.0,
            'queue_wait_times': deque(maxlen=1000)
        }
        
        # Task scheduling
        self.scheduler_thread = None
        self.scheduling_active = False
        
    def submit_distributed_task(self, task: DistributedTask) -> bool:
        """Submit a task for advanced distributed execution."""
        # Check if task can be executed (dependencies resolved)
        if not self._are_dependencies_resolved(task):
            self.task_queue.append(task)
            logger.info(f"Task {task.task_id} queued - waiting for dependencies")
            return True
        
        # Check if result is cached
        if self.enable_result_caching and self._check_result_cache(task):
            return True
        
        # Check if task should be split
        if self.enable_task_splitting and self._should_split_task(task):
            return self._split_and_submit_task(task)
        
        # Standard task submission
        return self._submit_single_task(task)
    
    def _submit_single_task(self, task: DistributedTask) -> bool:
        """Submit a single task for execution."""
        # Check resource requirements
        if not self.resource_manager.allocate_resources(task.task_id, task.requirements):
            logger.warning(f"Failed to allocate resources for task {task.task_id}")
            return False
        
        # Select optimal node
        selected_node = self.load_balancer.select_node(task.requirements)
        
        if not selected_node:
            logger.warning(f"No available nodes for task {task.task_id}")
            self.resource_manager.release_resources(task.task_id)
            
            # Retry later if no nodes available
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.task_queue.append(task)
                logger.info(f"Task {task.task_id} re-queued for retry {task.retry_count}")
            
            return False
        
        # Track task
        task_info = {
            'task': task,
            'node_id': selected_node.node_id,
            'start_time': time.time(),
            'submit_time': time.time()
        }
        self.active_tasks[task.task_id] = task_info
        
        # Update metrics
        self.task_metrics['total_submitted'] += 1
        
        # Update node state
        selected_node.current_tasks += 1
        
        logger.info(f"Submitted task {task.task_id} to node {selected_node.node_id}")
        return True
    
    def _are_dependencies_resolved(self, task: DistributedTask) -> bool:
        """Check if all task dependencies are resolved."""
        if not self.enable_dependency_resolution or not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    def _check_result_cache(self, task: DistributedTask) -> bool:
        """Check if task result is available in cache."""
        # Simple cache key based on task type and payload hash
        cache_key = f"{task.task_type}_{hash(str(sorted(task.payload.items())))}"
        
        # In a real implementation, this would check a distributed cache
        # For now, just return False to always execute
        return False
    
    def _should_split_task(self, task: DistributedTask) -> bool:
        """Determine if a task should be split into smaller tasks."""
        # Check if task type supports splitting
        splittable_types = ['batch_processing', 'data_analysis', 'neural_training']
        
        if task.task_type not in splittable_types:
            return False
        
        # Check if payload is large enough to benefit from splitting
        payload_size = len(str(task.payload))
        return payload_size > 10000  # Arbitrary threshold
    
    def _split_and_submit_task(self, task: DistributedTask) -> bool:
        """Split a large task into smaller subtasks."""
        try:
            # Generate subtasks (simplified implementation)
            subtasks = self._generate_subtasks(task)
            
            if len(subtasks) <= 1:
                return self._submit_single_task(task)
            
            # Submit all subtasks
            success_count = 0
            for subtask in subtasks:
                if self._submit_single_task(subtask):
                    success_count += 1
            
            logger.info(f"Split task {task.task_id} into {len(subtasks)} subtasks, {success_count} submitted successfully")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to split task {task.task_id}: {e}")
            return self._submit_single_task(task)
    
    def _generate_subtasks(self, task: DistributedTask) -> List[DistributedTask]:
        """Generate subtasks from a large task."""
        # Simplified task splitting - in practice this would be task-type specific
        subtasks = []
        
        if task.task_type == 'batch_processing' and 'data_chunks' in task.payload:
            chunks = task.payload['data_chunks']
            chunk_size = max(1, len(chunks) // 4)  # Split into 4 parts
            
            for i in range(0, len(chunks), chunk_size):
                chunk = chunks[i:i + chunk_size]
                subtask_payload = task.payload.copy()
                subtask_payload['data_chunks'] = chunk
                
                subtask = DistributedTask(
                    task_id=f"{task.task_id}_part_{i // chunk_size}",
                    task_type=task.task_type,
                    payload=subtask_payload,
                    requirements=task.requirements,
                    priority=task.priority,
                    max_retries=task.max_retries,
                    timeout=task.timeout
                )
                subtasks.append(subtask)
        
        return subtasks if subtasks else [task]
    
    def start_task_scheduler(self):
        """Start the task scheduler for processing queued tasks."""
        if self.scheduling_active:
            return
        
        self.scheduling_active = True
        
        def scheduler_loop():
            while self.scheduling_active:
                try:
                    self._process_task_queue()
                    self._check_task_timeouts()
                    self._retry_failed_tasks()
                    time.sleep(1.0)  # Check every second
                except Exception as e:
                    logger.error(f"Task scheduler error: {e}")
                    time.sleep(5.0)
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Task scheduler started")
    
    def stop_task_scheduler(self):
        """Stop the task scheduler."""
        self.scheduling_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        logger.info("Task scheduler stopped")
    
    def _process_task_queue(self):
        """Process queued tasks that are ready for execution."""
        if not self.task_queue:
            return
        
        ready_tasks = []
        remaining_tasks = deque()
        
        # Sort by priority if enabled
        if self.enable_priority_queuing:
            queue_items = sorted(self.task_queue, key=lambda t: t.priority, reverse=True)
        else:
            queue_items = list(self.task_queue)
        
        for task in queue_items:
            if self._are_dependencies_resolved(task):
                ready_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        # Update queue with remaining tasks
        self.task_queue = remaining_tasks
        
        # Submit ready tasks
        for task in ready_tasks:
            self._submit_single_task(task)
    
    def _check_task_timeouts(self):
        """Check for tasks that have exceeded their timeout."""
        current_time = time.time()
        timed_out_tasks = []
        
        for task_id, task_info in self.active_tasks.items():
            task = task_info['task']
            if current_time - task_info['start_time'] > task.timeout:
                timed_out_tasks.append(task_id)
        
        for task_id in timed_out_tasks:
            self._handle_task_timeout(task_id)
    
    def _handle_task_timeout(self, task_id: str):
        """Handle a task that has timed out."""
        if task_id not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_id]
        task = task_info['task']
        
        logger.warning(f"Task {task_id} timed out after {task.timeout}s")
        
        # Create timeout result
        result = TaskResult(
            task_id=task_id,
            success=False,
            result_data=None,
            execution_time=task.timeout,
            node_id=task_info['node_id'],
            error_message="Task timeout"
        )
        
        self._complete_task_with_result(task_id, result)
    
    def _retry_failed_tasks(self):
        """Retry failed tasks that haven't exceeded max retries."""
        retry_tasks = []
        
        for task_id, result in list(self.failed_tasks.items()):
            # Find the original task if it exists
            # In practice, you'd store the task definition for retries
            # For now, just log the retry intent
            logger.debug(f"Would retry failed task {task_id}")
        
        # Clear old failed tasks (simplified)
        if len(self.failed_tasks) > 100:
            oldest_tasks = sorted(self.failed_tasks.items(), 
                                key=lambda x: x[1].execution_time)[:50]
            for task_id, _ in oldest_tasks:
                del self.failed_tasks[task_id]
    
    def complete_distributed_task(self, task_id: str, result_data: Any = None, success: bool = True, error_message: str = None):
        """Mark a distributed task as completed with full result tracking."""
        if task_id not in self.active_tasks:
            logger.warning(f"Attempted to complete unknown task: {task_id}")
            return
        
        task_info = self.active_tasks[task_id]
        task = task_info['task']
        execution_time = time.time() - task_info['start_time']
        
        # Create result object
        result = TaskResult(
            task_id=task_id,
            success=success,
            result_data=result_data,
            execution_time=execution_time,
            node_id=task_info['node_id'],
            error_message=error_message
        )
        
        self._complete_task_with_result(task_id, result)
    
    def _complete_task_with_result(self, task_id: str, result: TaskResult):
        """Internal method to complete a task with a result object."""
        if task_id not in self.active_tasks:
            return
        
        task_info = self.active_tasks[task_id]
        task = task_info['task']
        
        # Record performance metrics
        self.load_balancer.record_task_completion(
            result.node_id, 
            result.execution_time, 
            result.success
        )
        
        # Release resources
        self.resource_manager.release_resources(task_id)
        
        # Store result
        if result.success:
            self.completed_tasks[task_id] = result
            self.task_metrics['total_completed'] += 1
            
            # Update average execution time
            current_avg = self.task_metrics['avg_execution_time']
            total_completed = self.task_metrics['total_completed']
            self.task_metrics['avg_execution_time'] = (
                (current_avg * (total_completed - 1) + result.execution_time) / total_completed
            )
        else:
            self.failed_tasks[task_id] = result
            self.task_metrics['total_failed'] += 1
        
        # Execute callback if provided
        if task.callback:
            try:
                task.callback(result)
            except Exception as e:
                logger.error(f"Task callback failed for {task_id}: {e}")
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        status_str = "SUCCESS" if result.success else "FAILED"
        logger.info(f"Task {task_id} completed: {status_str} in {result.execution_time:.2f}s")
        
        # Process any dependent tasks that might now be ready
        self._process_task_queue()
    
    def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all components."""
        available_nodes = self.load_balancer.get_available_nodes()
        scaling_recommendations = self.autoscaler.get_scaling_recommendations()
        
        # Calculate system utilization
        total_capacity = sum(node.max_capacity for node in self.load_balancer.nodes.values())
        active_tasks = sum(node.current_tasks for node in self.load_balancer.nodes.values())
        system_utilization = (active_tasks / total_capacity * 100) if total_capacity > 0 else 0
        
        # Calculate queue metrics
        avg_queue_wait = 0.0
        if self.task_metrics['queue_wait_times']:
            avg_queue_wait = statistics.mean(self.task_metrics['queue_wait_times'])
        
        # Calculate success rate
        total_completed = self.task_metrics['total_completed'] + self.task_metrics['total_failed']
        success_rate = (self.task_metrics['total_completed'] / total_completed * 100) if total_completed > 0 else 100
        
        return {
            'cluster_info': {
                'total_nodes': len(self.load_balancer.nodes),
                'available_nodes': len(available_nodes),
                'failed_nodes': len([n for n in self.load_balancer.nodes.values() if n.status == 'failed']),
                'system_utilization_percent': system_utilization
            },
            'task_metrics': {
                'active_tasks': len(self.active_tasks),
                'queued_tasks': len(self.task_queue),
                'completed_tasks': self.task_metrics['total_completed'],
                'failed_tasks': self.task_metrics['total_failed'],
                'success_rate_percent': success_rate,
                'avg_execution_time_seconds': self.task_metrics['avg_execution_time'],
                'avg_queue_wait_seconds': avg_queue_wait
            },
            'autoscaling': {
                'current_instances': self.autoscaler.current_instances,
                'scaling_enabled': self.autoscaler.scaling_enabled,
                'recommendations': scaling_recommendations
            },
            'load_balancing': {
                'algorithm': self.load_balancer.routing_algorithm,
                'circuit_breaker_enabled': self.load_balancer.circuit_breaker_enabled,
                'active_circuit_breakers': len([cb for cb in self.load_balancer.circuit_breakers.values() if cb.is_open])
            },
            'resource_management': {
                'active_allocations': len(self.resource_manager.allocations),
                'allocation_history_size': len(self.resource_manager.allocation_history)
            }
        }
    
    # Keep the legacy method for backward compatibility
    def complete_task(self, task_id: str, success: bool = True):
        """Legacy method for backward compatibility."""
        self.complete_distributed_task(task_id, success=success)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        return self.get_comprehensive_system_status()