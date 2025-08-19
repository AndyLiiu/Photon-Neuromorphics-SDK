"""
Autonomous Optimization Engine
==============================

Advanced optimization engine with machine learning-driven performance tuning,
predictive optimization, and adaptive algorithm selection.

Features ML-driven optimization for autonomous performance improvements and predictive_optimization.
"""

import time
import threading
import numpy as np
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import pickle
import json

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    NETWORK = "network"
    ALGORITHM = "algorithm"
    CACHING = "caching"

class OptimizationStrategy(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"

@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    optimization_type: OptimizationType
    strategy: OptimizationStrategy
    performance_before: float
    performance_after: float
    improvement_percent: float
    execution_time: float
    success: bool
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

@dataclass
class PerformanceBenchmark:
    """Performance benchmark data."""
    operation_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

class PerformancePredictor:
    """Predicts performance impact of optimizations using machine learning."""
    
    def __init__(self):
        self.feature_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        self.model_trained = False
        self.prediction_accuracy = 0.0
        
    def add_training_data(self, features: Dict[str, float], performance: float):
        """Add training data for performance prediction."""
        feature_vector = self._extract_features(features)
        self.feature_history.append(feature_vector)
        self.performance_history.append(performance)
        
        # Retrain model periodically
        if len(self.feature_history) >= 50 and len(self.feature_history) % 10 == 0:
            self._train_model()
    
    def predict_performance(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Predict performance and confidence."""
        if not self.model_trained or len(self.feature_history) < 20:
            return 0.0, 0.0  # No prediction available
        
        feature_vector = self._extract_features(features)
        
        # Simple nearest neighbor prediction
        prediction, confidence = self._nearest_neighbor_predict(feature_vector)
        return prediction, confidence
    
    def _extract_features(self, features: Dict[str, float]) -> List[float]:
        """Extract feature vector from features dictionary."""
        # Standard feature extraction
        feature_keys = [
            'data_size', 'complexity', 'memory_available', 'cpu_cores',
            'cache_hit_ratio', 'parallel_workers', 'batch_size'
        ]
        
        return [features.get(key, 0.0) for key in feature_keys]
    
    def _train_model(self):
        """Train performance prediction model."""
        if len(self.feature_history) < 20:
            return
        
        # Simple implementation - in practice would use scikit-learn or similar
        self.model_trained = True
        
        # Calculate prediction accuracy on recent data
        if len(self.feature_history) >= 40:
            recent_features = list(self.feature_history)[-20:]
            recent_performance = list(self.performance_history)[-20:]
            
            errors = []
            for i, (features, actual) in enumerate(zip(recent_features, recent_performance)):
                predicted, _ = self._nearest_neighbor_predict(features, exclude_index=len(self.feature_history) - 20 + i)
                if predicted > 0:
                    error = abs(predicted - actual) / actual
                    errors.append(error)
            
            if errors:
                self.prediction_accuracy = 1.0 - statistics.mean(errors)
                logger.info(f"Performance prediction accuracy: {self.prediction_accuracy:.3f}")
    
    def _nearest_neighbor_predict(self, feature_vector: List[float], exclude_index: Optional[int] = None) -> Tuple[float, float]:
        """Predict using k-nearest neighbors."""
        if len(self.feature_history) < 5:
            return 0.0, 0.0
        
        distances = []
        for i, historical_features in enumerate(self.feature_history):
            if exclude_index is not None and i == exclude_index:
                continue
                
            distance = self._euclidean_distance(feature_vector, historical_features)
            distances.append((distance, self.performance_history[i]))
        
        # Get 5 nearest neighbors
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:min(5, len(distances))]
        
        if not k_nearest:
            return 0.0, 0.0
        
        # Weighted average based on distance
        total_weight = 0.0
        weighted_sum = 0.0
        
        for distance, performance in k_nearest:
            weight = 1.0 / (1.0 + distance)  # Inverse distance weighting
            weighted_sum += weight * performance
            total_weight += weight
        
        prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
        confidence = min(1.0, total_weight / len(k_nearest))
        
        return prediction, confidence
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between feature vectors."""
        if len(vec1) != len(vec2):
            return float('inf')
        
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


class AdaptiveOptimizer:
    """Adaptive optimizer that learns optimal configurations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.optimization_history = deque(maxlen=500)
        self.current_configurations = {}
        self.best_configurations = {}
        self.exploration_rate = self.config.get("exploration_rate", 0.1)
        
    def suggest_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimization configuration based on context."""
        operation_type = context.get('operation_type', 'default')
        
        # Check if we have a known best configuration
        if operation_type in self.best_configurations:
            best_config = self.best_configurations[operation_type]
            
            # Exploration vs exploitation
            if np.random.random() < self.exploration_rate:
                return self._explore_configuration(best_config)
            else:
                return best_config.copy()
        
        # No known configuration, start with defaults
        return self._default_configuration(context)
    
    def record_optimization_result(self, context: Dict[str, Any], configuration: Dict[str, Any], 
                                 performance: float, success: bool):
        """Record the result of an optimization attempt."""
        operation_type = context.get('operation_type', 'default')
        
        result = {
            'operation_type': operation_type,
            'context': context.copy(),
            'configuration': configuration.copy(),
            'performance': performance,
            'success': success,
            'timestamp': time.time()
        }
        
        self.optimization_history.append(result)
        
        # Update best configuration if this one is better
        if success and (operation_type not in self.best_configurations or
                       performance > self._get_best_performance(operation_type)):
            self.best_configurations[operation_type] = configuration.copy()
            logger.info(f"New best configuration for {operation_type}: {performance:.3f}")
    
    def _explore_configuration(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Explore variations of a known good configuration."""
        config = base_config.copy()
        
        # Add small random variations
        for key, value in config.items():
            if isinstance(value, (int, float)):
                variation = np.random.normal(0, 0.1) * value  # 10% std deviation
                if isinstance(value, int):
                    config[key] = max(1, int(value + variation))
                else:
                    config[key] = max(0.1, value + variation)
        
        return config
    
    def _default_configuration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default configuration based on context."""
        data_size = context.get('data_size', 1000)
        complexity = context.get('complexity', 1.0)
        
        # Adaptive defaults based on problem size
        if data_size < 1000:
            return {
                'batch_size': 32,
                'parallel_workers': 2,
                'cache_size': 100,
                'optimization_level': 1
            }
        elif data_size < 10000:
            return {
                'batch_size': 64,
                'parallel_workers': 4,
                'cache_size': 500,
                'optimization_level': 2
            }
        else:
            return {
                'batch_size': 128,
                'parallel_workers': 8,
                'cache_size': 1000,
                'optimization_level': 3
            }
    
    def _get_best_performance(self, operation_type: str) -> float:
        """Get best recorded performance for operation type."""
        best_performance = 0.0
        
        for result in self.optimization_history:
            if (result['operation_type'] == operation_type and 
                result['success'] and 
                result['performance'] > best_performance):
                best_performance = result['performance']
        
        return best_performance
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get insights from optimization history."""
        if not self.optimization_history:
            return {}
        
        insights = {
            'total_optimizations': len(self.optimization_history),
            'success_rate': 0.0,
            'average_improvement': 0.0,
            'best_configurations': self.best_configurations.copy()
        }
        
        successful_optimizations = [r for r in self.optimization_history if r['success']]
        if successful_optimizations:
            insights['success_rate'] = len(successful_optimizations) / len(self.optimization_history)
            insights['average_improvement'] = statistics.mean([r['performance'] for r in successful_optimizations])
        
        return insights


class OptimizationEngine:
    """Main optimization engine coordinating all optimization efforts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.predictor = PerformancePredictor()
        self.adaptive_optimizer = AdaptiveOptimizer(config)
        self.benchmarks = deque(maxlen=1000)
        self.optimization_results = deque(maxlen=500)
        
        # Optimization modules
        self.optimizers = {
            OptimizationType.MEMORY: self._optimize_memory,
            OptimizationType.CPU: self._optimize_cpu,
            OptimizationType.CACHING: self._optimize_caching,
            OptimizationType.ALGORITHM: self._optimize_algorithm
        }
        
        # Continuous optimization
        self.continuous_optimization = self.config.get("continuous_optimization", True)
        self.optimization_interval = self.config.get("optimization_interval", 3600)  # 1 hour
        self.optimization_thread = None
        self.running = False
        
    def start_continuous_optimization(self):
        """Start continuous optimization process."""
        if self.running or not self.continuous_optimization:
            return
        
        self.running = True
        
        def optimization_loop():
            while self.running:
                try:
                    self._run_optimization_cycle()
                    time.sleep(self.optimization_interval)
                except Exception as e:
                    logger.error(f"Continuous optimization error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
        logger.info("Continuous optimization started")
    
    def stop_continuous_optimization(self):
        """Stop continuous optimization process."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10.0)
        logger.info("Continuous optimization stopped")
    
    def optimize_operation(self, operation_name: str, context: Dict[str, Any], 
                          target_metric: str = "execution_time") -> OptimizationResult:
        """Optimize a specific operation."""
        logger.info(f"Optimizing operation: {operation_name}")
        
        # Get baseline performance
        baseline_performance = self._measure_baseline_performance(operation_name, context)
        
        # Get optimization suggestion
        suggested_config = self.adaptive_optimizer.suggest_optimization(context)
        
        # Predict performance impact
        features = self._extract_optimization_features(context, suggested_config)
        predicted_performance, confidence = self.predictor.predict_performance(features)
        
        # Apply optimization
        optimization_type = self._determine_optimization_type(context)
        optimizer_func = self.optimizers.get(optimization_type, self._optimize_generic)
        
        start_time = time.time()
        optimized_performance, success, details = optimizer_func(context, suggested_config)
        execution_time = time.time() - start_time
        
        # Calculate improvement
        if baseline_performance > 0 and success:
            improvement_percent = ((optimized_performance - baseline_performance) / baseline_performance) * 100
        else:
            improvement_percent = 0.0
        
        # Create result
        result = OptimizationResult(
            optimization_type=optimization_type,
            strategy=OptimizationStrategy.ADAPTIVE,
            performance_before=baseline_performance,
            performance_after=optimized_performance,
            improvement_percent=improvement_percent,
            execution_time=execution_time,
            success=success,
            details=details
        )
        
        # Record result for learning
        self.optimization_results.append(result)
        self.adaptive_optimizer.record_optimization_result(
            context, suggested_config, optimized_performance, success
        )
        self.predictor.add_training_data(features, optimized_performance)
        
        logger.info(f"Optimization complete: {improvement_percent:.1f}% improvement")
        return result
    
    def _run_optimization_cycle(self):
        """Run a complete optimization cycle."""
        logger.info("Running optimization cycle")
        
        # Analyze recent performance data
        recent_benchmarks = list(self.benchmarks)[-50:]  # Last 50 benchmarks
        
        if len(recent_benchmarks) < 10:
            return  # Not enough data
        
        # Identify operations that need optimization
        operations_to_optimize = self._identify_optimization_candidates(recent_benchmarks)
        
        for operation_name, context in operations_to_optimize:
            try:
                result = self.optimize_operation(operation_name, context)
                if result.success and result.improvement_percent > 5:
                    logger.info(f"Significant improvement achieved for {operation_name}: "
                              f"{result.improvement_percent:.1f}%")
            except Exception as e:
                logger.error(f"Failed to optimize {operation_name}: {e}")
    
    def _identify_optimization_candidates(self, benchmarks: List[PerformanceBenchmark]) -> List[Tuple[str, Dict[str, Any]]]:
        """Identify operations that would benefit from optimization."""
        operation_stats = defaultdict(list)
        
        # Group benchmarks by operation
        for benchmark in benchmarks:
            operation_stats[benchmark.operation_name].append(benchmark)
        
        candidates = []
        
        for operation_name, operation_benchmarks in operation_stats.items():
            if len(operation_benchmarks) < 5:
                continue  # Need enough data
            
            # Calculate performance statistics
            execution_times = [b.execution_time for b in operation_benchmarks]
            avg_time = statistics.mean(execution_times)
            std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            
            # Check if operation is slow or inconsistent
            if avg_time > 1.0 or (std_dev / avg_time > 0.3 if avg_time > 0 else False):
                context = self._build_optimization_context(operation_benchmarks)
                candidates.append((operation_name, context))
        
        return candidates
    
    def _build_optimization_context(self, benchmarks: List[PerformanceBenchmark]) -> Dict[str, Any]:
        """Build optimization context from benchmark data."""
        latest_benchmark = benchmarks[-1]
        
        return {
            'operation_type': latest_benchmark.operation_name,
            'avg_execution_time': statistics.mean([b.execution_time for b in benchmarks]),
            'avg_memory_usage': statistics.mean([b.memory_usage for b in benchmarks]),
            'avg_cpu_usage': statistics.mean([b.cpu_usage for b in benchmarks]),
            'data_size': latest_benchmark.context.get('data_size', 1000),
            'complexity': latest_benchmark.context.get('complexity', 1.0),
            'recent_performance_trend': self._calculate_performance_trend(benchmarks)
        }
    
    def _calculate_performance_trend(self, benchmarks: List[PerformanceBenchmark]) -> str:
        """Calculate performance trend (improving, declining, stable)."""
        if len(benchmarks) < 5:
            return "stable"
        
        execution_times = [b.execution_time for b in benchmarks]
        recent_times = execution_times[-5:]
        older_times = execution_times[:-5]
        
        recent_avg = statistics.mean(recent_times)
        older_avg = statistics.mean(older_times)
        
        change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        
        if change_percent > 10:
            return "declining"  # Getting slower
        elif change_percent < -10:
            return "improving"  # Getting faster
        else:
            return "stable"
    
    def _measure_baseline_performance(self, operation_name: str, context: Dict[str, Any]) -> float:
        """Measure baseline performance for an operation."""
        # In a real implementation, this would run the actual operation
        # For now, return a synthetic baseline based on context
        
        data_size = context.get('data_size', 1000)
        complexity = context.get('complexity', 1.0)
        
        # Synthetic baseline calculation
        baseline = (data_size / 1000) * complexity * 0.1  # Base execution time
        return baseline
    
    def _determine_optimization_type(self, context: Dict[str, Any]) -> OptimizationType:
        """Determine the primary optimization type needed."""
        memory_usage = context.get('avg_memory_usage', 0)
        cpu_usage = context.get('avg_cpu_usage', 0)
        
        if memory_usage > 80:
            return OptimizationType.MEMORY
        elif cpu_usage > 80:
            return OptimizationType.CPU
        else:
            return OptimizationType.ALGORITHM
    
    def _extract_optimization_features(self, context: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for performance prediction."""
        return {
            'data_size': context.get('data_size', 1000),
            'complexity': context.get('complexity', 1.0),
            'memory_available': 100 - context.get('avg_memory_usage', 0),
            'cpu_cores': config.get('parallel_workers', 1),
            'cache_hit_ratio': 0.8,  # Would be measured in practice
            'parallel_workers': config.get('parallel_workers', 1),
            'batch_size': config.get('batch_size', 32)
        }
    
    # Optimization methods
    def _optimize_memory(self, context: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Optimize memory usage."""
        # Simulate memory optimization
        baseline = context.get('avg_execution_time', 1.0)
        memory_reduction = config.get('cache_size', 100) / 1000.0
        
        optimized_time = baseline * (1 - memory_reduction * 0.2)  # 20% improvement per cache reduction
        success = optimized_time < baseline
        
        details = {
            'optimization_type': 'memory',
            'cache_size_optimized': config.get('cache_size', 100),
            'memory_reduction_percent': memory_reduction * 100
        }
        
        return optimized_time, success, details
    
    def _optimize_cpu(self, context: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Optimize CPU usage."""
        baseline = context.get('avg_execution_time', 1.0)
        parallel_workers = config.get('parallel_workers', 1)
        
        # Simulate parallel optimization with diminishing returns
        speedup = min(parallel_workers, 8) * 0.8  # 80% efficiency per worker
        optimized_time = baseline / (1 + speedup * 0.1)
        success = optimized_time < baseline
        
        details = {
            'optimization_type': 'cpu',
            'parallel_workers': parallel_workers,
            'estimated_speedup': speedup
        }
        
        return optimized_time, success, details
    
    def _optimize_caching(self, context: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Optimize caching strategy."""
        baseline = context.get('avg_execution_time', 1.0)
        cache_size = config.get('cache_size', 100)
        
        # Simulate cache optimization
        cache_efficiency = min(cache_size / 1000.0, 0.5)  # Max 50% improvement
        optimized_time = baseline * (1 - cache_efficiency)
        success = optimized_time < baseline
        
        details = {
            'optimization_type': 'caching',
            'cache_size': cache_size,
            'estimated_cache_efficiency': cache_efficiency
        }
        
        return optimized_time, success, details
    
    def _optimize_algorithm(self, context: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Optimize algorithm selection."""
        baseline = context.get('avg_execution_time', 1.0)
        optimization_level = config.get('optimization_level', 1)
        
        # Simulate algorithmic optimization
        improvement = optimization_level * 0.15  # 15% per level
        optimized_time = baseline * (1 - improvement)
        success = optimized_time < baseline
        
        details = {
            'optimization_type': 'algorithm',
            'optimization_level': optimization_level,
            'algorithmic_improvement': improvement
        }
        
        return optimized_time, success, details
    
    def _optimize_generic(self, context: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Generic optimization fallback."""
        baseline = context.get('avg_execution_time', 1.0)
        
        # Apply conservative generic optimization
        optimized_time = baseline * 0.95  # 5% improvement
        success = True
        
        details = {
            'optimization_type': 'generic',
            'improvement_percent': 5.0
        }
        
        return optimized_time, success, details
    
    def record_benchmark(self, benchmark: PerformanceBenchmark):
        """Record a performance benchmark."""
        self.benchmarks.append(benchmark)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if not self.optimization_results:
            return {"status": "no_optimizations_performed"}
        
        recent_results = list(self.optimization_results)[-20:]  # Last 20 optimizations
        
        successful_optimizations = [r for r in recent_results if r.success]
        
        report = {
            'total_optimizations': len(self.optimization_results),
            'recent_optimizations': len(recent_results),
            'success_rate': len(successful_optimizations) / len(recent_results) if recent_results else 0,
            'average_improvement': statistics.mean([r.improvement_percent for r in successful_optimizations]) if successful_optimizations else 0,
            'best_improvement': max([r.improvement_percent for r in successful_optimizations]) if successful_optimizations else 0,
            'optimization_insights': self.adaptive_optimizer.get_optimization_insights(),
            'predictor_accuracy': self.predictor.prediction_accuracy
        }
        
        return report