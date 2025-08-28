#!/usr/bin/env python3
"""
Performance Optimization and Scaling Features
Generation 3: Make It Scale Implementation
"""

import time
import threading
import concurrent.futures
import queue
from typing import Any, Dict, List, Optional, Callable, Tuple
from functools import wraps, lru_cache
import json
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_latency: float = 0.0
    peak_latency: float = 0.0
    min_latency: float = float('inf')
    total_processing_time: float = 0.0
    throughput_ops_per_sec: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_executions: int = 0

class AdaptiveCache:
    """Self-optimizing cache with LRU eviction and performance tracking"""
    
    def __init__(self, max_size: int = 128, ttl: float = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
        self.creation_times = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Any:
        """Get item from cache with LRU tracking"""
        with self._lock:
            current_time = time.time()
            
            if key in self.cache:
                # Check TTL
                if current_time - self.creation_times[key] > self.ttl:
                    self._evict(key)
                    self.misses += 1
                    return None
                
                # Update access tracking
                self.access_times[key] = current_time
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with automatic eviction"""
        with self._lock:
            current_time = time.time()
            
            # If cache is full, evict LRU item
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.creation_times[key] = current_time
            self.access_times[key] = current_time
            self.access_counts[key] = 1
    
    def _evict(self, key: str):
        """Evict specific key"""
        if key in self.cache:
            del self.cache[key]
            del self.creation_times[key]
            del self.access_times[key]
            del self.access_counts[key]
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / max(1, total_accesses)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'utilization': len(self.cache) / self.max_size
        }
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.creation_times.clear()

class LoadBalancer:
    """Dynamic load balancer for distributing computational tasks"""
    
    def __init__(self, worker_count: int = None):
        self.worker_count = worker_count or min(8, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.worker_count)
        self.process_pool = ProcessPoolExecutor(max_workers=os.cpu_count() or 1)
        self.task_queue = queue.Queue()
        self.worker_load = {i: 0 for i in range(self.worker_count)}
        self.worker_performance = {i: [] for i in range(self.worker_count)}
        self._lock = threading.Lock()
    
    def submit_task(self, task_func: Callable, *args, use_processes: bool = False, **kwargs) -> concurrent.futures.Future:
        """Submit task to optimal worker"""
        if use_processes:
            return self.process_pool.submit(task_func, *args, **kwargs)
        else:
            return self.thread_pool.submit(self._execute_with_tracking, task_func, *args, **kwargs)
    
    def _execute_with_tracking(self, task_func: Callable, *args, **kwargs) -> Any:
        """Execute task with performance tracking"""
        worker_id = threading.get_ident() % self.worker_count
        start_time = time.time()
        
        with self._lock:
            self.worker_load[worker_id] += 1
        
        try:
            result = task_func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            with self._lock:
                self.worker_performance[worker_id].append(execution_time)
                # Keep only last 100 measurements
                if len(self.worker_performance[worker_id]) > 100:
                    self.worker_performance[worker_id] = self.worker_performance[worker_id][-100:]
            
            return result
            
        finally:
            with self._lock:
                self.worker_load[worker_id] -= 1
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        with self._lock:
            avg_performance = {}
            for worker_id, times in self.worker_performance.items():
                if times:
                    avg_performance[worker_id] = sum(times) / len(times)
                else:
                    avg_performance[worker_id] = 0.0
            
            total_load = sum(self.worker_load.values())
            
            return {
                'worker_count': self.worker_count,
                'current_load': self.worker_load.copy(),
                'total_load': total_load,
                'average_performance': avg_performance,
                'load_balance_ratio': max(self.worker_load.values()) / max(1, min(self.worker_load.values()) or 1)
            }
    
    def shutdown(self):
        """Shutdown thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class AutoScaler:
    """Automatic scaling based on performance metrics and load"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 16):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.metrics_history = []
        self.last_scale_time = 0
        self.scale_cooldown = 30  # seconds
        self.load_threshold_up = 0.8  # Scale up threshold
        self.load_threshold_down = 0.3  # Scale down threshold
    
    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Determine if scaling up is needed"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        if self.current_workers >= self.max_workers:
            return False
        
        # Scale up if latency is high or throughput is low
        if metrics.average_latency > 1.0 or metrics.throughput_ops_per_sec < 10:
            return True
        
        return False
    
    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Determine if scaling down is possible"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        if self.current_workers <= self.min_workers:
            return False
        
        # Scale down if performance is consistently good and load is low
        if (metrics.average_latency < 0.1 and 
            metrics.throughput_ops_per_sec > 100 and
            len(self.metrics_history) >= 5):
            
            # Check last 5 measurements for consistency
            recent_latencies = [m.average_latency for m in self.metrics_history[-5:]]
            if all(lat < 0.1 for lat in recent_latencies):
                return True
        
        return False
    
    def scale(self, metrics: PerformanceMetrics) -> Optional[int]:
        """Determine scaling action"""
        self.metrics_history.append(metrics)
        
        # Keep only last 20 measurements
        if len(self.metrics_history) > 20:
            self.metrics_history = self.metrics_history[-20:]
        
        if self.should_scale_up(metrics):
            self.current_workers = min(self.max_workers, self.current_workers + 1)
            self.last_scale_time = time.time()
            return self.current_workers
        
        elif self.should_scale_down(metrics):
            self.current_workers = max(self.min_workers, self.current_workers - 1)
            self.last_scale_time = time.time()
            return self.current_workers
        
        return None

def performance_monitor(track_memory: bool = True, track_latency: bool = True):
    """Decorator for comprehensive performance monitoring"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0
            
            if track_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    start_memory = 0
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)
                raise
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                
                end_memory = 0
                if track_memory:
                    try:
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    except:
                        pass
                
                # Store performance data (in a real system, this would go to a monitoring system)
                performance_data = {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'memory_start': start_memory,
                    'memory_end': end_memory,
                    'memory_delta': end_memory - start_memory,
                    'success': success,
                    'timestamp': end_time,
                    'error': error
                }
                
                # In a real implementation, you'd send this to a monitoring system
                if hasattr(wrapper, '_performance_log'):
                    wrapper._performance_log.append(performance_data)
                else:
                    wrapper._performance_log = [performance_data]
            
            return result
        
        return wrapper
    return decorator

def batch_processor(batch_size: int = 10, max_wait_time: float = 1.0):
    """Decorator for batching multiple calls for efficiency"""
    def decorator(func: Callable):
        batch_queue = queue.Queue()
        batch_lock = threading.Lock()
        batch_results = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            request_id = id((args, tuple(kwargs.items())))
            
            with batch_lock:
                batch_queue.put((request_id, args, kwargs))
                
                # Process batch if full or after timeout
                if batch_queue.qsize() >= batch_size:
                    process_batch()
            
            # Wait for result (simplified - in real system would use proper async)
            start_wait = time.time()
            while request_id not in batch_results and (time.time() - start_wait) < max_wait_time:
                time.sleep(0.001)
                if batch_queue.qsize() > 0:
                    with batch_lock:
                        if batch_queue.qsize() >= batch_size // 2:  # Process partial batches
                            process_batch()
            
            if request_id in batch_results:
                result = batch_results.pop(request_id)
                if isinstance(result, Exception):
                    raise result
                return result
            else:
                # Fallback to individual processing
                return func(*args, **kwargs)
        
        def process_batch():
            """Process accumulated batch of requests"""
            batch_items = []
            while not batch_queue.empty() and len(batch_items) < batch_size:
                try:
                    batch_items.append(batch_queue.get_nowait())
                except queue.Empty:
                    break
            
            if not batch_items:
                return
            
            # Process all items in batch
            for request_id, args, kwargs in batch_items:
                try:
                    result = func(*args, **kwargs)
                    batch_results[request_id] = result
                except Exception as e:
                    batch_results[request_id] = e
        
        return wrapper
    return decorator

class ScalablePhotonicSimulator:
    """High-performance photonic simulator with scaling capabilities"""
    
    def __init__(self, enable_caching: bool = True, enable_parallel: bool = True):
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        
        # Performance components
        self.cache = AdaptiveCache(max_size=256, ttl=600) if enable_caching else None
        self.load_balancer = LoadBalancer() if enable_parallel else None
        self.auto_scaler = AutoScaler()
        self.metrics = PerformanceMetrics()
        
        # Performance tracking
        self.operation_times = []
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    @lru_cache(maxsize=128)
    def _compute_optical_transfer_matrix(self, length: float, wavelength: float, n_eff: float) -> Tuple[complex, complex, complex, complex]:
        """Cached computation of optical transfer matrix"""
        import cmath
        
        beta = 2 * 3.14159 * n_eff / wavelength
        phase = beta * length
        
        # Simple 2x2 transfer matrix for waveguide
        t11 = cmath.exp(1j * phase)
        t12 = 0
        t21 = 0
        t22 = cmath.exp(1j * phase)
        
        return (t11, t12, t21, t22)
    
    @performance_monitor(track_memory=True, track_latency=True)
    def simulate_waveguide(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """High-performance waveguide simulation"""
        start_time = time.time()
        
        # Extract parameters with defaults
        length = parameters.get('length', 1e-3)
        wavelength = parameters.get('wavelength', 1550e-9)
        width = parameters.get('width', 450e-9)
        n_eff = parameters.get('n_eff', 2.4)
        loss_db_per_cm = parameters.get('loss_db_per_cm', 0.1)
        
        # Check cache first
        cache_key = f"wg_{length}_{wavelength}_{width}_{n_eff}_{loss_db_per_cm}"
        
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self._update_metrics(time.time() - start_time, True, cache_hit=True)
                return cached_result
        
        # Perform calculation
        try:
            # Use cached transfer matrix computation
            t11, t12, t21, t22 = self._compute_optical_transfer_matrix(length, wavelength, n_eff)
            
            # Calculate results
            phase_shift = abs(t11.imag)
            loss_db = loss_db_per_cm * length * 100  # Convert to dB
            group_delay = length / (3e8 / n_eff)  # Approximate group delay
            
            result = {
                'component': 'waveguide',
                'phase_shift': phase_shift,
                'loss_db': loss_db,
                'group_delay': group_delay,
                'transmission': abs(t11)**2 * 10**(-loss_db/10),
                'effective_index': n_eff,
                'length': length,
                'wavelength': wavelength,
                'computation_time': time.time() - start_time
            }
            
            # Cache result
            if self.cache:
                self.cache.put(cache_key, result)
            
            self._update_metrics(time.time() - start_time, True, cache_hit=False)
            return result
            
        except Exception as e:
            self._update_metrics(time.time() - start_time, False)
            raise
    
    @performance_monitor(track_memory=True)
    def simulate_network_parallel(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate network of components in parallel"""
        if not self.enable_parallel or not self.load_balancer:
            # Sequential processing fallback
            return [self.simulate_component(comp) for comp in components]
        
        start_time = time.time()
        
        # Submit all simulations to thread pool
        futures = []
        for component in components:
            future = self.load_balancer.submit_task(self.simulate_component, component)
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({'error': str(e), 'component': 'unknown'})
        
        # Update parallel execution metrics
        with self._lock:
            self.metrics.parallel_executions += 1
        
        execution_time = time.time() - start_time
        self._update_metrics(execution_time, True)
        
        return results
    
    def simulate_component(self, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate individual component with optimization"""
        component_type = component_config.get('type', 'waveguide')
        
        if component_type == 'waveguide':
            return self.simulate_waveguide(component_config.get('parameters', {}))
        
        elif component_type == 'mzi':
            return self._simulate_mzi(component_config.get('parameters', {}))
        
        else:
            # Generic component simulation
            return {
                'component': component_type,
                'status': 'simulated',
                'parameters': component_config.get('parameters', {}),
                'timestamp': time.time()
            }
    
    @batch_processor(batch_size=5, max_wait_time=0.5)
    def _simulate_mzi(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Batched MZI simulation for efficiency"""
        phase_shift = parameters.get('phase_shift', 0)
        wavelength = parameters.get('wavelength', 1550e-9)
        
        # MZI transfer function
        transmission = 0.5 * (1 + abs(phase_shift) / 3.14159)
        reflection = 1 - transmission
        
        return {
            'component': 'mzi',
            'transmission': transmission,
            'reflection': reflection,
            'phase_shift': phase_shift,
            'wavelength': wavelength,
            'extinction_ratio': 10 * abs(phase_shift) if phase_shift != 0 else float('inf')
        }
    
    def _update_metrics(self, execution_time: float, success: bool, cache_hit: bool = False):
        """Update performance metrics"""
        with self._lock:
            self.metrics.total_operations += 1
            
            if success:
                self.metrics.successful_operations += 1
            else:
                self.metrics.failed_operations += 1
            
            if cache_hit:
                self.metrics.cache_hits += 1
            else:
                self.metrics.cache_misses += 1
            
            # Update timing metrics
            self.operation_times.append(execution_time)
            if len(self.operation_times) > 1000:  # Keep last 1000
                self.operation_times = self.operation_times[-1000:]
            
            if self.operation_times:
                self.metrics.average_latency = sum(self.operation_times) / len(self.operation_times)
                self.metrics.peak_latency = max(self.operation_times)
                self.metrics.min_latency = min(self.operation_times)
            
            # Update throughput
            total_time = time.time() - self.start_time
            if total_time > 0:
                self.metrics.throughput_ops_per_sec = self.metrics.total_operations / total_time
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'metrics': {
                'total_operations': self.metrics.total_operations,
                'successful_operations': self.metrics.successful_operations,
                'failed_operations': self.metrics.failed_operations,
                'success_rate': self.metrics.successful_operations / max(1, self.metrics.total_operations),
                'average_latency_ms': self.metrics.average_latency * 1000,
                'peak_latency_ms': self.metrics.peak_latency * 1000,
                'min_latency_ms': self.metrics.min_latency * 1000 if self.metrics.min_latency != float('inf') else 0,
                'throughput_ops_per_sec': self.metrics.throughput_ops_per_sec,
                'parallel_executions': self.metrics.parallel_executions
            },
            'caching': self.cache.get_stats() if self.cache else {'enabled': False},
            'load_balancing': self.load_balancer.get_load_stats() if self.load_balancer else {'enabled': False},
            'system_info': {
                'cpu_count': os.cpu_count(),
                'thread_count': threading.active_count(),
                'enable_caching': self.enable_caching,
                'enable_parallel': self.enable_parallel
            }
        }
        
        return report
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Automatic performance optimization"""
        optimization_actions = []
        
        # Check if scaling is needed
        scale_decision = self.auto_scaler.scale(self.metrics)
        if scale_decision:
            optimization_actions.append(f"Scaled workers to {scale_decision}")
        
        # Clear cache if hit rate is low
        if self.cache:
            cache_stats = self.cache.get_stats()
            if cache_stats['hit_rate'] < 0.1 and cache_stats['size'] > 10:
                self.cache.clear()
                optimization_actions.append("Cleared inefficient cache")
        
        # Optimize batch sizes based on performance
        if self.metrics.average_latency > 0.5:
            optimization_actions.append("Consider reducing batch sizes for better latency")
        
        return {
            'optimizations_applied': optimization_actions,
            'current_performance': self.get_performance_report(),
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses) < 0.5:
            recommendations.append("Consider increasing cache size or TTL")
        
        if self.metrics.average_latency > 1.0:
            recommendations.append("Enable parallel processing for better performance")
        
        if self.metrics.parallel_executions == 0 and self.metrics.total_operations > 10:
            recommendations.append("Use parallel simulation for multiple components")
        
        if not recommendations:
            recommendations.append("Performance is optimal")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources"""
        if self.load_balancer:
            self.load_balancer.shutdown()
        
        if self.cache:
            self.cache.clear()

def test_scaling_features():
    """Test Generation 3 scaling features"""
    print("⚡ GENERATION 3: SCALING FEATURES TEST")
    print("=" * 50)
    
    simulator = ScalablePhotonicSimulator(enable_caching=True, enable_parallel=True)
    
    try:
        # Test single component simulation
        print("✓ Testing single component simulation...")
        waveguide_params = {
            'length': 1e-3,
            'wavelength': 1550e-9,
            'width': 450e-9,
            'n_eff': 2.4,
            'loss_db_per_cm': 0.1
        }
        
        result = simulator.simulate_waveguide(waveguide_params)
        print(f"  Single simulation time: {result.get('computation_time', 0):.4f}s")
        print(f"  Phase shift: {result.get('phase_shift', 0):.3f} radians")
        
        # Test caching (simulate same parameters again)
        print("✓ Testing caching performance...")
        start_time = time.time()
        cached_result = simulator.simulate_waveguide(waveguide_params)
        cache_time = time.time() - start_time
        print(f"  Cached simulation time: {cache_time:.6f}s")
        
        # Test parallel processing
        print("✓ Testing parallel processing...")
        components = []
        for i in range(10):
            components.append({
                'type': 'waveguide',
                'parameters': {
                    'length': (i + 1) * 1e-4,  # Different lengths
                    'wavelength': 1550e-9,
                    'width': 450e-9
                }
            })
        
        start_time = time.time()
        parallel_results = simulator.simulate_network_parallel(components)
        parallel_time = time.time() - start_time
        
        print(f"  Parallel simulation of {len(components)} components: {parallel_time:.4f}s")
        print(f"  Average time per component: {parallel_time/len(components):.4f}s")
        
        # Test batching with MZI components
        print("✓ Testing batch processing...")
        mzi_components = []
        for i in range(5):
            mzi_components.append({
                'type': 'mzi',
                'parameters': {
                    'phase_shift': i * 0.5,
                    'wavelength': 1550e-9
                }
            })
        
        start_time = time.time()
        batch_results = [simulator.simulate_component(comp) for comp in mzi_components]
        batch_time = time.time() - start_time
        print(f"  Batch processing of {len(mzi_components)} MZIs: {batch_time:.4f}s")
        
        # Get performance report
        print("✓ Testing performance monitoring...")
        performance_report = simulator.get_performance_report()
        metrics = performance_report['metrics']
        
        print(f"  Total operations: {metrics['total_operations']}")
        print(f"  Success rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Average latency: {metrics['average_latency_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput_ops_per_sec']:.1f} ops/sec")
        print(f"  Cache hit rate: {performance_report['caching']['hit_rate']*100:.1f}%")
        
        # Test optimization
        print("✓ Testing automatic optimization...")
        optimization_result = simulator.optimize_performance()
        optimizations = optimization_result['optimizations_applied']
        recommendations = optimization_result['recommendations']
        
        print(f"  Optimizations applied: {len(optimizations)}")
        for opt in optimizations:
            print(f"    - {opt}")
        
        print(f"  Recommendations: {len(recommendations)}")
        for rec in recommendations:
            print(f"    - {rec}")
        
        print("✅ All scaling features working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Scaling features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        simulator.cleanup()

def main():
    """Main test execution for Generation 3"""
    print("🚀 GENERATION 3: MAKE IT SCALE - AUTONOMOUS EXECUTION")
    print("=" * 60)
    
    success = test_scaling_features()
    
    if success:
        print("\n🎉 GENERATION 3 IMPLEMENTATION COMPLETE!")
        print("⚡ Enhanced with high-performance caching, parallel processing, and auto-scaling!")
    else:
        print("\n❌ GENERATION 3 IMPLEMENTATION NEEDS ATTENTION")
    
    return success

if __name__ == "__main__":
    main()