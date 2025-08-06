"""
Advanced performance monitoring, benchmarking, and analytics system.
"""

import torch
import numpy as np
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import psutil
import logging
import json
import sqlite3
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import tracemalloc
import cProfile
import pstats
import io

# Optional imports
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass 
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage_gb: float
    gpu_usage: float = 0.0
    gpu_memory_gb: float = 0.0
    network_io_mbps: float = 0.0
    disk_io_mbps: float = 0.0
    operations_per_second: float = 0.0
    latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    error_rate: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    name: str
    duration_seconds: float
    operations_per_second: float
    memory_peak_gb: float
    success: bool
    error_message: Optional[str] = None
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Advanced performance profiler with real-time monitoring."""
    
    def __init__(self, sampling_interval: float = 1.0, history_size: int = 10000):
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.active_operations = {}
        self.benchmark_results = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Database for persistent storage
        self.db_path = Path("performance_metrics.db")
        self._init_database()
        
        # Performance tracking
        self.operation_timings = defaultdict(list)
        self.memory_tracker = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Prometheus metrics (if available)
        self.prometheus_metrics = {}
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    cpu_usage REAL,
                    memory_usage_gb REAL,
                    gpu_usage REAL,
                    gpu_memory_gb REAL,
                    operations_per_second REAL,
                    latency_ms REAL,
                    throughput_mbps REAL,
                    error_rate REAL,
                    custom_metrics TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    timestamp REAL,
                    duration_seconds REAL,
                    operations_per_second REAL,
                    memory_peak_gb REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    detailed_metrics TEXT
                )
            """)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.prometheus_metrics = {
            'cpu_usage': prometheus_client.Gauge('photon_cpu_usage_percent', 'CPU usage percentage'),
            'memory_usage': prometheus_client.Gauge('photon_memory_usage_gb', 'Memory usage in GB'),
            'gpu_usage': prometheus_client.Gauge('photon_gpu_usage_percent', 'GPU usage percentage'),
            'operations_per_second': prometheus_client.Gauge('photon_operations_per_second', 'Operations per second'),
            'latency': prometheus_client.Histogram('photon_operation_latency_seconds', 'Operation latency in seconds'),
            'error_rate': prometheus_client.Gauge('photon_error_rate', 'Error rate percentage')
        }
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start memory tracking
        tracemalloc.start()
        
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self._stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Stop memory tracking
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        logging.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.wait(self.sampling_interval):
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self.metrics_history.append(metrics)
                    self._store_metrics_db(metrics)
                    
                    if PROMETHEUS_AVAILABLE:
                        self._update_prometheus_metrics(metrics)
                        
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # CPU and Memory
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage_gb = memory.used / (1024**3)
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io_mbps = (net_io.bytes_sent + net_io.bytes_recv) / (1024**2) / self.sampling_interval
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_mbps = 0.0
        if disk_io:
            disk_io_mbps = (disk_io.read_bytes + disk_io.write_bytes) / (1024**2) / self.sampling_interval
        
        # GPU metrics
        gpu_usage = 0.0
        gpu_memory_gb = 0.0
        
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory_gb = gpu.memoryUsed / 1024
            except Exception:
                pass
        elif torch.cuda.is_available():
            # Fallback to PyTorch CUDA info
            try:
                gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                # GPU usage estimation is not available through PyTorch
            except Exception:
                pass
        
        # Calculate derived metrics
        operations_per_second = self._calculate_ops_per_second()
        latency_ms = self._calculate_average_latency()
        error_rate = self._calculate_error_rate()
        
        return PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage_gb=memory_usage_gb,
            gpu_usage=gpu_usage,
            gpu_memory_gb=gpu_memory_gb,
            network_io_mbps=network_io_mbps,
            disk_io_mbps=disk_io_mbps,
            operations_per_second=operations_per_second,
            latency_ms=latency_ms,
            error_rate=error_rate
        )
    
    def _calculate_ops_per_second(self) -> float:
        """Calculate operations per second from recent timings."""
        if not self.operation_timings:
            return 0.0
        
        recent_ops = 0
        current_time = time.time()
        
        for op_name, timings in self.operation_timings.items():
            # Count operations in the last sampling interval
            recent_ops += sum(1 for t in timings if current_time - t < self.sampling_interval)
        
        return recent_ops / self.sampling_interval
    
    def _calculate_average_latency(self) -> float:
        """Calculate average latency from active operations."""
        if not self.active_operations:
            return 0.0
        
        current_time = time.time()
        latencies = [current_time - start_time for start_time in self.active_operations.values()]
        
        return np.mean(latencies) * 1000  # Convert to milliseconds
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate (placeholder - would need error tracking)."""
        # This would be populated by actual error tracking
        return 0.0
    
    def _store_metrics_db(self, metrics: PerformanceMetrics):
        """Store metrics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (timestamp, cpu_usage, memory_usage_gb, gpu_usage, gpu_memory_gb,
                     operations_per_second, latency_ms, throughput_mbps, error_rate, custom_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.cpu_usage,
                    metrics.memory_usage_gb,
                    metrics.gpu_usage,
                    metrics.gpu_memory_gb,
                    metrics.operations_per_second,
                    metrics.latency_ms,
                    metrics.throughput_mbps,
                    metrics.error_rate,
                    json.dumps(metrics.custom_metrics)
                ))
        except Exception as e:
            logging.error(f"Error storing metrics to database: {e}")
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics."""
        try:
            self.prometheus_metrics['cpu_usage'].set(metrics.cpu_usage)
            self.prometheus_metrics['memory_usage'].set(metrics.memory_usage_gb)
            self.prometheus_metrics['gpu_usage'].set(metrics.gpu_usage)
            self.prometheus_metrics['operations_per_second'].set(metrics.operations_per_second)
            self.prometheus_metrics['error_rate'].set(metrics.error_rate)
        except Exception as e:
            logging.error(f"Error updating Prometheus metrics: {e}")
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.perf_counter()
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        # Track active operation
        with self._lock:
            self.active_operations[operation_id] = start_time
        
        try:
            yield operation_id
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Update timing records
            with self._lock:
                self.operation_timings[operation_name].append(end_time)
                # Keep only recent timings
                cutoff_time = end_time - 60  # Keep last 60 seconds
                self.operation_timings[operation_name] = [
                    t for t in self.operation_timings[operation_name] if t > cutoff_time
                ]
                
                # Remove from active operations
                self.active_operations.pop(operation_id, None)
            
            # Update Prometheus latency histogram
            if PROMETHEUS_AVAILABLE and 'latency' in self.prometheus_metrics:
                self.prometheus_metrics['latency'].observe(duration)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_seconds: Optional[float] = None) -> List[PerformanceMetrics]:
        """Get metrics history for specified duration."""
        if duration_seconds is None:
            with self._lock:
                return list(self.metrics_history)
        
        cutoff_time = time.time() - duration_seconds
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_performance_summary(self, duration_seconds: float = 300) -> Dict[str, Any]:
        """Get performance summary for the specified duration."""
        metrics = self.get_metrics_history(duration_seconds)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        cpu_values = [m.cpu_usage for m in metrics]
        memory_values = [m.memory_usage_gb for m in metrics]
        gpu_values = [m.gpu_usage for m in metrics if m.gpu_usage > 0]
        ops_values = [m.operations_per_second for m in metrics if m.operations_per_second > 0]
        
        summary = {
            "duration_seconds": duration_seconds,
            "sample_count": len(metrics),
            "cpu_usage": {
                "avg": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory_usage_gb": {
                "avg": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std": np.std(memory_values)
            }
        }
        
        if gpu_values:
            summary["gpu_usage"] = {
                "avg": np.mean(gpu_values),
                "max": np.max(gpu_values),
                "min": np.min(gpu_values),
                "std": np.std(gpu_values)
            }
        
        if ops_values:
            summary["operations_per_second"] = {
                "avg": np.mean(ops_values),
                "max": np.max(ops_values),
                "min": np.min(ops_values),
                "std": np.std(ops_values)
            }
        
        return summary


class BenchmarkSuite:
    """Comprehensive benchmarking suite for photonic simulations."""
    
    def __init__(self, profiler: Optional[PerformanceProfiler] = None):
        self.profiler = profiler or PerformanceProfiler()
        self.benchmark_registry = {}
        self.results_history = []
        
        # Register default benchmarks
        self._register_default_benchmarks()
    
    def _register_default_benchmarks(self):
        """Register default benchmark tests."""
        self.register_benchmark("matrix_multiplication", self._benchmark_matrix_ops)
        self.register_benchmark("fft_operations", self._benchmark_fft_ops)
        self.register_benchmark("memory_allocation", self._benchmark_memory_ops)
        self.register_benchmark("gpu_throughput", self._benchmark_gpu_throughput)
        self.register_benchmark("cache_performance", self._benchmark_cache_ops)
    
    def register_benchmark(self, name: str, benchmark_func: Callable):
        """Register a custom benchmark function."""
        self.benchmark_registry[name] = benchmark_func
    
    def run_benchmark(self, name: str, **kwargs) -> BenchmarkResult:
        """Run a specific benchmark."""
        if name not in self.benchmark_registry:
            return BenchmarkResult(
                name=name,
                duration_seconds=0.0,
                operations_per_second=0.0,
                memory_peak_gb=0.0,
                success=False,
                error_message=f"Benchmark '{name}' not found"
            )
        
        benchmark_func = self.benchmark_registry[name]
        
        try:
            # Start profiling
            with self.profiler.profile_operation(f"benchmark_{name}"):
                # Track memory usage
                tracemalloc.start()
                start_memory = self._get_memory_usage()
                start_time = time.perf_counter()
                
                # Run benchmark
                result = benchmark_func(**kwargs)
                
                # Measure completion
                end_time = time.perf_counter()
                end_memory = self._get_memory_usage()
                peak_memory = max(start_memory, end_memory)
                
                # Stop memory tracking for this benchmark
                if tracemalloc.is_tracing():
                    tracemalloc.stop()
                
                duration = end_time - start_time
                
                benchmark_result = BenchmarkResult(
                    name=name,
                    duration_seconds=duration,
                    operations_per_second=result.get('ops_per_second', 0.0),
                    memory_peak_gb=peak_memory,
                    success=True,
                    detailed_metrics=result
                )
                
                self.results_history.append(benchmark_result)
                self._store_benchmark_result(benchmark_result)
                
                return benchmark_result
                
        except Exception as e:
            error_result = BenchmarkResult(
                name=name,
                duration_seconds=0.0,
                operations_per_second=0.0,
                memory_peak_gb=0.0,
                success=False,
                error_message=str(e)
            )
            self.results_history.append(error_result)
            return error_result
    
    def run_all_benchmarks(self, **kwargs) -> List[BenchmarkResult]:
        """Run all registered benchmarks."""
        results = []
        
        for name in self.benchmark_registry:
            result = self.run_benchmark(name, **kwargs)
            results.append(result)
            
            # Brief pause between benchmarks
            time.sleep(1.0)
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.Process().memory_info().rss / (1024**3)
    
    def _benchmark_matrix_ops(self, size: int = 2048, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark matrix operations."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test matrices
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(2):
            _ = torch.mm(A, B)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = torch.mm(A, B)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        ops_per_second = iterations / duration
        flops = 2 * size**3 * iterations  # Approximate FLOPS
        gflops_per_second = flops / (duration * 1e9)
        
        return {
            'ops_per_second': ops_per_second,
            'gflops_per_second': gflops_per_second,
            'matrix_size': size,
            'iterations': iterations,
            'device': str(device)
        }
    
    def _benchmark_fft_ops(self, size: int = 4096, iterations: int = 100) -> Dict[str, Any]:
        """Benchmark FFT operations."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test signal
        signal = torch.randn(size, dtype=torch.complex64, device=device)
        
        # Warmup
        for _ in range(2):
            _ = torch.fft.fft(signal)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            result = torch.fft.fft(signal)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        ops_per_second = iterations / duration
        
        return {
            'ops_per_second': ops_per_second,
            'signal_size': size,
            'iterations': iterations,
            'device': str(device)
        }
    
    def _benchmark_memory_ops(self, allocation_mb: int = 100, iterations: int = 1000) -> Dict[str, Any]:
        """Benchmark memory allocation operations."""
        allocation_size = allocation_mb * 1024 * 1024 // 4  # Float32 elements
        
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            tensor = torch.zeros(allocation_size)
            del tensor
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        ops_per_second = iterations / duration
        mb_per_second = allocation_mb * iterations / duration
        
        return {
            'ops_per_second': ops_per_second,
            'mb_per_second': mb_per_second,
            'allocation_mb': allocation_mb,
            'iterations': iterations
        }
    
    def _benchmark_gpu_throughput(self, data_size_mb: int = 1000) -> Dict[str, Any]:
        """Benchmark GPU memory throughput."""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        device = torch.device('cuda')
        data_size = data_size_mb * 1024 * 1024 // 4  # Float32 elements
        
        # CPU to GPU transfer
        cpu_data = torch.randn(data_size)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        gpu_data = cpu_data.to(device)
        
        torch.cuda.synchronize()
        upload_time = time.perf_counter() - start_time
        
        # GPU to CPU transfer
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        cpu_result = gpu_data.cpu()
        
        torch.cuda.synchronize()
        download_time = time.perf_counter() - start_time
        
        upload_mbps = data_size_mb / upload_time
        download_mbps = data_size_mb / download_time
        
        return {
            'ops_per_second': 2.0 / (upload_time + download_time),  # Round trip
            'upload_mbps': upload_mbps,
            'download_mbps': download_mbps,
            'data_size_mb': data_size_mb
        }
    
    def _benchmark_cache_ops(self, cache_size: int = 10000, iterations: int = 100000) -> Dict[str, Any]:
        """Benchmark cache operations."""
        from .cache import AdaptiveCache
        
        cache = AdaptiveCache(max_size=cache_size)
        
        # Fill cache
        for i in range(cache_size // 2):
            cache.put(f"key_{i}", f"value_{i}")
        
        start_time = time.perf_counter()
        
        # Mixed read/write operations
        for i in range(iterations):
            if i % 4 == 0:  # 25% writes
                cache.put(f"key_{i}", f"value_{i}")
            else:  # 75% reads
                cache.get(f"key_{i % (cache_size // 2)}")
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        ops_per_second = iterations / duration
        
        stats = cache.get_stats()
        
        return {
            'ops_per_second': ops_per_second,
            'cache_hit_rate': stats.get('hit_rate', 0.0),
            'cache_utilization': stats.get('utilization', 0.0),
            'iterations': iterations,
            'cache_size': cache_size
        }
    
    def _store_benchmark_result(self, result: BenchmarkResult):
        """Store benchmark result in database."""
        if not hasattr(self.profiler, 'db_path'):
            return
        
        try:
            with sqlite3.connect(self.profiler.db_path) as conn:
                conn.execute("""
                    INSERT INTO benchmark_results 
                    (name, timestamp, duration_seconds, operations_per_second, 
                     memory_peak_gb, success, error_message, detailed_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.name,
                    time.time(),
                    result.duration_seconds,
                    result.operations_per_second,
                    result.memory_peak_gb,
                    result.success,
                    result.error_message,
                    json.dumps(result.detailed_metrics)
                ))
        except Exception as e:
            logging.error(f"Error storing benchmark result: {e}")
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive benchmark report."""
        if not self.results_history:
            return "No benchmark results available."
        
        # Create report content
        report_lines = [
            "# Photon Neuromorphics SDK - Performance Benchmark Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total benchmarks run: {len(self.results_history)}",
            ""
        ]
        
        # Summary statistics
        successful_results = [r for r in self.results_history if r.success]
        failed_results = [r for r in self.results_history if not r.success]
        
        report_lines.extend([
            "## Summary",
            f"Successful benchmarks: {len(successful_results)}",
            f"Failed benchmarks: {len(failed_results)}",
            ""
        ])
        
        # Individual benchmark results
        report_lines.append("## Benchmark Results")
        
        for result in self.results_history:
            report_lines.extend([
                f"### {result.name}",
                f"- Duration: {result.duration_seconds:.4f} seconds",
                f"- Operations/sec: {result.operations_per_second:.2f}",
                f"- Peak Memory: {result.memory_peak_gb:.4f} GB",
                f"- Success: {'✓' if result.success else '✗'}",
            ])
            
            if result.error_message:
                report_lines.append(f"- Error: {result.error_message}")
            
            if result.detailed_metrics:
                report_lines.append("- Detailed metrics:")
                for key, value in result.detailed_metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"  - {key}: {value:.4f}")
                    else:
                        report_lines.append(f"  - {key}: {value}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
        
        return report_content


class PredictiveScaler:
    """Predictive scaling system based on workload patterns."""
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.workload_history = deque(maxlen=history_window)
        self.scaling_decisions = []
        self.prediction_model = None
        
    def record_workload(self, metrics: PerformanceMetrics):
        """Record workload metrics for pattern analysis."""
        self.workload_history.append(metrics)
        
        # Trigger prediction if we have enough data
        if len(self.workload_history) >= 100:  # Minimum data points
            self._update_predictions()
    
    def _update_predictions(self):
        """Update workload predictions based on historical data."""
        if len(self.workload_history) < 100:
            return
        
        # Extract features for prediction
        features = []
        targets = []
        
        for i in range(10, len(self.workload_history) - 10):
            # Use 10 previous points to predict next 10 points
            prev_metrics = list(self.workload_history)[i-10:i]
            next_metrics = list(self.workload_history)[i:i+10]
            
            # Simple feature extraction: average of previous metrics
            feature_vector = [
                np.mean([m.cpu_usage for m in prev_metrics]),
                np.mean([m.memory_usage_gb for m in prev_metrics]),
                np.mean([m.operations_per_second for m in prev_metrics])
            ]
            
            # Target: average of next metrics
            target_vector = [
                np.mean([m.cpu_usage for m in next_metrics]),
                np.mean([m.memory_usage_gb for m in next_metrics]),
                np.mean([m.operations_per_second for m in next_metrics])
            ]
            
            features.append(feature_vector)
            targets.append(target_vector)
        
        # Simple linear prediction (in a real system, would use ML models)
        if len(features) > 50:
            self.prediction_model = self._fit_simple_model(features, targets)
    
    def _fit_simple_model(self, features: List[List[float]], targets: List[List[float]]) -> Dict[str, Any]:
        """Fit a simple prediction model."""
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        # Simple linear regression (would use scikit-learn in practice)
        # For now, just compute mean trends
        feature_means = np.mean(X, axis=0)
        target_means = np.mean(y, axis=0)
        
        return {
            'feature_means': feature_means,
            'target_means': target_means,
            'trend': target_means - feature_means
        }
    
    def predict_workload(self, horizon_minutes: int = 10) -> Optional[Dict[str, float]]:
        """Predict workload for the next horizon_minutes."""
        if not self.prediction_model or len(self.workload_history) < 10:
            return None
        
        # Get recent metrics for prediction
        recent_metrics = list(self.workload_history)[-10:]
        
        current_features = [
            np.mean([m.cpu_usage for m in recent_metrics]),
            np.mean([m.memory_usage_gb for m in recent_metrics]),
            np.mean([m.operations_per_second for m in recent_metrics])
        ]
        
        # Simple prediction based on trends
        trend = self.prediction_model['trend']
        prediction = np.array(current_features) + trend * (horizon_minutes / 10)
        
        return {
            'predicted_cpu_usage': prediction[0],
            'predicted_memory_usage_gb': prediction[1], 
            'predicted_operations_per_second': prediction[2],
            'confidence': min(0.8, len(self.workload_history) / self.history_window)
        }
    
    def recommend_scaling_action(self, current_resources: Dict[str, float]) -> Dict[str, Any]:
        """Recommend scaling actions based on predictions."""
        prediction = self.predict_workload()
        
        if not prediction:
            return {'action': 'no_change', 'reason': 'Insufficient data for prediction'}
        
        recommendations = []
        
        # CPU scaling recommendations
        if prediction['predicted_cpu_usage'] > 80:
            recommendations.append({
                'resource': 'cpu',
                'action': 'scale_up',
                'current': current_resources.get('cpu_cores', 1),
                'recommended': int(current_resources.get('cpu_cores', 1) * 1.5),
                'reason': f"Predicted CPU usage: {prediction['predicted_cpu_usage']:.1f}%"
            })
        elif prediction['predicted_cpu_usage'] < 30 and current_resources.get('cpu_cores', 1) > 1:
            recommendations.append({
                'resource': 'cpu',
                'action': 'scale_down',
                'current': current_resources.get('cpu_cores', 1),
                'recommended': max(1, int(current_resources.get('cpu_cores', 1) * 0.8)),
                'reason': f"Predicted CPU usage: {prediction['predicted_cpu_usage']:.1f}%"
            })
        
        # Memory scaling recommendations
        memory_threshold_gb = current_resources.get('memory_gb', 8) * 0.8
        if prediction['predicted_memory_usage_gb'] > memory_threshold_gb:
            recommendations.append({
                'resource': 'memory',
                'action': 'scale_up',
                'current': current_resources.get('memory_gb', 8),
                'recommended': current_resources.get('memory_gb', 8) * 1.5,
                'reason': f"Predicted memory usage: {prediction['predicted_memory_usage_gb']:.1f} GB"
            })
        
        return {
            'predictions': prediction,
            'recommendations': recommendations,
            'confidence': prediction['confidence']
        }


# Global profiler instance
_global_profiler = None

def get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def start_global_monitoring():
    """Start global performance monitoring."""
    profiler = get_global_profiler()
    profiler.start_monitoring()

def stop_global_monitoring():
    """Stop global performance monitoring."""
    profiler = get_global_profiler()
    profiler.stop_monitoring()

@contextmanager
def profile_operation(operation_name: str):
    """Global operation profiling context manager."""
    profiler = get_global_profiler()
    with profiler.profile_operation(operation_name):
        yield