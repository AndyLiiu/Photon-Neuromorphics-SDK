"""
Comprehensive logging and monitoring system for Photon Neuromorphics SDK.

This module provides:
- Structured logging with multiple handlers
- Performance monitoring and metrics collection
- Debugging tools and diagnostic modes
- Simulation progress tracking
- Real-time monitoring dashboard
- Log analysis and reporting
"""

import logging
import logging.handlers
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import functools
import traceback
import sys
import psutil
import torch
import numpy as np

from ..core.exceptions import ValidationError, validate_parameter


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    component: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ProgressUpdate:
    """Simulation progress update."""
    timestamp: datetime
    task_name: str
    progress_percent: float
    stage: str
    eta_seconds: Optional[float] = None
    throughput: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def __init__(self, include_metrics: bool = True):
        super().__init__()
        self.include_metrics = include_metrics
    
    def format(self, record):
        """Format log record with structured data."""
        try:
            # Basic log data
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add thread info
            log_data['thread'] = record.thread
            log_data['thread_name'] = record.threadName
            
            # Add process info
            log_data['process'] = record.process
            
            # Add exception info if present
            if record.exc_info:
                log_data['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': traceback.format_exception(*record.exc_info)
                }
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    log_data['extra'][key] = value
            
            # Add system metrics if enabled
            if self.include_metrics and record.levelno >= logging.INFO:
                try:
                    log_data['system_metrics'] = {
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_percent': psutil.virtual_memory().percent,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # GPU metrics if available
                    if torch.cuda.is_available():
                        log_data['gpu_metrics'] = {
                            'gpu_memory_allocated': torch.cuda.memory_allocated(),
                            'gpu_memory_reserved': torch.cuda.memory_reserved(),
                            'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
                        }
                except Exception:
                    pass  # Don't fail logging due to metrics collection
            
            return json.dumps(log_data, default=str, separators=(',', ':'))
            
        except Exception as e:
            # Fallback to simple format if structured formatting fails
            return f"{datetime.now().isoformat()} - {record.levelname} - {record.name} - {record.getMessage()}"


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.Lock()
        self.enabled = True
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     component: str = None, **metadata):
        """Record a performance metric."""
        if not self.enabled:
            return
        
        try:
            with self._lock:
                metric = PerformanceMetric(
                    timestamp=datetime.now(),
                    metric_name=name,
                    value=float(value),
                    unit=unit,
                    component=component,
                    metadata=metadata
                )
                self.metrics[name].append(metric)
                
        except Exception as e:
            # Don't fail operations due to metrics collection
            print(f"Warning: Failed to record metric {name}: {e}")
    
    def get_metric_history(self, name: str, hours: float = 1.0) -> List[PerformanceMetric]:
        """Get metric history for specified time window."""
        try:
            with self._lock:
                if name not in self.metrics:
                    return []
                
                cutoff_time = datetime.now() - timedelta(hours=hours)
                recent_metrics = [
                    metric for metric in self.metrics[name]
                    if metric.timestamp > cutoff_time
                ]
                
                return list(recent_metrics)
                
        except Exception:
            return []
    
    def get_metric_stats(self, name: str, hours: float = 1.0) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        try:
            history = self.get_metric_history(name, hours)
            if not history:
                return {}
            
            values = [metric.value for metric in history]
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1] if values else 0.0
            }
            
        except Exception:
            return {}
    
    def clear_metrics(self, name: str = None):
        """Clear metrics history."""
        with self._lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
            else:
                self.metrics.clear()


class ProgressTracker:
    """Tracks simulation and operation progress."""
    
    def __init__(self):
        self.active_tasks = {}
        self._lock = threading.Lock()
        self.callbacks = []
    
    def start_task(self, task_name: str, total_steps: int = 100):
        """Start tracking a new task."""
        with self._lock:
            self.active_tasks[task_name] = {
                'total_steps': total_steps,
                'current_step': 0,
                'start_time': datetime.now(),
                'last_update': datetime.now(),
                'stage': 'initializing',
                'throughput_history': deque(maxlen=10)
            }
    
    def update_progress(self, task_name: str, step: int = None, 
                       stage: str = None, additional_info: Dict[str, Any] = None):
        """Update task progress."""
        try:
            with self._lock:
                if task_name not in self.active_tasks:
                    return
                
                task = self.active_tasks[task_name]
                
                if step is not None:
                    task['current_step'] = step
                
                if stage is not None:
                    task['stage'] = stage
                
                current_time = datetime.now()
                
                # Calculate throughput
                time_delta = (current_time - task['last_update']).total_seconds()
                if time_delta > 0:
                    throughput = 1.0 / time_delta  # steps per second
                    task['throughput_history'].append(throughput)
                
                # Calculate ETA
                if task['current_step'] > 0:
                    elapsed = (current_time - task['start_time']).total_seconds()
                    progress_ratio = task['current_step'] / task['total_steps']
                    eta = (elapsed / progress_ratio) - elapsed if progress_ratio > 0 else None
                else:
                    eta = None
                
                # Create progress update
                progress = ProgressUpdate(
                    timestamp=current_time,
                    task_name=task_name,
                    progress_percent=100.0 * task['current_step'] / task['total_steps'],
                    stage=task['stage'],
                    eta_seconds=eta,
                    throughput=np.mean(task['throughput_history']) if task['throughput_history'] else None,
                    memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                    gpu_utilization=self._get_gpu_utilization()
                )
                
                task['last_update'] = current_time
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(progress)
                    except Exception as e:
                        print(f"Progress callback failed: {e}")
                        
        except Exception as e:
            print(f"Progress update failed: {e}")
    
    def finish_task(self, task_name: str):
        """Mark task as finished."""
        with self._lock:
            if task_name in self.active_tasks:
                self.update_progress(task_name, 
                                   step=self.active_tasks[task_name]['total_steps'],
                                   stage='completed')
                del self.active_tasks[task_name]
    
    def add_progress_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Add callback for progress updates."""
        self.callbacks.append(callback)
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get GPU utilization if available."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.utilization()
        except Exception:
            pass
        return None
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active tasks."""
        with self._lock:
            return {name: {
                'progress_percent': 100.0 * task['current_step'] / task['total_steps'],
                'stage': task['stage'],
                'elapsed_time': (datetime.now() - task['start_time']).total_seconds(),
                'throughput': np.mean(task['throughput_history']) if task['throughput_history'] else 0.0
            } for name, task in self.active_tasks.items()}


class DiagnosticMode:
    """Enhanced diagnostic and debugging modes."""
    
    def __init__(self):
        self.enabled = False
        self.diagnostic_data = {}
        self.capture_functions = []
        self.debug_breakpoints = set()
    
    def enable(self, capture_tensors: bool = True, capture_gradients: bool = True,
               capture_memory: bool = True):
        """Enable diagnostic mode with specified capture options."""
        self.enabled = True
        self.diagnostic_data.clear()
        
        if capture_tensors:
            self.capture_functions.append(self._capture_tensor_stats)
        if capture_gradients:
            self.capture_functions.append(self._capture_gradient_stats)
        if capture_memory:
            self.capture_functions.append(self._capture_memory_stats)
    
    def disable(self):
        """Disable diagnostic mode."""
        self.enabled = False
        self.capture_functions.clear()
    
    def capture_state(self, name: str, **kwargs):
        """Capture current state for diagnostics."""
        if not self.enabled:
            return
        
        try:
            timestamp = datetime.now()
            state_data = {'timestamp': timestamp, 'name': name}
            
            for capture_func in self.capture_functions:
                try:
                    captured = capture_func(**kwargs)
                    state_data.update(captured)
                except Exception as e:
                    state_data[f'capture_error_{capture_func.__name__}'] = str(e)
            
            self.diagnostic_data[f"{timestamp.isoformat()}_{name}"] = state_data
            
        except Exception as e:
            print(f"Diagnostic capture failed: {e}")
    
    def _capture_tensor_stats(self, **kwargs) -> Dict[str, Any]:
        """Capture tensor statistics."""
        tensor_stats = {}
        
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                try:
                    tensor_stats[f'{name}_shape'] = list(value.shape)
                    tensor_stats[f'{name}_dtype'] = str(value.dtype)
                    tensor_stats[f'{name}_device'] = str(value.device)
                    tensor_stats[f'{name}_mean'] = float(value.mean()) if value.numel() > 0 else 0.0
                    tensor_stats[f'{name}_std'] = float(value.std()) if value.numel() > 1 else 0.0
                    tensor_stats[f'{name}_min'] = float(value.min()) if value.numel() > 0 else 0.0
                    tensor_stats[f'{name}_max'] = float(value.max()) if value.numel() > 0 else 0.0
                    tensor_stats[f'{name}_has_nan'] = bool(torch.isnan(value).any())
                    tensor_stats[f'{name}_has_inf'] = bool(torch.isinf(value).any())
                except Exception as e:
                    tensor_stats[f'{name}_error'] = str(e)
        
        return tensor_stats
    
    def _capture_gradient_stats(self, **kwargs) -> Dict[str, Any]:
        """Capture gradient statistics."""
        grad_stats = {}
        
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.grad is not None:
                try:
                    grad = value.grad
                    grad_stats[f'{name}_grad_norm'] = float(torch.norm(grad))
                    grad_stats[f'{name}_grad_mean'] = float(grad.mean())
                    grad_stats[f'{name}_grad_std'] = float(grad.std())
                    grad_stats[f'{name}_grad_has_nan'] = bool(torch.isnan(grad).any())
                    grad_stats[f'{name}_grad_has_inf'] = bool(torch.isinf(grad).any())
                except Exception as e:
                    grad_stats[f'{name}_grad_error'] = str(e)
        
        return grad_stats
    
    def _capture_memory_stats(self, **kwargs) -> Dict[str, Any]:
        """Capture memory usage statistics."""
        memory_stats = {}
        
        try:
            # System memory
            memory = psutil.virtual_memory()
            memory_stats['system_memory_percent'] = memory.percent
            memory_stats['system_memory_available'] = memory.available
            
            # GPU memory if available
            if torch.cuda.is_available():
                memory_stats['gpu_memory_allocated'] = torch.cuda.memory_allocated()
                memory_stats['gpu_memory_reserved'] = torch.cuda.memory_reserved()
                memory_stats['gpu_memory_max_allocated'] = torch.cuda.max_memory_allocated()
                
        except Exception as e:
            memory_stats['memory_error'] = str(e)
        
        return memory_stats
    
    def add_breakpoint(self, name: str):
        """Add debug breakpoint."""
        self.debug_breakpoints.add(name)
    
    def check_breakpoint(self, name: str) -> bool:
        """Check if breakpoint should trigger."""
        return self.enabled and name in self.debug_breakpoints
    
    def export_diagnostics(self, filepath: str):
        """Export diagnostic data to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.diagnostic_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to export diagnostics: {e}")


class PhotonLogger:
    """Main logging system for Photon Neuromorphics SDK."""
    
    def __init__(self, name: str = "photon_neuro", log_level: str = "INFO",
                 log_dir: str = None, enable_console: bool = True,
                 enable_file: bool = True, enable_metrics: bool = True):
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.progress_tracker = ProgressTracker()
        self.diagnostic_mode = DiagnosticMode()
        
        # Setup main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path.home() / ".photon_neuro" / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup handlers
        self.handlers = {}
        
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handlers()
        
        if enable_metrics:
            self._setup_metrics_handler()
        
        # Performance monitoring
        self.enable_performance_monitoring = enable_metrics
        self._performance_monitor_thread = None
        self._stop_monitoring = threading.Event()
    
    def _setup_console_handler(self):
        """Setup console logging handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.handlers['console'] = handler
    
    def _setup_file_handlers(self):
        """Setup file logging handlers."""
        # Main log file with rotation
        main_log = self.log_dir / "photon_neuro.log"
        handler = logging.handlers.RotatingFileHandler(
            main_log, maxBytes=10*1024*1024, backupCount=5
        )
        handler.setFormatter(StructuredFormatter(include_metrics=True))
        self.logger.addHandler(handler)
        self.handlers['file'] = handler
        
        # Error-only log file
        error_log = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log, maxBytes=5*1024*1024, backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter(include_metrics=False))
        self.logger.addHandler(error_handler)
        self.handlers['error'] = error_handler
        
        # Performance metrics log
        metrics_log = self.log_dir / "metrics.log"
        metrics_handler = logging.handlers.TimedRotatingFileHandler(
            metrics_log, when='midnight', backupCount=7
        )
        metrics_handler.addFilter(lambda record: hasattr(record, 'metric_type'))
        metrics_handler.setFormatter(StructuredFormatter(include_metrics=False))
        self.logger.addHandler(metrics_handler)
        self.handlers['metrics'] = metrics_handler
    
    def _setup_metrics_handler(self):
        """Setup metrics collection handler."""
        class MetricsHandler(logging.Handler):
            def __init__(self, metrics_collector):
                super().__init__()
                self.metrics_collector = metrics_collector
            
            def emit(self, record):
                if hasattr(record, 'metric_name') and hasattr(record, 'metric_value'):
                    self.metrics_collector.record_metric(
                        name=record.metric_name,
                        value=record.metric_value,
                        unit=getattr(record, 'metric_unit', ''),
                        component=getattr(record, 'component', None),
                        **getattr(record, 'metric_metadata', {})
                    )
        
        metrics_handler = MetricsHandler(self.metrics_collector)
        self.logger.addHandler(metrics_handler)
        self.handlers['metrics_collector'] = metrics_handler
    
    def start_performance_monitoring(self, interval: float = 5.0):
        """Start background performance monitoring."""
        if self._performance_monitor_thread and self._performance_monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._performance_monitor_thread = threading.Thread(
            target=self._performance_monitor_loop,
            args=(interval,)
        )
        self._performance_monitor_thread.daemon = True
        self._performance_monitor_thread.start()
    
    def stop_performance_monitoring(self):
        """Stop background performance monitoring."""
        if self._performance_monitor_thread:
            self._stop_monitoring.set()
            self._performance_monitor_thread.join(timeout=5.0)
    
    def _performance_monitor_loop(self, interval: float):
        """Background performance monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                # System metrics
                self.metrics_collector.record_metric(
                    'cpu_percent', psutil.cpu_percent(), 'percent'
                )
                
                memory = psutil.virtual_memory()
                self.metrics_collector.record_metric(
                    'memory_percent', memory.percent, 'percent'
                )
                self.metrics_collector.record_metric(
                    'memory_used_gb', memory.used / 1024**3, 'GB'
                )
                
                # GPU metrics
                if torch.cuda.is_available():
                    self.metrics_collector.record_metric(
                        'gpu_memory_allocated_mb',
                        torch.cuda.memory_allocated() / 1024**2,
                        'MB'
                    )
                    
                    if hasattr(torch.cuda, 'utilization'):
                        self.metrics_collector.record_metric(
                            'gpu_utilization', torch.cuda.utilization(), 'percent'
                        )
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
            
            self._stop_monitoring.wait(interval)
    
    # Convenience methods
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def metric(self, name: str, value: float, unit: str = "", 
              component: str = None, **metadata):
        """Log performance metric."""
        self.logger.info(f"Metric: {name} = {value} {unit}", extra={
            'metric_type': True,
            'metric_name': name,
            'metric_value': value,
            'metric_unit': unit,
            'component': component,
            'metric_metadata': metadata
        })


# Global logger instance
global_logger = PhotonLogger()


# Decorators for automatic logging and monitoring
def log_execution_time(logger: PhotonLogger = None, metric_name: str = None):
    """Decorator to log function execution time."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or global_logger
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                _logger.info(f"Function {func.__name__} completed in {execution_time:.4f}s")
                
                if metric_name:
                    _logger.metrics_collector.record_metric(
                        metric_name, execution_time, 'seconds',
                        component=func.__module__
                    )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                _logger.error(f"Function {func.__name__} failed after {execution_time:.4f}s: {e}")
                raise
        
        return wrapper
    return decorator


def monitor_memory_usage(logger: PhotonLogger = None):
    """Decorator to monitor memory usage during function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or global_logger
            
            # Memory before
            memory_before = psutil.virtual_memory().used / 1024**2  # MB
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            try:
                result = func(*args, **kwargs)
                
                # Memory after
                memory_after = psutil.virtual_memory().used / 1024**2
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                memory_delta = memory_after - memory_before
                gpu_memory_delta = gpu_memory_after - gpu_memory_before
                
                _logger.info(f"Function {func.__name__} memory delta: {memory_delta:.1f} MB RAM, {gpu_memory_delta:.1f} MB GPU")
                
                return result
                
            except Exception as e:
                memory_after = psutil.virtual_memory().used / 1024**2
                memory_delta = memory_after - memory_before
                _logger.error(f"Function {func.__name__} failed with memory delta: {memory_delta:.1f} MB")
                raise
        
        return wrapper
    return decorator


def track_progress(task_name: str, total_steps: int, logger: PhotonLogger = None):
    """Decorator to automatically track progress for iterative functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or global_logger
            _logger.progress_tracker.start_task(task_name, total_steps)
            
            try:
                # Inject progress update function into kwargs
                def update_progress(step: int, stage: str = None):
                    _logger.progress_tracker.update_progress(task_name, step, stage)
                
                kwargs['progress_callback'] = update_progress
                
                result = func(*args, **kwargs)
                _logger.progress_tracker.finish_task(task_name)
                return result
                
            except Exception as e:
                _logger.progress_tracker.finish_task(task_name)
                raise
        
        return wrapper
    return decorator