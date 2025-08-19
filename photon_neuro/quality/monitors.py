"""
Autonomous Real-time Monitoring System
=====================================

Comprehensive monitoring, alerting, and autonomous quality assurance with
predictive analytics and self-healing capabilities.
"""

import time
import threading
import queue
import json
import statistics
import asyncio
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import psutil

from ..utils.logging_system import global_logger


@dataclass
class MetricPoint:
    """Single metric measurement."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]


@dataclass
class Alert:
    """Quality alert."""
    level: str  # 'warning', 'error', 'critical'
    message: str
    metric_name: str
    threshold: float
    actual_value: float
    timestamp: float


class MetricsCollector:
    """Collects and stores metrics in real-time."""
    
    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self.metrics = queue.Queue(maxsize=buffer_size)
        self.metric_history = {}
        self.lock = threading.Lock()
        self.logger = global_logger
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a single metric point."""
        tags = tags or {}
        metric = MetricPoint(name, value, time.time(), tags)
        
        try:
            self.metrics.put_nowait(metric)
            
            # Store in history
            with self.lock:
                if name not in self.metric_history:
                    self.metric_history[name] = []
                
                self.metric_history[name].append(metric)
                
                # Keep only recent history
                if len(self.metric_history[name]) > 1000:
                    self.metric_history[name] = self.metric_history[name][-500:]
                    
        except queue.Full:
            self.logger.warning(f"Metrics buffer full, dropping metric: {name}")
    
    def get_recent_metrics(self, name: str, count: int = 100) -> List[MetricPoint]:
        """Get recent metrics for a specific name."""
        with self.lock:
            if name in self.metric_history:
                return self.metric_history[name][-count:]
            return []
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, float]]:
        """Get summary statistics for a metric."""
        recent = self.get_recent_metrics(name, 100)
        
        if not recent:
            return None
        
        values = [m.value for m in recent]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0.0
        }


class AlertSystem:
    """Manages alerts and thresholds."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.thresholds = {}
        self.alert_handlers = []
        self.active_alerts = {}
        self.logger = global_logger
    
    def add_threshold(self, metric_name: str, warning_threshold: float, 
                     error_threshold: float, critical_threshold: float):
        """Add alerting thresholds for a metric."""
        self.thresholds[metric_name] = {
            "warning": warning_threshold,
            "error": error_threshold,
            "critical": critical_threshold
        }
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def check_thresholds(self):
        """Check all metrics against thresholds and trigger alerts."""
        for metric_name, thresholds in self.thresholds.items():
            summary = self.metrics_collector.get_metric_summary(metric_name)
            
            if not summary:
                continue
            
            latest_value = summary["latest"]
            
            # Check thresholds (assuming higher values are worse)
            alert_level = None
            threshold_value = None
            
            if latest_value >= thresholds["critical"]:
                alert_level = "critical"
                threshold_value = thresholds["critical"]
            elif latest_value >= thresholds["error"]:
                alert_level = "error"
                threshold_value = thresholds["error"]
            elif latest_value >= thresholds["warning"]:
                alert_level = "warning"
                threshold_value = thresholds["warning"]
            
            if alert_level:
                alert_key = f"{metric_name}_{alert_level}"
                
                # Avoid duplicate alerts
                if alert_key not in self.active_alerts:
                    alert = Alert(
                        level=alert_level,
                        message=f"Metric {metric_name} exceeded {alert_level} threshold",
                        metric_name=metric_name,
                        threshold=threshold_value,
                        actual_value=latest_value,
                        timestamp=time.time()
                    )
                    
                    self.active_alerts[alert_key] = alert
                    self._trigger_alert(alert)
            else:
                # Clear resolved alerts
                for level in ["warning", "error", "critical"]:
                    alert_key = f"{metric_name}_{level}"
                    if alert_key in self.active_alerts:
                        del self.active_alerts[alert_key]
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert to all handlers."""
        self.logger.warning(f"ALERT [{alert.level.upper()}]: {alert.message}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")


class PerformanceMonitor:
    """Monitors system and application performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        self.logger = global_logger
    
    def start_monitoring(self, interval: float = 5.0):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # System metrics
                self._collect_system_metrics()
                
                # Application metrics
                self._collect_application_metrics()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics_collector.record_metric("system.cpu_percent", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric("system.memory_percent", memory.percent)
        self.metrics_collector.record_metric("system.memory_available_mb", memory.available / 1024 / 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics_collector.record_metric("system.disk_percent", disk_percent)
        
        # Process-specific metrics
        process = psutil.Process()
        proc_memory = process.memory_info()
        self.metrics_collector.record_metric("process.memory_rss_mb", proc_memory.rss / 1024 / 1024)
        self.metrics_collector.record_metric("process.memory_vms_mb", proc_memory.vms / 1024 / 1024)
        
        try:
            cpu_times = process.cpu_times()
            self.metrics_collector.record_metric("process.cpu_user_time", cpu_times.user)
            self.metrics_collector.record_metric("process.cpu_system_time", cpu_times.system)
        except AttributeError:
            pass  # Some systems don't support cpu_times
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        # Thread count
        thread_count = threading.active_count()
        self.metrics_collector.record_metric("app.thread_count", thread_count)
        
        # Metric buffer usage
        try:
            buffer_size = self.metrics_collector.metrics.qsize()
            max_size = self.metrics_collector.buffer_size
            buffer_percent = (buffer_size / max_size) * 100
            self.metrics_collector.record_metric("app.metrics_buffer_percent", buffer_percent)
        except Exception:
            pass


class RealTimeMonitor:
    """Main real-time monitoring coordinator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector(
            buffer_size=self.config.get("buffer_size", 10000)
        )
        self.alert_system = AlertSystem(self.metrics_collector)
        self.performance_monitor = PerformanceMonitor(self.metrics_collector)
        self.logger = global_logger
        
        # Setup default thresholds
        self._setup_default_thresholds()
        
        # Setup default alert handlers
        self._setup_default_handlers()
        
        # Start monitoring
        self.start()
    
    def start(self):
        """Start all monitoring components."""
        self.performance_monitor.start_monitoring(
            interval=self.config.get("monitoring_interval", 5.0)
        )
        
        # Start threshold checking
        self._start_threshold_checking()
        
        self.logger.info("Real-time monitoring started")
    
    def stop(self):
        """Stop all monitoring components."""
        self.performance_monitor.stop_monitoring()
        self.logger.info("Real-time monitoring stopped")
    
    def _setup_default_thresholds(self):
        """Setup default alerting thresholds."""
        # System thresholds
        self.alert_system.add_threshold("system.cpu_percent", 70.0, 85.0, 95.0)
        self.alert_system.add_threshold("system.memory_percent", 80.0, 90.0, 95.0)
        self.alert_system.add_threshold("system.disk_percent", 85.0, 92.0, 98.0)
        
        # Process thresholds
        self.alert_system.add_threshold("process.memory_rss_mb", 500.0, 1000.0, 2000.0)
        self.alert_system.add_threshold("app.metrics_buffer_percent", 70.0, 85.0, 95.0)
    
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        def log_alert(alert: Alert):
            self.logger.warning(f"Alert: {alert.message} (Value: {alert.actual_value:.2f}, Threshold: {alert.threshold:.2f})")
        
        self.alert_system.add_alert_handler(log_alert)
    
    def _start_threshold_checking(self):
        """Start threshold checking in background."""
        def check_loop():
            while True:
                try:
                    self.alert_system.check_thresholds()
                    time.sleep(self.config.get("alert_check_interval", 10.0))
                except Exception as e:
                    self.logger.error(f"Threshold checking error: {e}")
                    time.sleep(10.0)
        
        thread = threading.Thread(target=check_loop, daemon=True)
        thread.start()


class QualityDashboard:
    """Simple quality metrics dashboard."""
    
    def __init__(self, monitor: RealTimeMonitor, output_file: Optional[str] = None):
        self.monitor = monitor
        self.output_file = output_file or "quality_dashboard.json"
        self.logger = global_logger
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate current quality report."""
        report = {
            "timestamp": time.time(),
            "system_metrics": {},
            "application_metrics": {},
            "active_alerts": [],
            "metric_summaries": {}
        }
        
        # System metrics
        system_metrics = [
            "system.cpu_percent", "system.memory_percent", "system.disk_percent"
        ]
        
        for metric in system_metrics:
            summary = self.monitor.metrics_collector.get_metric_summary(metric)
            if summary:
                report["system_metrics"][metric] = summary
        
        # Application metrics
        app_metrics = [
            "process.memory_rss_mb", "app.thread_count", "app.metrics_buffer_percent"
        ]
        
        for metric in app_metrics:
            summary = self.monitor.metrics_collector.get_metric_summary(metric)
            if summary:
                report["application_metrics"][metric] = summary
        
        # Active alerts
        for alert in self.monitor.alert_system.active_alerts.values():
            report["active_alerts"].append(asdict(alert))
        
        return report
    
    def save_report(self):
        """Save current report to file."""
        report = self.generate_report()
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Quality dashboard saved to {self.output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dashboard: {e}")
    
    def print_summary(self):
        """Print a summary to console."""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("QUALITY DASHBOARD SUMMARY")
        print("="*60)
        
        # System status
        print("\nSYSTEM METRICS:")
        for metric, summary in report["system_metrics"].items():
            print(f"  {metric}: {summary['latest']:.1f} (avg: {summary['avg']:.1f})")
        
        # Application status
        print("\nAPPLICATION METRICS:")
        for metric, summary in report["application_metrics"].items():
            print(f"  {metric}: {summary['latest']:.1f} (avg: {summary['avg']:.1f})")
        
        # Alerts
        print(f"\nACTIVE ALERTS: {len(report['active_alerts'])}")
        for alert in report["active_alerts"]:
            print(f"  [{alert['level'].upper()}] {alert['message']}")
        
        print("="*60)


class AlertSeverity(Enum):
    """Alert severity levels for enhanced monitoring."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TrendAnalyzer:
    """Analyzes metric trends for predictive alerting."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metric_windows = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_data_point(self, metric_name: str, value: float, timestamp: float):
        """Add a data point for trend analysis."""
        self.metric_windows[metric_name].append((timestamp, value))
    
    def get_trend(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get trend analysis for a metric."""
        window = self.metric_windows[metric_name]
        if len(window) < 10:  # Need minimum data points
            return None
            
        values = [point[1] for point in window]
        timestamps = [point[0] for point in window]
        
        # Calculate basic statistics
        trend_analysis = {
            'current': values[-1],
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'trend_direction': self._calculate_trend_direction(values),
            'volatility': self._calculate_volatility(values),
            'predicted_next': self._predict_next_value(values, timestamps)
        }
        
        return trend_analysis
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 5:
            return "insufficient_data"
            
        # Compare recent values with older values
        recent = values[-5:]
        older = values[:-5]
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate metric volatility."""
        if len(values) < 2:
            return 0.0
            
        # Calculate coefficient of variation
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.0
            
        std_dev = statistics.stdev(values)
        return (std_dev / mean_val) * 100
    
    def _predict_next_value(self, values: List[float], timestamps: List[float]) -> float:
        """Simple linear prediction of next value."""
        if len(values) < 3:
            return values[-1] if values else 0.0
            
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope and intercept
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next value
        next_x = n
        predicted = slope * next_x + intercept
        
        return max(0, predicted)  # Ensure non-negative


class AutonomousQualityMonitor(RealTimeMonitor):
    """Autonomous quality monitor with predictive analytics and self-healing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.auto_remediation_enabled = config.get("auto_remediation", True) if config else True
        self.prediction_alerts = {}
        self.adaptive_thresholds = {}
        
        # Enhanced alert handlers
        self._setup_autonomous_handlers()
        
    def _setup_autonomous_handlers(self):
        """Setup autonomous alert handling."""
        def autonomous_alert_handler(alert: Alert):
            """Handle alerts autonomously with potential remediation."""
            self.logger.info(f"Autonomous handler processing alert: {alert.message}")
            
            # Record alert for learning
            self._record_alert_for_learning(alert)
            
            # Attempt autonomous remediation
            if self.auto_remediation_enabled:
                remediation_applied = self._attempt_remediation(alert)
                if remediation_applied:
                    self.logger.info(f"Applied autonomous remediation for {alert.metric_name}")
            
            # Update adaptive thresholds
            self._update_adaptive_threshold(alert.metric_name, alert.actual_value)
        
        self.alert_system.add_alert_handler(autonomous_alert_handler)
    
    def _record_alert_for_learning(self, alert: Alert):
        """Record alert for machine learning and pattern recognition."""
        alert_record = {
            'timestamp': alert.timestamp,
            'metric': alert.metric_name,
            'level': alert.level,
            'value': alert.actual_value,
            'threshold': alert.threshold,
            'context': self._get_system_context()
        }
        
        # Store for pattern analysis (simplified implementation)
        if not hasattr(self, 'alert_history'):
            self.alert_history = deque(maxlen=1000)
        
        self.alert_history.append(alert_record)
    
    def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context for alert correlation."""
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'thread_count': threading.active_count(),
            'time_of_day': time.strftime('%H'),
            'day_of_week': time.strftime('%A')
        }
    
    def _attempt_remediation(self, alert: Alert) -> bool:
        """Attempt autonomous remediation based on alert type."""
        metric_name = alert.metric_name
        
        if metric_name == "system.memory_percent" and alert.actual_value > 90:
            return self._remediate_high_memory_usage()
        elif metric_name == "app.metrics_buffer_percent" and alert.actual_value > 85:
            return self._remediate_buffer_overflow()
        elif "cpu_percent" in metric_name and alert.actual_value > 95:
            return self._remediate_high_cpu_usage()
        
        return False
    
    def _remediate_high_memory_usage(self) -> bool:
        """Remediate high memory usage."""
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear metric history to free memory
            if hasattr(self, 'alert_history'):
                self.alert_history.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory remediation failed: {e}")
            return False
    
    def _remediate_buffer_overflow(self) -> bool:
        """Remediate metrics buffer overflow."""
        try:
            # Clear old metrics from buffer
            cleared_count = 0
            while not self.metrics_collector.metrics.empty() and cleared_count < 1000:
                try:
                    self.metrics_collector.metrics.get_nowait()
                    cleared_count += 1
                except queue.Empty:
                    break
            
            self.logger.info(f"Cleared {cleared_count} metrics from buffer")
            return True
            
        except Exception as e:
            self.logger.error(f"Buffer remediation failed: {e}")
            return False
    
    def _remediate_high_cpu_usage(self) -> bool:
        """Remediate high CPU usage."""
        try:
            # Increase monitoring intervals to reduce CPU load
            current_interval = self.config.get("monitoring_interval", 5.0)
            new_interval = min(current_interval * 1.5, 30.0)
            
            self.config["monitoring_interval"] = new_interval
            self.logger.info(f"Increased monitoring interval to {new_interval}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CPU remediation failed: {e}")
            return False
    
    def _update_adaptive_threshold(self, metric_name: str, current_value: float):
        """Update adaptive thresholds based on historical performance."""
        if metric_name not in self.adaptive_thresholds:
            self.adaptive_thresholds[metric_name] = {
                'values': deque(maxlen=100),
                'last_update': time.time()
            }
        
        threshold_data = self.adaptive_thresholds[metric_name]
        threshold_data['values'].append(current_value)
        
        # Update thresholds every hour
        if time.time() - threshold_data['last_update'] > 3600:
            self._recalculate_thresholds(metric_name)
            threshold_data['last_update'] = time.time()
    
    def _recalculate_thresholds(self, metric_name: str):
        """Recalculate adaptive thresholds based on historical data."""
        if metric_name not in self.adaptive_thresholds:
            return
            
        values = list(self.adaptive_thresholds[metric_name]['values'])
        if len(values) < 20:  # Need sufficient data
            return
        
        # Calculate percentile-based thresholds
        values_sorted = sorted(values)
        n = len(values_sorted)
        
        # 75th, 90th, 95th percentiles as thresholds
        warning_threshold = values_sorted[int(n * 0.75)]
        error_threshold = values_sorted[int(n * 0.90)]
        critical_threshold = values_sorted[int(n * 0.95)]
        
        # Update alert system thresholds
        self.alert_system.add_threshold(
            metric_name, 
            warning_threshold, 
            error_threshold, 
            critical_threshold
        )
        
        self.logger.info(f"Updated adaptive thresholds for {metric_name}: "
                        f"W:{warning_threshold:.1f}, E:{error_threshold:.1f}, C:{critical_threshold:.1f}")


class AnomalyDetector:
    """Detects anomalies in metric patterns."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baseline_data = defaultdict(lambda: deque(maxlen=200))
        
    def add_data_point(self, metric_name: str, value: float):
        """Add data point for anomaly detection."""
        self.baseline_data[metric_name].append(value)
    
    def is_anomaly(self, metric_name: str, value: float) -> bool:
        """Detect if a value is anomalous."""
        baseline = self.baseline_data[metric_name]
        if len(baseline) < 30:  # Need sufficient baseline
            return False
        
        baseline_list = list(baseline)
        mean = statistics.mean(baseline_list)
        std_dev = statistics.stdev(baseline_list)
        
        # Z-score based anomaly detection
        if std_dev == 0:
            return False
            
        z_score = abs(value - mean) / std_dev
        return z_score > self.sensitivity


# Global autonomous monitor instance
autonomous_monitor = None

def get_global_monitor() -> AutonomousQualityMonitor:
    """Get or create global autonomous monitor."""
    global autonomous_monitor
    if autonomous_monitor is None:
        autonomous_monitor = AutonomousQualityMonitor()
    return autonomous_monitor