"""
Real-time Monitoring System
===========================

Continuous monitoring and alerting for quality metrics.
"""

import time
import threading
import queue
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
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