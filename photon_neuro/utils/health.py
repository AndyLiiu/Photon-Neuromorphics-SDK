"""
Health Check System
==================
"""

import time
import threading
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    response_time_ms: float

class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
    
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        
        try:
            status, message, details = self._perform_check()
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time_ms=response_time
            )
    
    def _perform_check(self) -> tuple:
        """Override this method to implement specific health check."""
        return HealthStatus.HEALTHY, "OK", {}

class SystemHealthCheck(HealthCheck):
    """System resource health check."""
    
    def _perform_check(self) -> tuple:
        try:
            import psutil
        except ImportError:
            return HealthStatus.WARNING, "psutil not available", {}
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": (disk.used / disk.total) * 100
        }
        
        # Determine status
        if cpu_percent > 90 or memory.percent > 90:
            return HealthStatus.UNHEALTHY, "High resource usage", details
        elif cpu_percent > 75 or memory.percent > 75:
            return HealthStatus.WARNING, "Elevated resource usage", details
        else:
            return HealthStatus.HEALTHY, "System resources normal", details

class HealthCheckManager:
    """Manages multiple health checks."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.monitoring = False
        self.monitor_thread = None
    
    def register_check(self, check: HealthCheck):
        """Register a health check."""
        self.checks[check.name] = check
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, check in self.checks.items():
            results[name] = check.check()
            self.results[name] = results[name]
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous health monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self, interval: float):
        """Health monitoring loop."""
        while self.monitoring:
            try:
                self.run_all_checks()
                time.sleep(interval)
            except Exception:
                time.sleep(interval)

# Global health check manager
global_health_manager = HealthCheckManager()

# Register default checks
global_health_manager.register_check(SystemHealthCheck("system"))