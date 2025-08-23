"""
Autonomous Performance Optimization and Scaling
==============================================

Advanced performance optimization, auto-scaling, load balancing, and distributed
processing capabilities for high-performance photonic neural network execution.
"""

from .cache import IntelligentCache as PhotonicCache
from .autoscaler import AutoScaler as QuantumAutoscaler
from .optimization_engine import OptimizationEngine
# from .monitoring import AdaptiveMonitor  # Skip for now - missing seaborn dependency

__all__ = [
    "PhotonicCache",
    "QuantumAutoscaler", 
    "OptimizationEngine"
]