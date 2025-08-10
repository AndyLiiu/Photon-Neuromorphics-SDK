"""
Quantum Computing Module
========================

Advanced quantum computing capabilities for photonic neural networks.
"""

from .error_correction import (
    SurfaceCode, StabilizerCode, QuantumErrorCorrector,
    LogicalQubitEncoder, SyndromeDecoder, ErrorRecovery
)

__all__ = [
    "SurfaceCode",
    "StabilizerCode", 
    "QuantumErrorCorrector",
    "LogicalQubitEncoder",
    "SyndromeDecoder",
    "ErrorRecovery"
]