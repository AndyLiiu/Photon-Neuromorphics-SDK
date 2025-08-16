"""
Training algorithms for photonic neural networks.
"""

from .optimizers import OpticalAdam, OpticalSGD, OpticalTrainer
from .losses import spike_loss, mse_loss

# Simplified imports for Generation 5 - focus on core functionality
STDPTrainer = None
OpticalBackprop = None
InSituTraining = None

__all__ = [
    "OpticalAdam",
    "OpticalSGD", 
    "OpticalTrainer",
    "spike_loss",
    "mse_loss",
    "STDPTrainer",
    "OpticalBackprop",
    "InSituTraining",
]