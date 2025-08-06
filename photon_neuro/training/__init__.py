"""
Training algorithms for photonic neural networks.
"""

from .optimizers import OpticalAdam, OpticalSGD, OpticalTrainer
from .losses import spike_loss, mse_loss, photonic_loss, coherent_loss
from .algorithms import STDPTrainer, OpticalBackprop, InSituTraining

__all__ = [
    "OpticalAdam",
    "OpticalSGD", 
    "OpticalTrainer",
    "spike_loss",
    "mse_loss",
    "photonic_loss",
    "coherent_loss",
    "STDPTrainer",
    "OpticalBackprop",
    "InSituTraining",
]