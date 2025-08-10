"""
Distributed Photonic AI Module
==============================

Federated learning, distributed training, and multi-node photonic AI systems.
"""

from .federated_learning import (
    FederatedPhotonicTrainer, PhotonicFederatedClient, PhotonicFederatedServer,
    PhotonicAggregationStrategy, OptoelectronicSecureAggregation,
    DistributedPhotonicsManager
)

from .distributed_training import (
    DistributedPhotonicTraining, PhotonicDataParallel, PhotonicModelParallel,
    OpticalAllReduce, PhotonicGradientCompression
)

__all__ = [
    # Federated Learning
    "FederatedPhotonicTrainer",
    "PhotonicFederatedClient",
    "PhotonicFederatedServer",
    "PhotonicAggregationStrategy",
    "OptoelectronicSecureAggregation",
    "DistributedPhotonicsManager",
    
    # Distributed Training
    "DistributedPhotonicTraining",
    "PhotonicDataParallel",
    "PhotonicModelParallel",
    "OpticalAllReduce",
    "PhotonicGradientCompression"
]