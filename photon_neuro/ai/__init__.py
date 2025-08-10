"""
Advanced AI Integration Module
==============================

Revolutionary AI architectures adapted for photonic neural networks.
"""

from .transformers import (
    OpticalTransformer, PhotonicSelfAttention, OpticalMultiHeadAttention,
    OpticalPositionalEncoding, OpticalFeedForward, PhotonicGPT,
    InterferenceSelfAttention, MZIAttentionMesh
)

from .neural_architecture_search import (
    PhotonicNAS, ArchitectureSearchSpace, PhotonicArchitectureEvaluator,
    OpticalArchitectureOptimizer, EvolutionaryPhotonicNAS
)

__all__ = [
    # Optical Transformers
    "OpticalTransformer",
    "PhotonicSelfAttention", 
    "OpticalMultiHeadAttention",
    "OpticalPositionalEncoding",
    "OpticalFeedForward",
    "PhotonicGPT",
    "InterferenceSelfAttention",
    "MZIAttentionMesh",
    
    # Neural Architecture Search
    "PhotonicNAS",
    "ArchitectureSearchSpace",
    "PhotonicArchitectureEvaluator", 
    "OpticalArchitectureOptimizer",
    "EvolutionaryPhotonicNAS"
]