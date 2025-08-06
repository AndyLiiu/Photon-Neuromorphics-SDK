"""
Loss functions for photonic neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def mse_loss(output: torch.Tensor, target: torch.Tensor, 
            optical_efficiency: Optional[float] = None) -> torch.Tensor:
    """Mean squared error loss with optional optical efficiency weighting."""
    base_loss = F.mse_loss(output, target)
    
    if optical_efficiency is not None and optical_efficiency > 0:
        # Scale loss by optical efficiency to encourage high-efficiency solutions
        efficiency_weight = 1.0 / (optical_efficiency + 1e-6)
        return base_loss * efficiency_weight
    
    return base_loss


def spike_loss(output_spikes: torch.Tensor, target_spikes: torch.Tensor,
               temporal_weight: float = 1.0, rate_weight: float = 1.0) -> torch.Tensor:
    """Loss function for spiking neural networks."""
    
    # Spike timing loss (MSE on spike times)
    timing_loss = F.mse_loss(output_spikes, target_spikes)
    
    # Spike rate loss (difference in total spike counts)
    output_rate = torch.sum(output_spikes, dim=-1)  # Sum over time dimension
    target_rate = torch.sum(target_spikes, dim=-1)
    rate_loss = F.mse_loss(output_rate, target_rate)
    
    return temporal_weight * timing_loss + rate_weight * rate_loss


def photonic_loss(output: torch.Tensor, target: torch.Tensor,
                 power_penalty: float = 0.1, phase_penalty: float = 0.05) -> torch.Tensor:
    """Photonic-specific loss function accounting for optical constraints."""
    
    # Base prediction loss
    prediction_loss = F.mse_loss(output, target)
    
    # Power consumption penalty (encourage low power solutions)
    if torch.is_complex(output):
        optical_power = torch.abs(output)**2
    else:
        optical_power = output**2
        
    power_loss = power_penalty * torch.mean(optical_power)
    
    # Phase stability penalty (encourage stable phases)
    phase_loss = 0.0
    if torch.is_complex(output):
        phases = torch.angle(output)
        # Penalize large phase variations
        phase_variations = torch.std(phases, dim=0)
        phase_loss = phase_penalty * torch.mean(phase_variations)
    
    return prediction_loss + power_loss + phase_loss


def coherent_loss(output: torch.Tensor, target: torch.Tensor,
                 coherence_weight: float = 0.1) -> torch.Tensor:
    """Loss function that preserves optical coherence."""
    
    if not torch.is_complex(output) or not torch.is_complex(target):
        # Fall back to standard MSE if not complex
        return F.mse_loss(output.real if torch.is_complex(output) else output,
                         target.real if torch.is_complex(target) else target)
    
    # Amplitude loss
    output_amp = torch.abs(output)
    target_amp = torch.abs(target)
    amplitude_loss = F.mse_loss(output_amp, target_amp)
    
    # Phase loss (using complex exponential)
    output_phase = torch.exp(1j * torch.angle(output))
    target_phase = torch.exp(1j * torch.angle(target))
    phase_loss = F.mse_loss(output_phase.real, target_phase.real) + \
                 F.mse_loss(output_phase.imag, target_phase.imag)
    
    # Coherence preservation term
    # Encourage maintaining relative phases between different modes/channels
    if output.dim() > 1:
        coherence_loss = 0.0
        for dim in range(1, output.dim()):
            # Cross-correlation between adjacent channels
            output_rolled = torch.roll(output, shifts=1, dims=dim)
            target_rolled = torch.roll(target, shifts=1, dims=dim)
            
            output_correlation = torch.sum(output * torch.conj(output_rolled), dim=0)
            target_correlation = torch.sum(target * torch.conj(target_rolled), dim=0)
            
            coherence_loss += F.mse_loss(output_correlation.real, target_correlation.real) + \
                             F.mse_loss(output_correlation.imag, target_correlation.imag)
                             
        coherence_loss /= (output.dim() - 1)  # Average over dimensions
    else:
        coherence_loss = 0.0
    
    return amplitude_loss + phase_loss + coherence_weight * coherence_loss


class AdaptiveLoss(nn.Module):
    """Adaptive loss function that adjusts based on training progress."""
    
    def __init__(self, base_loss_fn=F.mse_loss, adaptation_rate: float = 0.01):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.adaptation_rate = adaptation_rate
        self.register_buffer('loss_history', torch.zeros(100))  # Store last 100 losses
        self.register_buffer('step_count', torch.zeros(1))
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calculate base loss
        base_loss = self.base_loss_fn(output, target)
        
        # Update loss history
        step = int(self.step_count.item()) % 100
        self.loss_history[step] = base_loss.detach()
        self.step_count += 1
        
        # Adaptive weighting based on loss trend
        if self.step_count > 10:
            recent_losses = self.loss_history[:min(int(self.step_count), 100)]
            loss_trend = torch.mean(recent_losses[-10:]) - torch.mean(recent_losses[-20:-10])
            
            # If loss is increasing, increase learning emphasis
            if loss_trend > 0:
                adaptation_factor = 1.0 + self.adaptation_rate
            else:
                adaptation_factor = 1.0
                
            return base_loss * adaptation_factor
        
        return base_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for optical feature learning."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim) tensor of optical features
            labels: (batch_size,) tensor of class labels
        """
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarities
        similarities = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive and negative masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        positives_mask = labels_equal & ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        negatives_mask = ~labels_equal
        
        # Compute contrastive loss
        # Positive pairs should have high similarity
        positive_loss = 0.0
        if positives_mask.any():
            positive_similarities = similarities[positives_mask]
            positive_loss = -torch.mean(positive_similarities)
        
        # Negative pairs should have low similarity
        negative_loss = 0.0
        if negatives_mask.any():
            negative_similarities = similarities[negatives_mask]
            negative_loss = torch.mean(F.relu(negative_similarities - self.margin))
        
        return positive_loss + negative_loss


class QuantumLoss(nn.Module):
    """Loss function for quantum photonic systems."""
    
    def __init__(self, fidelity_weight: float = 1.0, purity_weight: float = 0.1):
        super().__init__()
        self.fidelity_weight = fidelity_weight
        self.purity_weight = purity_weight
        
    def forward(self, output_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output_state: Complex tensor representing output quantum state
            target_state: Complex tensor representing target quantum state
        """
        # State fidelity loss
        # F = |⟨ψ_target|ψ_output⟩|²
        overlap = torch.sum(torch.conj(target_state) * output_state, dim=-1)
        fidelity = torch.abs(overlap)**2
        fidelity_loss = -torch.mean(fidelity)  # Maximize fidelity
        
        # State purity loss (encourage pure states)
        # Purity = Tr(ρ²) where ρ is the density matrix
        # For pure states: ρ = |ψ⟩⟨ψ|, so Tr(ρ²) = 1
        state_norm_sq = torch.sum(torch.abs(output_state)**2, dim=-1)
        purity = state_norm_sq**2 / torch.sum(torch.abs(output_state)**4, dim=-1)
        purity_loss = -torch.mean(purity)  # Maximize purity
        
        return self.fidelity_weight * fidelity_loss + self.purity_weight * purity_loss


class OpticalDistillationLoss(nn.Module):
    """Knowledge distillation loss for transferring electronic NN to photonic NN."""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation vs hard target loss
        
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor,
               targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            student_output: Output from photonic (student) network
            teacher_output: Output from electronic (teacher) network  
            targets: Hard targets (optional)
        """
        # Soft target loss (knowledge distillation)
        student_soft = F.log_softmax(student_output / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_output / self.temperature, dim=1)
        
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss *= self.temperature**2  # Scale by temperature squared
        
        # Hard target loss
        hard_loss = 0.0
        if targets is not None:
            hard_loss = F.cross_entropy(student_output, targets)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


def variational_loss(mu: torch.Tensor, logvar: torch.Tensor,
                    reconstruction: torch.Tensor, target: torch.Tensor,
                    beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variational loss for optical variational autoencoders."""
    
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstruction, target, reduction='sum')
    
    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0,I)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                margin: float = 1.0) -> torch.Tensor:
    """Triplet loss for optical embeddings."""
    
    # Calculate distances
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
    loss = F.relu(pos_dist - neg_dist + margin)
    
    return torch.mean(loss)