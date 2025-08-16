"""
Optimizers for photonic neural networks.
"""

import torch
import torch.optim as optim
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np


class OpticalAdam(optim.Optimizer):
    """Adam optimizer adapted for optical neural networks."""
    
    def __init__(self, params, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0, phase_wrap: bool = True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       phase_wrap=phase_wrap)
        super(OpticalAdam, self).__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('OpticalAdam does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                    
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = (exp_avg_sq.sqrt() / np.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction1
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Phase wrapping for optical phase shifters
                if group['phase_wrap']:
                    p.data = torch.remainder(p.data + np.pi, 2*np.pi) - np.pi
                    
        return loss


class OpticalSGD(optim.SGD):
    """SGD optimizer adapted for optical neural networks."""
    
    def __init__(self, params, lr: float = 1e-2, momentum: float = 0, dampening: float = 0,
                 weight_decay: float = 0, nesterov: bool = False, phase_wrap: bool = True):
        
        # Add optical-specific parameters
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov, 
                       phase_wrap=phase_wrap)
        super(OpticalSGD, self).__init__(params, **defaults)
        
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = super().step(closure)
        
        # Apply phase wrapping after standard SGD update
        for group in self.param_groups:
            if group['phase_wrap']:
                for p in group['params']:
                    if p.grad is not None:
                        # Wrap phases to [-π, π] for phase shifters
                        p.data = torch.remainder(p.data + np.pi, 2*np.pi) - np.pi
                        
        return loss


class OpticalTrainer:
    """High-level trainer for optical neural networks with hardware integration."""
    
    def __init__(self, model, optimizer, loss_function, device: str = "cpu",
                 hardware_interface=None, calibration_data=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.hardware_interface = hardware_interface
        self.calibration_data = calibration_data
        
        # Training state
        self.training_history = {
            'loss': [],
            'optical_power': [],
            'electrical_power': [],
            'temperature': [],
            'phase_drift': []
        }
        
    def train_epoch(self, dataloader, hardware_in_loop: bool = False) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_stats = {
            'loss': 0.0,
            'accuracy': 0.0,
            'optical_efficiency': 0.0,
            'power_consumption': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            try:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                if hardware_in_loop and self.hardware_interface:
                    outputs = self._hardware_forward(data)
                else:
                    outputs = self.model(data)
                    
                # Calculate loss
                loss = self.loss_function(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if hardware_in_loop:
                    # Hardware-aware backpropagation
                    self._hardware_backward(loss)
                else:
                    loss.backward()
                    
                self.optimizer.step()
                
                # Update statistics
                epoch_stats['loss'] += loss.item()
                
                # Calculate accuracy (for classification)
                if targets.dim() == 1 or targets.shape[1] == 1:
                    predicted = torch.argmax(outputs, dim=1) if outputs.dim() > 1 else outputs.round()
                    accuracy = (predicted == targets).float().mean().item()
                    epoch_stats['accuracy'] += accuracy
                    
                # Monitor optical parameters
                if hasattr(self.model, 'efficiency'):
                    epoch_stats['optical_efficiency'] += self.model.efficiency
                    
                if hasattr(self.model, 'calculate_power_consumption'):
                    power = self.model.calculate_power_consumption()
                    epoch_stats['power_consumption'] += power
                    
                # Hardware monitoring
                if hardware_in_loop and self.hardware_interface:
                    self._monitor_hardware()
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
                
        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= num_batches
            
        # Update training history
        for key, value in epoch_stats.items():
            if key in self.training_history:
                self.training_history[key].append(value)
            else:
                self.training_history[key] = [value]
                
        return epoch_stats
        
    def _hardware_forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass through actual hardware."""
        if not self.hardware_interface:
            raise ValueError("No hardware interface available")
            
        try:
            # Convert data to optical signals
            optical_signals = self._encode_optical_input(data)
            
            # Send to hardware
            hardware_output = self.hardware_interface.process(optical_signals)
            
            # Convert back to tensor
            output_tensor = self._decode_optical_output(hardware_output)
            
            return output_tensor
            
        except Exception as e:
            print(f"Hardware forward pass failed: {e}")
            # Fallback to simulation
            return self.model(data)
            
    def _encode_optical_input(self, data: torch.Tensor) -> Any:
        """Encode tensor data as optical signals."""
        # This would interface with actual hardware
        # For now, return data as-is
        return data.detach().numpy()
        
    def _decode_optical_output(self, hardware_output: Any) -> torch.Tensor:
        """Decode optical signals back to tensor."""
        # Convert hardware output back to PyTorch tensor
        if isinstance(hardware_output, np.ndarray):
            return torch.from_numpy(hardware_output).float()
        return torch.tensor(hardware_output).float()
        
    def _hardware_backward(self, loss: torch.Tensor):
        """Hardware-aware backpropagation."""
        # For optical systems, gradients might need special handling
        # due to phase wrapping, saturation, etc.
        
        # Standard backprop first
        loss.backward()
        
        # Apply optical constraints to gradients
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
                    
                    # Apply hardware-specific constraints
                    if hasattr(param, 'optical_constraint'):
                        param.grad = param.optical_constraint(param.grad)
                        
    def _monitor_hardware(self):
        """Monitor hardware parameters during training."""
        if not self.hardware_interface:
            return
            
        try:
            # Monitor optical power
            optical_power = self.hardware_interface.get_optical_power()
            self.training_history['optical_power'].append(optical_power)
            
            # Monitor electrical power
            electrical_power = self.hardware_interface.get_electrical_power()
            self.training_history['electrical_power'].append(electrical_power)
            
            # Monitor temperature
            temperature = self.hardware_interface.get_temperature()
            self.training_history['temperature'].append(temperature)
            
            # Monitor phase drift
            if hasattr(self.hardware_interface, 'get_phase_drift'):
                phase_drift = self.hardware_interface.get_phase_drift()
                self.training_history['phase_drift'].append(phase_drift)
                
        except Exception as e:
            print(f"Hardware monitoring failed: {e}")
            
    def validate(self, dataloader, hardware_in_loop: bool = False) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_stats = {
            'loss': 0.0,
            'accuracy': 0.0,
            'optical_efficiency': 0.0
        }
        
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for data, targets in dataloader:
                try:
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    if hardware_in_loop and self.hardware_interface:
                        outputs = self._hardware_forward(data)
                    else:
                        outputs = self.model(data)
                        
                    loss = self.loss_function(outputs, targets)
                    val_stats['loss'] += loss.item()
                    
                    # Calculate accuracy
                    if targets.dim() == 1 or targets.shape[1] == 1:
                        predicted = torch.argmax(outputs, dim=1) if outputs.dim() > 1 else outputs.round()
                        accuracy = (predicted == targets).float().mean().item()
                        val_stats['accuracy'] += accuracy
                        
                    # Monitor optical efficiency
                    if hasattr(self.model, 'efficiency'):
                        val_stats['optical_efficiency'] += self.model.efficiency
                        
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
                    
        # Average statistics
        for key in val_stats:
            val_stats[key] /= num_batches
            
        return val_stats
        
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'calibration_data': self.calibration_data
        }
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.calibration_data = checkpoint.get('calibration_data', None)
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_history['loss']:
            return {"message": "No training data available"}
            
        return {
            'epochs_trained': len(self.training_history['loss']),
            'final_loss': self.training_history['loss'][-1],
            'best_loss': min(self.training_history['loss']),
            'final_accuracy': self.training_history.get('accuracy', [0])[-1] if self.training_history.get('accuracy') else 0,
            'avg_optical_efficiency': np.mean(self.training_history.get('optical_efficiency', [0])),
            'avg_power_consumption': np.mean(self.training_history.get('power_consumption', [0]))
        }