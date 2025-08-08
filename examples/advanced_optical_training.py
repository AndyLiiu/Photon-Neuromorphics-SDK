#!/usr/bin/env python3
"""
Advanced Optical Training Example
=================================

Demonstrates light-in-the-loop training with hardware calibration,
real-time error correction, and adaptive learning rates.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

import photon_neuro as pn


def generate_synthetic_dataset(n_samples: int = 1000, 
                              input_dim: int = 784,
                              output_dim: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic dataset for optical neural network training."""
    # Create spiral classification dataset
    t = torch.linspace(0, 4*np.pi, n_samples)
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    
    # Create input features (simulate flattened images)
    inputs = torch.randn(n_samples, input_dim)
    
    # Add structured patterns based on spiral coordinates
    for i in range(min(28, int(np.sqrt(input_dim)))):
        for j in range(min(28, int(np.sqrt(input_dim)))):
            idx = i * 28 + j if i * 28 + j < input_dim else 0
            inputs[:, idx] += 0.1 * (x * np.cos(i * 0.1) + y * np.sin(j * 0.1))
    
    # Create labels (0-9 classification)
    labels = (t / (4*np.pi) * output_dim).long() % output_dim
    
    return inputs.float(), labels


class HybridPhotonicMLP(nn.Module):
    """Hybrid photonic-electronic multilayer perceptron."""
    
    def __init__(self, input_dim: int = 784, hidden_dims: List[int] = [256, 128], 
                 output_dim: int = 10, photonic_backend: str = "mzi"):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.photonic_backend = photonic_backend
        
        # Electronic preprocessing layer
        self.input_encoder = nn.Linear(input_dim, hidden_dims[0])
        
        # Photonic layers (MZI meshes)
        self.photonic_layers = nn.ModuleList()
        layer_dims = [hidden_dims[0]] + hidden_dims
        
        for i in range(len(layer_dims) - 1):
            if photonic_backend == "mzi":
                photonic_layer = pn.MZIMesh(
                    size=(layer_dims[i], layer_dims[i+1]),
                    topology='rectangular',
                    phase_encoding='differential'
                )
            elif photonic_backend == "microring":
                photonic_layer = pn.MicroringArray(
                    n_rings=layer_dims[i] * layer_dims[i+1] // 8,
                    free_spectral_range=20e9,
                    quality_factor=10000
                )
            else:
                raise ValueError(f"Unknown photonic backend: {photonic_backend}")
            
            self.photonic_layers.append(photonic_layer)
        
        # Electronic output layer
        self.output_decoder = nn.Linear(hidden_dims[-1], output_dim)
        
        # Optical-to-electronic interfaces
        self.photodetector_arrays = nn.ModuleList([
            pn.PhotodetectorArray(n_detectors=dim, responsivity=1.0, dark_current=1e-9)
            for dim in hidden_dims
        ])
        
        # Activation functions (electronic)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid photonic-electronic network."""
        # Electronic preprocessing
        x = self.activation(self.input_encoder(x))
        
        # Convert to optical domain (simulate E-O conversion)
        optical_field = x.to(torch.complex64) * (1 + 0j)
        
        # Photonic processing layers
        for i, (photonic_layer, photodetector) in enumerate(
            zip(self.photonic_layers, self.photodetector_arrays)
        ):
            # Optical transformation
            if isinstance(photonic_layer, pn.MZIMesh):
                optical_field = photonic_layer(optical_field)
            else:  # Microring
                # Simulate wavelength-division processing
                optical_field = photonic_layer.process_wavelength_modes(optical_field)
            
            # Add optical noise
            optical_field = self._add_optical_noise(optical_field, snr_db=40)
            
            # O-E conversion
            electrical_signal = photodetector.detect(optical_field)
            
            # Electronic activation (except last layer)
            if i < len(self.photonic_layers) - 1:
                electrical_signal = self.activation(electrical_signal)
                optical_field = electrical_signal.to(torch.complex64) * (1 + 0j)
        
        # Final electronic processing
        output = self.output_decoder(electrical_signal.real)
        
        return output
    
    def _add_optical_noise(self, field: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add realistic optical noise to the field."""
        signal_power = torch.mean(torch.abs(field) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise_real = torch.randn_like(field.real) * torch.sqrt(noise_power / 2)
        noise_imag = torch.randn_like(field.imag) * torch.sqrt(noise_power / 2)
        noise = torch.complex(noise_real, noise_imag)
        
        return field + noise


class OpticalTrainingLoop:
    """Advanced optical training loop with hardware calibration."""
    
    def __init__(self, model: HybridPhotonicMLP, 
                 hardware_chip: Optional[pn.PhotonicChip] = None):
        self.model = model
        self.hardware_chip = hardware_chip
        self.use_hardware = hardware_chip is not None
        
        # Training metrics
        self.training_history = {
            'loss': [], 'accuracy': [], 'optical_efficiency': [], 
            'calibration_error': [], 'learning_rate': []
        }
        
        # Adaptive learning rate controller
        self.lr_scheduler = None
        self.patience = 10
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Hardware calibration manager
        if self.use_hardware:
            self.calibration_manager = pn.CalibrationManager(hardware_chip)
            self.error_corrector = pn.RealTimeErrorCorrector(hardware_chip)
        
    def setup_training(self, train_loader: DataLoader, 
                      val_loader: DataLoader,
                      initial_lr: float = 0.001) -> None:
        """Setup training components."""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optical optimizer with momentum
        self.optimizer = pn.OpticalAdam(
            self.model.parameters(),
            lr=initial_lr,
            optical_loss_compensation=True,
            adaptive_phase_correction=True
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function with optical noise robustness
        self.criterion = pn.OpticalCrossEntropyLoss(
            noise_robust=True,
            thermal_compensation=True
        )
        
        print(f"Training setup complete:")
        print(f"  - Hardware-in-the-loop: {self.use_hardware}")
        print(f"  - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - Training samples: {len(train_loader.dataset)}")
        print(f"  - Validation samples: {len(val_loader.dataset)}")
    
    def calibrate_hardware(self) -> Dict[str, float]:
        """Perform comprehensive hardware calibration."""
        if not self.use_hardware:
            return {"status": "no_hardware"}
        
        print("Starting hardware calibration...")
        
        # Full system calibration
        calibration_results = self.calibration_manager.run_full_calibration(
            wavelength_sweep=(1540e-9, 1560e-9),
            power_levels=[-10, 0, 10],  # dBm
            temperature_points=[20, 25, 30],  # Celsius
            include_crosstalk=True,
            measure_thermal_response=True
        )
        
        # Apply calibration to photonic layers
        for layer in self.model.photonic_layers:
            if hasattr(layer, 'load_calibration'):
                layer.load_calibration(calibration_results)
        
        print(f"Hardware calibration complete:")
        print(f"  - Insertion loss: {calibration_results.get('insertion_loss_db', 'N/A'):.2f} dB")
        print(f"  - Phase accuracy: {calibration_results.get('phase_accuracy_deg', 'N/A'):.3f}°")
        print(f"  - Crosstalk suppression: {calibration_results.get('crosstalk_db', 'N/A'):.1f} dB")
        
        return calibration_results
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with optical feedback."""
        self.model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        optical_efficiency_sum = 0.0
        
        # Real-time calibration error tracking
        calibration_error_sum = 0.0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Hardware error correction (if available)
            if self.use_hardware:
                correction = self.error_corrector.get_realtime_correction()
                if correction['requires_update']:
                    self._apply_hardware_correction(correction)
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Optical efficiency calculation
            optical_efficiency = self._calculate_optical_efficiency()
            optical_efficiency_sum += optical_efficiency
            
            # Backward pass with optical gradient computation
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step with optical feedback
            self.optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
            
            # Real-time calibration monitoring
            if self.use_hardware:
                calib_error = self.calibration_manager.get_current_error()
                calibration_error_sum += calib_error
            
            # Progress reporting
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                      f'Loss={loss.item():.4f}, Efficiency={optical_efficiency:.3f}')
        
        # Epoch statistics
        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        avg_optical_efficiency = optical_efficiency_sum / len(self.train_loader)
        avg_calibration_error = calibration_error_sum / len(self.train_loader) if self.use_hardware else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'optical_efficiency': avg_optical_efficiency,
            'calibration_error': avg_calibration_error
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model performance."""
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, num_epochs: int) -> None:
        """Complete training loop with optical feedback."""
        print(f"\nStarting optical training for {num_epochs} epochs...")
        
        # Initial hardware calibration
        if self.use_hardware:
            self.calibrate_hardware()
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate()
            
            # Learning rate scheduling
            self.lr_scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['accuracy'].append(val_metrics['accuracy'])
            self.training_history['optical_efficiency'].append(train_metrics['optical_efficiency'])
            self.training_history['calibration_error'].append(train_metrics['calibration_error'])
            self.training_history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):')
            print(f'  Train Loss: {train_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
            print(f'  Optical Efficiency: {train_metrics["optical_efficiency"]:.3f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            if self.use_hardware:
                print(f'  Calibration Error: {train_metrics["calibration_error"]:.4f}')
            
            # Early stopping check
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_photonic_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            # Periodic recalibration (every 10 epochs)
            if self.use_hardware and (epoch + 1) % 10 == 0:
                print("Performing periodic recalibration...")
                self.calibrate_hardware()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation loss: {self.best_loss:.4f}")
    
    def _calculate_optical_efficiency(self) -> float:
        """Calculate current optical efficiency of the system."""
        total_power_in = 0.0
        total_power_out = 0.0
        
        for layer in self.model.photonic_layers:
            if hasattr(layer, 'get_power_budget'):
                power_budget = layer.get_power_budget()
                total_power_in += power_budget.get('input_power', 1.0)
                total_power_out += power_budget.get('output_power', 0.8)
        
        return total_power_out / total_power_in if total_power_in > 0 else 1.0
    
    def _apply_hardware_correction(self, correction: Dict) -> None:
        """Apply real-time hardware error correction."""
        for layer_idx, layer in enumerate(self.model.photonic_layers):
            if hasattr(layer, 'apply_correction'):
                layer_correction = correction.get(f'layer_{layer_idx}', {})
                layer.apply_correction(layer_correction)
    
    def plot_training_history(self) -> None:
        """Plot comprehensive training history."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Optical Neural Network Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(self.training_history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['accuracy'])
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].grid(True)
        
        # Optical Efficiency
        axes[0, 2].plot(self.training_history['optical_efficiency'])
        axes[0, 2].set_title('Optical Efficiency')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Efficiency')
        axes[0, 2].grid(True)
        
        # Learning Rate
        axes[1, 0].semilogy(self.training_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate (log)')
        axes[1, 0].grid(True)
        
        # Calibration Error
        if any(self.training_history['calibration_error']):
            axes[1, 1].plot(self.training_history['calibration_error'])
            axes[1, 1].set_title('Hardware Calibration Error')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Error')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Hardware', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Hardware Calibration Error')
        
        # Power Budget Analysis
        axes[1, 2].plot(np.array(self.training_history['optical_efficiency']) * 100)
        axes[1, 2].set_title('Optical Power Efficiency (%)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Efficiency (%)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('optical_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main training demonstration."""
    print("🌟 Advanced Optical Training Example")
    print("=" * 50)
    
    # Generate dataset
    print("Generating synthetic dataset...")
    inputs, labels = generate_synthetic_dataset(n_samples=2000, input_dim=784)
    
    # Create data loaders
    dataset = TensorDataset(inputs, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create hybrid photonic model
    print("Creating hybrid photonic neural network...")
    model = HybridPhotonicMLP(
        input_dim=784,
        hidden_dims=[256, 128],
        output_dim=10,
        photonic_backend="mzi"
    )
    
    # Optional: Initialize hardware connection
    hardware_chip = None  # Would connect to actual hardware: pn.PhotonicChip('visa://192.168.1.100')
    
    # Create training loop
    trainer = OpticalTrainingLoop(model, hardware_chip)
    trainer.setup_training(train_loader, val_loader, initial_lr=0.001)
    
    # Start training
    try:
        trainer.train(num_epochs=50)
        
        # Plot results
        trainer.plot_training_history()
        
        # Performance summary
        final_metrics = trainer.training_history
        print(f"\n🎯 Training Summary:")
        print(f"   Final Loss: {final_metrics['loss'][-1]:.4f}")
        print(f"   Best Accuracy: {max(final_metrics['accuracy']):.2f}%")
        print(f"   Average Optical Efficiency: {np.mean(final_metrics['optical_efficiency']):.3f}")
        
        if hardware_chip:
            print(f"   Hardware Calibration Error: {np.mean(final_metrics['calibration_error']):.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()