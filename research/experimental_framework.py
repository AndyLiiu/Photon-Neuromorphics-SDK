#!/usr/bin/env python3
"""
Experimental Research Framework
===============================

Comprehensive framework for conducting reproducible photonic neural network
research with statistical analysis, comparative studies, and publication-ready results.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
import json
import time
import pickle
import hashlib
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

import photon_neuro as pn

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')  # Updated for newer seaborn
sns.set_palette("husl")


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    model_type: str
    model_params: Dict[str, Any]
    dataset_params: Dict[str, Any]
    training_params: Dict[str, Any]
    evaluation_metrics: List[str]
    n_runs: int = 5
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def get_hash(self) -> str:
        """Get unique hash for this configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config_hash: str
    run_id: int
    metrics: Dict[str, float]
    training_history: Dict[str, List[float]]
    model_state: Optional[bytes] = None
    runtime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding model_state for size)."""
        data = asdict(self)
        data.pop('model_state', None)  # Remove for JSON serialization
        return data


class ExperimentalDataset:
    """Generate synthetic datasets for photonic neural network research."""
    
    @staticmethod
    def spiral_classification(n_samples: int = 1000, 
                            n_classes: int = 3,
                            noise: float = 0.1,
                            input_dim: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate spiral classification dataset."""
        X = torch.zeros(n_samples, input_dim)
        y = torch.zeros(n_samples, dtype=torch.long)
        
        samples_per_class = n_samples // n_classes
        
        for class_id in range(n_classes):
            start_idx = class_id * samples_per_class
            end_idx = min((class_id + 1) * samples_per_class, n_samples)
            
            t = torch.linspace(0.0, 4 * np.pi, end_idx - start_idx)
            r = torch.linspace(0.1, 1.0, end_idx - start_idx)
            
            # Add class-specific rotation
            angle_offset = class_id * 2 * np.pi / n_classes
            
            X[start_idx:end_idx, 0] = r * torch.cos(t + angle_offset)
            X[start_idx:end_idx, 1] = r * torch.sin(t + angle_offset)
            y[start_idx:end_idx] = class_id
        
        # Add noise
        X += torch.randn_like(X) * noise
        
        # Extend to higher dimensions if needed
        if input_dim > 2:
            extra_dims = torch.randn(n_samples, input_dim - 2) * 0.1
            X = torch.cat([X, extra_dims], dim=1)
        
        return X, y
    
    @staticmethod
    def optical_channel_dataset(n_samples: int = 1000,
                               n_channels: int = 4,
                               snr_db: float = 20.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate optical communication channel dataset."""
        # Generate random bit sequences
        bits = torch.randint(0, 2, (n_samples, n_channels))
        
        # Convert to optical intensities
        intensities = bits.float() * 2.0 - 1.0  # -1 for 0, +1 for 1
        
        # Add optical noise (signal-dependent)
        signal_power = torch.mean(intensities ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = torch.randn_like(intensities) * torch.sqrt(noise_power)
        received_signal = intensities + noise
        
        # Classification target: detect bit errors
        y = (torch.sign(received_signal) == torch.sign(intensities)).all(dim=1).long()
        
        return received_signal, y
    
    @staticmethod
    def photonic_interference_dataset(n_samples: int = 1000,
                                    n_modes: int = 8,
                                    coherence_length: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dataset based on photonic interference patterns."""
        # Random phase differences between modes
        phases = torch.rand(n_samples, n_modes) * 2 * np.pi
        
        # Random amplitudes
        amplitudes = torch.rand(n_samples, n_modes)
        
        # Compute interference pattern
        complex_fields = amplitudes * torch.exp(1j * phases)
        interference_power = torch.abs(torch.fft.fft(complex_fields, dim=1)) ** 2
        
        # Classification based on dominant frequency components
        dominant_freq = torch.argmax(interference_power, dim=1)
        y = dominant_freq % 3  # 3-class classification
        
        # Feature vector: amplitude and phase information
        X = torch.cat([amplitudes, phases], dim=1)
        
        return X, y


class PhotonicModelFactory:
    """Factory for creating different photonic neural network models."""
    
    @staticmethod
    def create_mzi_classifier(input_dim: int, 
                            n_classes: int,
                            mzi_size: Tuple[int, int] = (8, 8)) -> nn.Module:
        """Create MZI mesh classifier."""
        
        class MZIClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, mzi_size[0])
                self.mzi_mesh = pn.MZIMesh(size=mzi_size, topology='rectangular')
                self.photodetector = nn.Linear(mzi_size[1], 64)
                self.classifier = nn.Linear(64, n_classes)
                self.activation = nn.ReLU()
                
            def forward(self, x):
                x = self.input_projection(x)
                x = x.to(torch.complex64)  # Convert to complex
                x = self.mzi_mesh(x)
                x = torch.abs(x)  # Photodetection (intensity)
                x = self.activation(self.photodetector(x))
                return self.classifier(x)
        
        return MZIClassifier()
    
    @staticmethod
    def create_microring_classifier(input_dim: int,
                                  n_classes: int,
                                  n_rings: int = 16) -> nn.Module:
        """Create microring-based classifier."""
        
        class MicroringClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_encoder = nn.Linear(input_dim, n_rings)
                self.microring_bank = pn.MicroringArray(
                    n_rings=n_rings,
                    free_spectral_range=20e9,
                    quality_factor=10000
                )
                self.readout = nn.Linear(n_rings, n_classes)
                
            def forward(self, x):
                # Encode input as ring detunings
                ring_weights = torch.tanh(self.input_encoder(x))
                
                # Simulate ring transmission (simplified)
                wavelengths = torch.linspace(1540e-9, 1560e-9, 1).repeat(x.shape[0], 1)
                
                # Mock ring response based on input encoding
                ring_response = torch.sigmoid(ring_weights)
                
                return self.readout(ring_response)
        
        return MicroringClassifier()
    
    @staticmethod
    def create_hybrid_snn(input_dim: int,
                         n_classes: int,
                         hidden_dims: List[int] = [128, 64]) -> nn.Module:
        """Create hybrid spiking neural network."""
        
        class HybridSNN(nn.Module):
            def __init__(self):
                super().__init__()
                # Electronic preprocessing
                self.encoder = nn.Linear(input_dim, hidden_dims[0])
                
                # Photonic SNN core
                topology = [hidden_dims[0]] + hidden_dims + [n_classes]
                self.photonic_snn = pn.PhotonicSNN(
                    topology=topology,
                    neuron_model='photonic_lif',
                    synapse_type='microring',
                    timestep=1e-12
                )
                
            def forward(self, x):
                # Convert to spike trains
                encoded = torch.sigmoid(self.encoder(x))
                spikes = (torch.rand_like(encoded) < encoded).float()
                
                # Add temporal dimension
                spikes = spikes.unsqueeze(1).repeat(1, 10, 1)  # 10 timesteps
                
                # Process through photonic SNN
                output = self.photonic_snn(spikes)
                
                return output.mean(dim=1)  # Average over time
        
        return HybridSNN()


class ComparativeStudy:
    """Framework for conducting comparative studies between different approaches."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results = {}
        self.statistical_tests = {}
        
    def add_baseline(self, name: str, model_factory: Callable, 
                    model_params: Dict[str, Any]) -> None:
        """Add baseline model for comparison."""
        self.results[name] = {
            'type': 'baseline',
            'model_factory': model_factory,
            'model_params': model_params,
            'results': []
        }
    
    def add_photonic_model(self, name: str, model_factory: Callable,
                          model_params: Dict[str, Any]) -> None:
        """Add photonic model for comparison."""
        self.results[name] = {
            'type': 'photonic',
            'model_factory': model_factory,
            'model_params': model_params,
            'results': []
        }
    
    def run_comparative_study(self, 
                            dataset_generator: Callable,
                            dataset_params: Dict[str, Any],
                            training_params: Dict[str, Any],
                            n_runs: int = 10) -> Dict[str, Any]:
        """Run comparative study across all models."""
        
        # Generate dataset
        X, y = dataset_generator(**dataset_params)
        dataset = TensorDataset(X, y)
        
        # Split into train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # Run experiments for each model
        for model_name, config in self.results.items():
            print(f"Running experiments for {model_name}...")
            
            model_results = []
            
            for run in range(n_runs):
                # Set random seed for reproducibility
                torch.manual_seed(42 + run)
                np.random.seed(42 + run)
                
                # Create model
                model = config['model_factory'](**config['model_params'])
                
                # Train and evaluate
                metrics = self._train_and_evaluate(
                    model, train_dataset, test_dataset, training_params
                )
                
                model_results.append(metrics)
            
            config['results'] = model_results
        
        # Perform statistical analysis
        self._perform_statistical_analysis()
        
        # Generate report
        return self._generate_comparative_report()
    
    def _train_and_evaluate(self, model: nn.Module,
                           train_dataset: torch.utils.data.Dataset,
                           test_dataset: torch.utils.data.Dataset,
                           training_params: Dict[str, Any]) -> Dict[str, float]:
        """Train and evaluate a single model."""
        
        # Data loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=training_params.get('batch_size', 32),
                                shuffle=True)
        test_loader = DataLoader(test_dataset,
                               batch_size=training_params.get('batch_size', 32),
                               shuffle=False)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=training_params.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        start_time = time.time()
        
        for epoch in range(training_params.get('n_epochs', 50)):
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        training_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time
        }
    
    def _perform_statistical_analysis(self) -> None:
        """Perform statistical significance tests."""
        model_names = list(self.results.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # Extract accuracy scores
                scores1 = [r['accuracy'] for r in self.results[model1]['results']]
                scores2 = [r['accuracy'] for r in self.results[model2]['results']]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                     (len(scores2) - 1) * np.var(scores2, ddof=1)) /
                    (len(scores1) + len(scores2) - 2)
                )
                
                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                
                self.statistical_tests[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                }
    
    def _generate_comparative_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparative report."""
        report = {
            'experiment_name': self.experiment_name,
            'model_summaries': {},
            'statistical_analysis': self.statistical_tests,
            'recommendations': []
        }
        
        # Summarize results for each model
        for model_name, config in self.results.items():
            results = config['results']
            
            summary = {
                'mean_accuracy': np.mean([r['accuracy'] for r in results]),
                'std_accuracy': np.std([r['accuracy'] for r in results]),
                'mean_f1': np.mean([r['f1_score'] for r in results]),
                'mean_training_time': np.mean([r['training_time'] for r in results]),
                'n_runs': len(results)
            }
            
            report['model_summaries'][model_name] = summary
        
        # Generate recommendations
        best_accuracy = max(
            summary['mean_accuracy'] 
            for summary in report['model_summaries'].values()
        )
        
        best_models = [
            name for name, summary in report['model_summaries'].items()
            if summary['mean_accuracy'] == best_accuracy
        ]
        
        report['recommendations'].append(
            f"Best performing model(s): {', '.join(best_models)} "
            f"(accuracy: {best_accuracy:.4f})"
        )
        
        # Identify statistically significant improvements
        for comparison, test_result in self.statistical_tests.items():
            if test_result['significant'] and test_result['cohens_d'] > 0.5:
                report['recommendations'].append(
                    f"Significant improvement found: {comparison} "
                    f"(p={test_result['p_value']:.4f}, d={test_result['cohens_d']:.3f})"
                )
        
        return report
    
    def plot_comparative_results(self, save_path: Optional[str] = None) -> None:
        """Generate comparative visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Comparative Study: {self.experiment_name}', fontsize=16)
        
        # Prepare data
        model_names = []
        accuracy_data = []
        f1_data = []
        time_data = []
        
        for model_name, config in self.results.items():
            model_names.append(model_name)
            results = config['results']
            
            accuracy_data.append([r['accuracy'] for r in results])
            f1_data.append([r['f1_score'] for r in results])
            time_data.append([r['training_time'] for r in results])
        
        # Box plots
        axes[0, 0].boxplot(accuracy_data, labels=model_names)
        axes[0, 0].set_title('Accuracy Distribution')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].boxplot(f1_data, labels=model_names)
        axes[0, 1].set_title('F1-Score Distribution')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        mean_times = [np.mean(times) for times in time_data]
        axes[1, 0].bar(model_names, mean_times)
        axes[1, 0].set_title('Average Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Statistical significance heatmap
        n_models = len(model_names)
        p_value_matrix = np.ones((n_models, n_models))
        
        for comparison, test_result in self.statistical_tests.items():
            models = comparison.split('_vs_')
            if len(models) == 2:
                i = model_names.index(models[0])
                j = model_names.index(models[1])
                p_value_matrix[i, j] = test_result['p_value']
                p_value_matrix[j, i] = test_result['p_value']
        
        im = axes[1, 1].imshow(p_value_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
        axes[1, 1].set_xticks(range(n_models))
        axes[1, 1].set_yticks(range(n_models))
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].set_yticklabels(model_names)
        axes[1, 1].set_title('Statistical Significance (p-values)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('p-value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ResearchExperimentRunner:
    """Main class for running comprehensive research experiments."""
    
    def __init__(self, output_dir: str = "research_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_classification_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive classification benchmark."""
        self.logger.info("Starting classification benchmark...")
        
        study = ComparativeStudy("Photonic vs Classical Classification")
        
        # Add baseline models
        study.add_baseline(
            "MLP_Baseline",
            lambda input_dim, n_classes: nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes)
            ),
            {'input_dim': 16, 'n_classes': 3}
        )
        
        # Add photonic models
        study.add_photonic_model(
            "MZI_Classifier",
            PhotonicModelFactory.create_mzi_classifier,
            {'input_dim': 16, 'n_classes': 3, 'mzi_size': (8, 8)}
        )
        
        study.add_photonic_model(
            "Microring_Classifier", 
            PhotonicModelFactory.create_microring_classifier,
            {'input_dim': 16, 'n_classes': 3, 'n_rings': 16}
        )
        
        # Run study
        results = study.run_comparative_study(
            dataset_generator=ExperimentalDataset.spiral_classification,
            dataset_params={'n_samples': 2000, 'n_classes': 3, 'input_dim': 16},
            training_params={'n_epochs': 100, 'batch_size': 32, 'learning_rate': 0.001},
            n_runs=10
        )
        
        # Save results and generate plots
        self._save_results(results, "classification_benchmark")
        study.plot_comparative_results(
            save_path=self.output_dir / "classification_comparison.png"
        )
        
        return results
    
    def run_scaling_analysis(self) -> Dict[str, Any]:
        """Analyze how photonic models scale with problem size."""
        self.logger.info("Starting scaling analysis...")
        
        sizes = [4, 8, 16, 32, 64]
        scaling_results = {}
        
        for size in sizes:
            self.logger.info(f"Testing size {size}x{size}...")
            
            # Generate size-appropriate dataset
            X, y = ExperimentalDataset.spiral_classification(
                n_samples=1000, n_classes=3, input_dim=size
            )
            
            # Create and train MZI model
            model = PhotonicModelFactory.create_mzi_classifier(
                input_dim=size, n_classes=3, mzi_size=(size, size)
            )
            
            # Time training
            start_time = time.time()
            
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(50):
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            training_time = time.time() - start_time
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            
            scaling_results[size] = {
                'training_time': training_time,
                'n_parameters': n_params,
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
        
        # Plot scaling results
        self._plot_scaling_analysis(scaling_results)
        self._save_results(scaling_results, "scaling_analysis")
        
        return scaling_results
    
    def run_noise_robustness_study(self) -> Dict[str, Any]:
        """Study robustness to optical noise."""
        self.logger.info("Starting noise robustness study...")
        
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        robustness_results = {}
        
        for noise_level in noise_levels:
            self.logger.info(f"Testing noise level {noise_level}...")
            
            # Generate noisy dataset
            X, y = ExperimentalDataset.optical_channel_dataset(
                n_samples=1000, n_channels=8, snr_db=20 - 20*noise_level
            )
            
            # Test different models
            models = {
                'MZI': PhotonicModelFactory.create_mzi_classifier(8, 2, (8, 8)),
                'Microring': PhotonicModelFactory.create_microring_classifier(8, 2, 16),
                'MLP_Baseline': nn.Sequential(
                    nn.Linear(8, 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(), 
                    nn.Linear(32, 2)
                )
            }
            
            noise_results = {}
            
            for model_name, model in models.items():
                # Train and evaluate
                accuracy = self._quick_train_evaluate(model, X, y)
                noise_results[model_name] = accuracy
            
            robustness_results[noise_level] = noise_results
        
        # Plot and save results
        self._plot_noise_robustness(robustness_results)
        self._save_results(robustness_results, "noise_robustness")
        
        return robustness_results
    
    def _quick_train_evaluate(self, model: nn.Module, 
                             X: torch.Tensor, y: torch.Tensor) -> float:
        """Quick training and evaluation for noise studies."""
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(20):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def _plot_scaling_analysis(self, results: Dict[int, Dict[str, float]]) -> None:
        """Plot scaling analysis results."""
        sizes = list(results.keys())
        times = [results[size]['training_time'] for size in sizes]
        params = [results[size]['n_parameters'] for size in sizes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.loglog(sizes, times, 'bo-', label='Training Time')
        ax1.set_xlabel('Model Size')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('Training Time Scaling')
        ax1.grid(True)
        
        ax2.loglog(sizes, params, 'ro-', label='Parameters')
        ax2.set_xlabel('Model Size')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Parameter Count Scaling')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_noise_robustness(self, results: Dict[float, Dict[str, float]]) -> None:
        """Plot noise robustness results."""
        noise_levels = list(results.keys())
        model_names = list(next(iter(results.values())).keys())
        
        plt.figure(figsize=(10, 6))
        
        for model_name in model_names:
            accuracies = [results[noise][model_name] for noise in noise_levels]
            plt.plot(noise_levels, accuracies, 'o-', label=model_name, linewidth=2)
        
        plt.xlabel('Noise Level')
        plt.ylabel('Accuracy')
        plt.title('Noise Robustness Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'noise_robustness.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save results to JSON and pickle files."""
        # JSON for human readability
        with open(self.output_dir / f"{filename}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Pickle for complete data preservation
        with open(self.output_dir / f"{filename}.pkl", 'wb') as f:
            pickle.dump(results, f)
    
    def generate_research_report(self) -> str:
        """Generate comprehensive research report."""
        report = "# Photonic Neural Network Research Report\n\n"
        
        # Load all results
        result_files = list(self.output_dir.glob("*.json"))
        
        for result_file in result_files:
            with open(result_file, 'r') as f:
                results = json.load(f)
            
            experiment_name = result_file.stem.replace('_', ' ').title()
            report += f"## {experiment_name}\n\n"
            
            # Add analysis based on file content
            if 'model_summaries' in results:
                # Classification benchmark
                best_model = max(
                    results['model_summaries'].items(),
                    key=lambda x: x[1]['mean_accuracy']
                )
                
                report += f"Best performing model: **{best_model[0]}** "
                report += f"(Accuracy: {best_model[1]['mean_accuracy']:.4f} ± "
                report += f"{best_model[1]['std_accuracy']:.4f})\n\n"
                
                # Statistical significance
                significant_improvements = [
                    comp for comp, test in results['statistical_analysis'].items()
                    if test['significant'] and test['cohens_d'] > 0.5
                ]
                
                if significant_improvements:
                    report += "Statistically significant improvements found:\n"
                    for improvement in significant_improvements:
                        report += f"- {improvement}\n"
                    report += "\n"
            
            elif isinstance(results, dict) and all(isinstance(k, str) and k.replace('.', '').isdigit() for k in results.keys()):
                # Scaling analysis
                sizes = sorted([float(k) for k in results.keys()])
                report += "Scaling characteristics:\n"
                report += f"- Tested sizes: {sizes}\n"
                report += f"- Training time range: {min(results[str(int(s))]['training_time'] for s in sizes):.2f}s - "
                report += f"{max(results[str(int(s))]['training_time'] for s in sizes):.2f}s\n\n"
        
        # Save report
        with open(self.output_dir / "research_report.md", 'w') as f:
            f.write(report)
        
        return report


def main():
    """Main function to run all research experiments."""
    print("🔬 Starting Comprehensive Photonic Neural Network Research")
    print("=" * 60)
    
    runner = ResearchExperimentRunner()
    
    try:
        # Run classification benchmark
        print("\n📊 Running Classification Benchmark...")
        classification_results = runner.run_classification_benchmark()
        
        # Run scaling analysis
        print("\n📈 Running Scaling Analysis...")
        scaling_results = runner.run_scaling_analysis()
        
        # Run noise robustness study
        print("\n🔊 Running Noise Robustness Study...")
        noise_results = runner.run_noise_robustness_study()
        
        # Generate comprehensive report
        print("\n📝 Generating Research Report...")
        report = runner.generate_research_report()
        
        print(f"\n✅ Research experiments completed successfully!")
        print(f"📁 Results saved to: {runner.output_dir}")
        print(f"📊 Generated plots and statistical analysis")
        print(f"📋 Comprehensive report available")
        
        # Print summary
        print("\n🎯 Key Findings Summary:")
        print(f"  - Classification benchmark: {len(classification_results['model_summaries'])} models compared")
        print(f"  - Scaling analysis: Tested sizes from 4x4 to 64x64")
        print(f"  - Noise robustness: Evaluated across 6 noise levels")
        print(f"  - Statistical significance testing performed")
        
    except Exception as e:
        print(f"❌ Research experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()