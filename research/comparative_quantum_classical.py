#!/usr/bin/env python3
"""
Comprehensive Comparative Study: Quantum vs Classical Photonic Neural Networks
===============================================================================

Rigorous experimental framework comparing quantum-enhanced photonic neural networks
against classical baselines with statistical significance testing and reproducible results.

Based on 2025 breakthrough research in quantum photonic machine learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import our quantum implementations
from .quantum_limited_networks import (
    QuantumLimitedMLP, QuantumKernelEnhancement, QuantumEnhancedClassifier,
    SinglePhotonActivation, QuantumCoherentLayer
)
from .quantum_error_correction_ml import (
    FaultTolerantQuantumMLP, ErrorCorrectionConfig
)
from .experimental_framework import ExperimentalDataset

# Set style for publication plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")


@dataclass
class ComparisonConfig:
    """Configuration for comparative study."""
    n_runs: int = 10
    n_cv_folds: int = 5
    test_size: float = 0.2
    random_seed: int = 42
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.5  # Cohen's d threshold for meaningful difference
    statistical_tests: List[str] = None
    
    def __post_init__(self):
        if self.statistical_tests is None:
            self.statistical_tests = ['ttest', 'mannwhitney', 'wilcoxon']


@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""
    accuracy: List[float]
    precision: List[float] 
    recall: List[float]
    f1_score: List[float]
    auc_score: List[float]
    training_time: List[float]
    inference_time: List[float]
    energy_consumption: List[float]
    quantum_advantage_score: float = 0.0
    
    def get_mean_std(self, metric: str) -> Tuple[float, float]:
        """Get mean and standard deviation for a metric."""
        values = getattr(self, metric)
        return np.mean(values), np.std(values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class QuantumAdvantageMetrics:
    """Comprehensive metrics for evaluating quantum advantage."""
    
    @staticmethod
    def energy_efficiency_ratio(quantum_energy: float, classical_energy: float) -> float:
        """Compute energy efficiency ratio (higher is better for quantum)."""
        return classical_energy / max(quantum_energy, 1e-10)
    
    @staticmethod
    def computational_speedup(quantum_time: float, classical_time: float) -> float:
        """Compute computational speedup (higher is better for quantum)."""
        return classical_time / max(quantum_time, 1e-10)
    
    @staticmethod
    def accuracy_advantage(quantum_acc: float, classical_acc: float) -> float:
        """Compute accuracy advantage."""
        return quantum_acc - classical_acc
    
    @staticmethod
    def quantum_volume_score(n_qubits: int, circuit_depth: int, fidelity: float) -> float:
        """Compute quantum volume score."""
        return min(n_qubits, circuit_depth) * fidelity
    
    @staticmethod
    def small_dataset_advantage(quantum_perf: List[float], classical_perf: List[float],
                               dataset_sizes: List[int]) -> float:
        """Measure advantage on small datasets (key quantum ML advantage)."""
        advantages = []
        for q_perf, c_perf, size in zip(quantum_perf, classical_perf, dataset_sizes):
            if size < 1000:  # Small dataset threshold
                advantage = (q_perf - c_perf) / max(c_perf, 1e-10)
                advantages.append(advantage)
        return np.mean(advantages) if advantages else 0.0


class StatisticalSignificanceTester:
    """Statistical testing suite for quantum vs classical comparisons."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def t_test_comparison(self, quantum_scores: List[float], 
                         classical_scores: List[float]) -> Dict[str, float]:
        """Perform t-test comparison."""
        statistic, p_value = ttest_ind(quantum_scores, classical_scores)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(quantum_scores) - 1) * np.var(quantum_scores, ddof=1) +
             (len(classical_scores) - 1) * np.var(classical_scores, ddof=1)) /
            (len(quantum_scores) + len(classical_scores) - 2)
        )
        
        cohens_d = (np.mean(quantum_scores) - np.mean(classical_scores)) / max(pooled_std, 1e-10)
        
        return {
            'test_statistic': statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < self.alpha,
            'effect_size': self._interpret_effect_size(abs(cohens_d))
        }
    
    def mann_whitney_test(self, quantum_scores: List[float], 
                         classical_scores: List[float]) -> Dict[str, float]:
        """Perform Mann-Whitney U test (non-parametric)."""
        statistic, p_value = mannwhitneyu(quantum_scores, classical_scores, alternative='two-sided')
        
        # Calculate rank-biserial correlation as effect size
        n1, n2 = len(quantum_scores), len(classical_scores)
        r = 1 - (2 * statistic) / (n1 * n2)
        
        return {
            'test_statistic': statistic,
            'p_value': p_value,
            'rank_biserial_r': r,
            'significant': p_value < self.alpha,
            'effect_size': self._interpret_effect_size(abs(r))
        }
    
    def wilcoxon_test(self, quantum_scores: List[float], 
                     classical_scores: List[float]) -> Dict[str, float]:
        """Perform Wilcoxon signed-rank test (paired)."""
        if len(quantum_scores) != len(classical_scores):
            raise ValueError("Wilcoxon test requires equal-length paired samples")
        
        statistic, p_value = wilcoxon(quantum_scores, classical_scores)
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(quantum_scores)
        z_score = abs(stats.norm.ppf(p_value / 2))  # Approximate Z from p-value
        r = z_score / np.sqrt(n)
        
        return {
            'test_statistic': statistic,
            'p_value': p_value,
            'effect_size_r': r,
            'significant': p_value < self.alpha,
            'effect_size': self._interpret_effect_size(r)
        }
    
    def comprehensive_comparison(self, quantum_scores: List[float], 
                               classical_scores: List[float]) -> Dict[str, Any]:
        """Perform comprehensive statistical comparison."""
        results = {}
        
        # Descriptive statistics
        results['quantum_stats'] = {
            'mean': np.mean(quantum_scores),
            'std': np.std(quantum_scores),
            'median': np.median(quantum_scores),
            'iqr': np.percentile(quantum_scores, 75) - np.percentile(quantum_scores, 25)
        }
        
        results['classical_stats'] = {
            'mean': np.mean(classical_scores),
            'std': np.std(classical_scores), 
            'median': np.median(classical_scores),
            'iqr': np.percentile(classical_scores, 75) - np.percentile(classical_scores, 25)
        }
        
        # Statistical tests
        results['t_test'] = self.t_test_comparison(quantum_scores, classical_scores)
        results['mann_whitney'] = self.mann_whitney_test(quantum_scores, classical_scores)
        
        if len(quantum_scores) == len(classical_scores):
            results['wilcoxon'] = self.wilcoxon_test(quantum_scores, classical_scores)
        
        # Confidence intervals
        results['quantum_ci'] = self._confidence_interval(quantum_scores)
        results['classical_ci'] = self._confidence_interval(classical_scores)
        
        # Overall assessment
        results['quantum_advantage'] = {
            'mean_difference': np.mean(quantum_scores) - np.mean(classical_scores),
            'relative_improvement': (np.mean(quantum_scores) - np.mean(classical_scores)) / max(np.mean(classical_scores), 1e-10),
            'consistently_better': np.mean(quantum_scores) > np.mean(classical_scores),
            'statistical_significance': results['t_test']['significant']
        }
        
        return results
    
    def _confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval."""
        mean = np.mean(scores)
        sem = stats.sem(scores)  # Standard error of mean
        h = sem * stats.t.ppf((1 + self.confidence_level) / 2, len(scores) - 1)
        return mean - h, mean + h
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"


class QuantumClassicalComparison:
    """Main framework for quantum vs classical ML comparisons."""
    
    def __init__(self, config: ComparisonConfig, output_dir: str = "comparison_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.statistical_tester = StatisticalSignificanceTester(config.confidence_level)
        self.results = {}
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def run_comprehensive_comparison(self, datasets: Dict[str, Dataset]) -> Dict[str, Any]:
        """Run comprehensive comparison across multiple datasets."""
        print("🔬 Starting Comprehensive Quantum vs Classical Comparison")
        print("=" * 65)
        
        all_results = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\n📊 Analyzing Dataset: {dataset_name}")
            print("-" * 40)
            
            dataset_results = self._compare_on_dataset(dataset, dataset_name)
            all_results[dataset_name] = dataset_results
            
            # Save intermediate results
            self._save_results(dataset_results, f"{dataset_name}_comparison")
        
        # Generate summary report
        summary_report = self._generate_summary_report(all_results)
        all_results['summary'] = summary_report
        
        # Save complete results
        self._save_results(all_results, "complete_comparison")
        
        return all_results
    
    def _compare_on_dataset(self, dataset: Dataset, dataset_name: str) -> Dict[str, Any]:
        """Compare quantum vs classical models on a single dataset."""
        
        # Get dataset properties
        sample_data, sample_label = dataset[0]
        input_dim = len(sample_data) if isinstance(sample_data, torch.Tensor) else sample_data.shape[-1]
        n_classes = len(torch.unique(torch.stack([dataset[i][1] for i in range(min(100, len(dataset)))])))
        
        print(f"  Dataset properties: {len(dataset)} samples, {input_dim}D input, {n_classes} classes")
        
        # Define model architectures
        models = self._create_model_suite(input_dim, n_classes)
        
        results = {}
        
        for model_name, model_factory in models.items():
            print(f"\n  🧠 Testing {model_name}...")
            
            model_results = self._evaluate_model_performance(
                model_factory, dataset, model_name
            )
            
            results[model_name] = model_results
            
            # Print quick summary
            acc_mean, acc_std = model_results.get_mean_std('accuracy')
            time_mean, _ = model_results.get_mean_std('training_time')
            print(f"    Accuracy: {acc_mean:.4f} ± {acc_std:.4f}, Time: {time_mean:.2f}s")
        
        # Statistical comparisons
        print(f"\n  📈 Statistical Analysis...")
        statistical_results = self._perform_statistical_analysis(results)
        
        return {
            'model_results': results,
            'statistical_analysis': statistical_results,
            'dataset_properties': {
                'n_samples': len(dataset),
                'input_dim': input_dim,
                'n_classes': n_classes,
                'name': dataset_name
            }
        }
    
    def _create_model_suite(self, input_dim: int, n_classes: int) -> Dict[str, Callable]:
        """Create suite of quantum and classical models for comparison."""
        
        hidden_dim = max(32, min(128, input_dim * 2))
        
        models = {
            # Classical Baselines
            'MLP_Classical': lambda: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, n_classes)
            ),
            
            'CNN_Classical': lambda: self._create_cnn_classifier(input_dim, n_classes),
            
            'ResNet_Classical': lambda: self._create_resnet_classifier(input_dim, n_classes),
            
            # Quantum Models
            'Quantum_Limited_MLP': lambda: QuantumLimitedMLP(
                input_dim=input_dim,
                hidden_dims=[hidden_dim, hidden_dim // 2],
                output_dim=n_classes,
                quantum_efficiency=0.9,
                use_quantum_interference=True
            ),
            
            'Quantum_Kernel_Enhanced': lambda: QuantumEnhancedClassifier(
                feature_dim=input_dim,
                n_classes=n_classes,
                n_qubits=min(6, max(3, input_dim.bit_length())),
                kernel_layers=3
            ),
            
            'Fault_Tolerant_Quantum': lambda: FaultTolerantQuantumMLP(
                input_dim=input_dim,
                hidden_dims=[hidden_dim // 2],
                output_dim=n_classes,
                code_distance=3,
                config=ErrorCorrectionConfig(
                    physical_error_rate=1e-3,
                    correction_algorithm='ml_decoder'
                )
            ),
            
            # Hybrid Models
            'Hybrid_Quantum_Classical': lambda: self._create_hybrid_model(input_dim, n_classes),
        }
        
        return models
    
    def _create_cnn_classifier(self, input_dim: int, n_classes: int) -> nn.Module:
        """Create CNN classifier (adapted for 1D input)."""
        
        class CNN1DClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                # Reshape 1D input to pseudo-2D for convolution
                self.input_reshape = nn.Linear(input_dim, 64)  # Map to reasonable size
                
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.AdaptiveAvgPool1d(8)
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(32 * 8, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, n_classes)
                )
            
            def forward(self, x):
                x = self.input_reshape(x)
                x = x.unsqueeze(1)  # Add channel dimension
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return CNN1DClassifier()
    
    def _create_resnet_classifier(self, input_dim: int, n_classes: int) -> nn.Module:
        """Create ResNet-style classifier."""
        
        class ResNetBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim)
                )
                
            def forward(self, x):
                return F.relu(x + self.block(x))
        
        class ResNetClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                hidden_dim = max(32, min(128, input_dim * 2))
                
                self.input_layer = nn.Linear(input_dim, hidden_dim)
                self.blocks = nn.ModuleList([
                    ResNetBlock(hidden_dim) for _ in range(3)
                ])
                self.output_layer = nn.Linear(hidden_dim, n_classes)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                x = F.relu(self.input_layer(x))
                for block in self.blocks:
                    x = self.dropout(block(x))
                return self.output_layer(x)
        
        return ResNetClassifier()
    
    def _create_hybrid_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Create hybrid quantum-classical model."""
        
        class HybridQuantumClassical(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Classical preprocessing
                self.classical_encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16)
                )
                
                # Quantum processing core
                self.quantum_core = QuantumCoherentLayer(
                    in_features=16,
                    out_features=8,
                    use_quantum_interference=True
                )
                
                # Quantum activation
                self.quantum_activation = SinglePhotonActivation(
                    temperature=1.0,
                    quantum_efficiency=0.9
                )
                
                # Classical postprocessing
                self.classical_decoder = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, n_classes)
                )
                
            def forward(self, x):
                # Classical → Quantum → Classical pipeline
                x = self.classical_encoder(x)
                x = self.quantum_core(x)
                x = self.quantum_activation(x)
                x = self.classical_decoder(x)
                return x
        
        return HybridQuantumClassical()
    
    def _evaluate_model_performance(self, model_factory: Callable, 
                                  dataset: Dataset, model_name: str) -> ModelPerformance:
        """Evaluate model performance with multiple runs."""
        
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        auc_scores = []
        training_times = []
        inference_times = []
        energy_consumptions = []
        
        for run in range(self.config.n_runs):
            # Set seed for reproducibility
            torch.manual_seed(self.config.random_seed + run)
            np.random.seed(self.config.random_seed + run)
            
            # Create fresh model instance
            model = model_factory()
            
            # Evaluate single run
            metrics = self._single_run_evaluation(model, dataset)
            
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
            auc_scores.append(metrics['auc_score'])
            training_times.append(metrics['training_time'])
            inference_times.append(metrics['inference_time'])
            energy_consumptions.append(metrics['energy_consumption'])
        
        # Calculate quantum advantage score
        quantum_advantage_score = self._calculate_quantum_advantage_score(
            model_name, accuracies, energy_consumptions
        )
        
        return ModelPerformance(
            accuracy=accuracies,
            precision=precisions,
            recall=recalls,
            f1_score=f1_scores,
            auc_score=auc_scores,
            training_time=training_times,
            inference_time=inference_times,
            energy_consumption=energy_consumptions,
            quantum_advantage_score=quantum_advantage_score
        )
    
    def _single_run_evaluation(self, model: nn.Module, dataset: Dataset) -> Dict[str, float]:
        """Single run evaluation of a model."""
        
        # Split dataset
        train_size = int((1 - self.config.test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        model.train()
        
        start_time = time.time()
        energy_consumption = 0.0
        
        # Training loop
        for epoch in range(50):  # Fixed number of epochs for fair comparison
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Estimate energy consumption
                if hasattr(model, 'get_quantum_efficiency'):
                    quantum_eff = model.get_quantum_efficiency()
                    energy_consumption += batch_x.shape[0] * (2.0 - quantum_eff)
                else:
                    energy_consumption += batch_x.shape[0] * 2.0
            
            scheduler.step(epoch_loss)
        
        training_time = time.time() - start_time
        
        # Evaluation
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        start_inference = time.time()
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        inference_time = time.time() - start_inference
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
        
        # AUC score (for multi-class)
        try:
            if len(set(all_targets)) > 2:
                auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(all_targets, [p[1] for p in all_probabilities])
        except:
            auc = 0.5  # Default for cases where AUC can't be calculated
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'training_time': training_time,
            'inference_time': inference_time,
            'energy_consumption': energy_consumption / max(len(train_dataset), 1)
        }
    
    def _calculate_quantum_advantage_score(self, model_name: str, 
                                         accuracies: List[float],
                                         energies: List[float]) -> float:
        """Calculate quantum advantage score."""
        if 'Quantum' not in model_name and 'quantum' not in model_name.lower():
            return 0.0
        
        # Composite score based on accuracy and energy efficiency
        mean_accuracy = np.mean(accuracies)
        mean_energy = np.mean(energies)
        
        # Higher accuracy and lower energy consumption are better
        accuracy_score = mean_accuracy
        efficiency_score = 1.0 / max(mean_energy, 1e-6)
        
        # Quantum advantage score (0-1 scale)
        quantum_score = (accuracy_score + 0.1 * efficiency_score) / 1.1
        
        return min(quantum_score, 1.0)
    
    def _perform_statistical_analysis(self, results: Dict[str, ModelPerformance]) -> Dict[str, Any]:
        """Perform statistical analysis comparing all models."""
        
        analysis = {}
        model_names = list(results.keys())
        
        # Identify quantum vs classical models
        quantum_models = [name for name in model_names 
                         if 'quantum' in name.lower() or 'Quantum' in name]
        classical_models = [name for name in model_names 
                          if name not in quantum_models]
        
        # Pairwise comparisons
        pairwise_comparisons = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                comparison_result = self.statistical_tester.comprehensive_comparison(
                    results[model1].accuracy,
                    results[model2].accuracy
                )
                
                pairwise_comparisons[comparison_key] = comparison_result
        
        analysis['pairwise_comparisons'] = pairwise_comparisons
        
        # Quantum vs Classical group analysis
        if quantum_models and classical_models:
            # Aggregate quantum performance
            all_quantum_accuracies = []
            for q_model in quantum_models:
                all_quantum_accuracies.extend(results[q_model].accuracy)
            
            # Aggregate classical performance
            all_classical_accuracies = []
            for c_model in classical_models:
                all_classical_accuracies.extend(results[c_model].accuracy)
            
            quantum_vs_classical = self.statistical_tester.comprehensive_comparison(
                all_quantum_accuracies, all_classical_accuracies
            )
            
            analysis['quantum_vs_classical'] = quantum_vs_classical
        
        # Performance rankings
        rankings = self._compute_performance_rankings(results)
        analysis['performance_rankings'] = rankings
        
        # Effect size analysis
        effect_sizes = self._analyze_effect_sizes(pairwise_comparisons)
        analysis['effect_size_analysis'] = effect_sizes
        
        return analysis
    
    def _compute_performance_rankings(self, results: Dict[str, ModelPerformance]) -> Dict[str, Any]:
        """Compute performance rankings across different metrics."""
        
        rankings = {}
        metrics = ['accuracy', 'f1_score', 'auc_score']
        
        for metric in metrics:
            model_scores = []
            for model_name, performance in results.items():
                mean_score = np.mean(getattr(performance, metric))
                model_scores.append((model_name, mean_score))
            
            # Sort by score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            rankings[metric] = {
                'ranking': [(name, score, rank+1) for rank, (name, score) in enumerate(model_scores)],
                'best_model': model_scores[0][0],
                'best_score': model_scores[0][1]
            }
        
        return rankings
    
    def _analyze_effect_sizes(self, pairwise_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effect sizes across comparisons."""
        
        large_effects = []
        medium_effects = []
        small_effects = []
        
        for comparison, results in pairwise_comparisons.items():
            if 't_test' in results:
                cohens_d = abs(results['t_test']['cohens_d'])
                effect_category = results['t_test']['effect_size']
                
                comparison_info = {
                    'comparison': comparison,
                    'cohens_d': cohens_d,
                    'category': effect_category,
                    'significant': results['t_test']['significant']
                }
                
                if effect_category == 'large':
                    large_effects.append(comparison_info)
                elif effect_category == 'medium':
                    medium_effects.append(comparison_info)
                elif effect_category == 'small':
                    small_effects.append(comparison_info)
        
        return {
            'large_effects': large_effects,
            'medium_effects': medium_effects, 
            'small_effects': small_effects,
            'summary': {
                'n_large_effects': len(large_effects),
                'n_medium_effects': len(medium_effects),
                'n_small_effects': len(small_effects)
            }
        }
    
    def _generate_summary_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        
        summary = {
            'datasets_analyzed': list(all_results.keys()),
            'models_compared': [],
            'overall_quantum_advantage': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Extract model names
        if all_results:
            first_dataset = next(iter(all_results.values()))
            summary['models_compared'] = list(first_dataset['model_results'].keys())
        
        # Aggregate quantum advantage analysis
        quantum_wins = 0
        total_comparisons = 0
        significant_improvements = []
        
        for dataset_name, dataset_results in all_results.items():
            if 'statistical_analysis' in dataset_results:
                stats = dataset_results['statistical_analysis']
                
                if 'quantum_vs_classical' in stats:
                    qvc = stats['quantum_vs_classical']
                    total_comparisons += 1
                    
                    if qvc['quantum_advantage']['consistently_better']:
                        quantum_wins += 1
                    
                    if qvc['quantum_advantage']['statistical_significance']:
                        significant_improvements.append({
                            'dataset': dataset_name,
                            'improvement': qvc['quantum_advantage']['relative_improvement'],
                            'p_value': qvc['t_test']['p_value']
                        })
        
        summary['overall_quantum_advantage'] = {
            'quantum_wins': quantum_wins,
            'total_comparisons': total_comparisons,
            'win_rate': quantum_wins / max(total_comparisons, 1),
            'significant_improvements': significant_improvements
        }
        
        # Generate key findings
        if quantum_wins > total_comparisons / 2:
            summary['key_findings'].append("✅ Quantum models show consistent advantages across datasets")
        else:
            summary['key_findings'].append("⚠️ Mixed results for quantum advantage")
        
        if len(significant_improvements) > 0:
            summary['key_findings'].append(f"🔬 {len(significant_improvements)} statistically significant quantum improvements found")
        
        # Generate recommendations
        if summary['overall_quantum_advantage']['win_rate'] > 0.6:
            summary['recommendations'].append("Consider quantum-enhanced models for improved performance")
        
        summary['recommendations'].append("Further investigation needed for domain-specific quantum advantages")
        
        return summary
    
    def _save_results(self, results: Any, filename: str):
        """Save results to JSON and pickle files."""
        # JSON for human readability
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Pickle for complete data preservation  
        pickle_path = self.output_dir / f"{filename}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
    
    def generate_comparison_plots(self, results: Dict[str, Any]):
        """Generate comprehensive comparison plots."""
        
        print("📊 Generating Comparison Plots...")
        
        # Plot 1: Performance comparison across datasets
        self._plot_performance_comparison(results)
        
        # Plot 2: Statistical significance heatmap
        self._plot_significance_heatmap(results)
        
        # Plot 3: Effect size analysis
        self._plot_effect_sizes(results)
        
        # Plot 4: Quantum advantage visualization
        self._plot_quantum_advantage(results)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_performance_comparison(self, results: Dict[str, Any]):
        """Plot performance comparison across datasets."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Quantum vs Classical Performance Comparison', fontsize=16)
        
        metrics = ['accuracy', 'f1_score', 'training_time', 'energy_consumption']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            dataset_names = []
            model_data = {}
            
            for dataset_name, dataset_results in results.items():
                if dataset_name == 'summary':
                    continue
                    
                dataset_names.append(dataset_name)
                
                for model_name, model_results in dataset_results['model_results'].items():
                    if model_name not in model_data:
                        model_data[model_name] = []
                    
                    values = getattr(model_results, metric)
                    model_data[model_name].append(np.mean(values))
            
            # Create grouped bar plot
            x = np.arange(len(dataset_names))
            width = 0.1
            
            quantum_models = [name for name in model_data.keys() if 'quantum' in name.lower()]
            classical_models = [name for name in model_data.keys() if name not in quantum_models]
            
            colors = sns.color_palette("Set2", len(model_data))
            
            for i, (model_name, values) in enumerate(model_data.items()):
                offset = (i - len(model_data) / 2) * width
                color = 'red' if model_name in quantum_models else 'blue'
                ax.bar(x + offset, values, width, label=model_name, alpha=0.8, color=colors[i])
            
            ax.set_xlabel('Datasets')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(dataset_names, rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    
    def _plot_significance_heatmap(self, results: Dict[str, Any]):
        """Plot statistical significance heatmap."""
        
        # Collect p-values across all datasets
        all_comparisons = {}
        
        for dataset_name, dataset_results in results.items():
            if dataset_name == 'summary':
                continue
                
            if 'statistical_analysis' in dataset_results:
                pairwise = dataset_results['statistical_analysis']['pairwise_comparisons']
                
                for comparison, stats in pairwise.items():
                    if comparison not in all_comparisons:
                        all_comparisons[comparison] = []
                    
                    p_value = stats['t_test']['p_value']
                    all_comparisons[comparison].append(p_value)
        
        # Create heatmap data
        comparison_names = list(all_comparisons.keys())
        mean_p_values = [np.mean(p_vals) for p_vals in all_comparisons.values()]
        
        # Create matrix for heatmap
        model_names = set()
        for comp in comparison_names:
            models = comp.split('_vs_')
            model_names.update(models)
        
        model_names = sorted(list(model_names))
        n_models = len(model_names)
        p_matrix = np.ones((n_models, n_models))
        
        for comparison, mean_p in zip(comparison_names, mean_p_values):
            models = comparison.split('_vs_')
            if len(models) == 2:
                i = model_names.index(models[0])
                j = model_names.index(models[1])
                p_matrix[i, j] = mean_p
                p_matrix[j, i] = mean_p
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(p_matrix, 
                   xticklabels=model_names,
                   yticklabels=model_names,
                   annot=True,
                   cmap='RdYlBu_r',
                   vmin=0, vmax=0.1,
                   fmt='.4f')
        
        plt.title('Statistical Significance Heatmap (p-values)')
        plt.xlabel('Models')
        plt.ylabel('Models') 
        plt.tight_layout()
        plt.savefig(self.output_dir / 'significance_heatmap.png', dpi=300, bbox_inches='tight')
    
    def _plot_effect_sizes(self, results: Dict[str, Any]):
        """Plot effect size analysis."""
        
        effect_sizes = []
        comparison_names = []
        significance = []
        
        for dataset_name, dataset_results in results.items():
            if dataset_name == 'summary':
                continue
            
            if 'statistical_analysis' in dataset_results:
                pairwise = dataset_results['statistical_analysis']['pairwise_comparisons']
                
                for comparison, stats in pairwise.items():
                    effect_sizes.append(abs(stats['t_test']['cohens_d']))
                    comparison_names.append(f"{dataset_name}:{comparison}")
                    significance.append(stats['t_test']['significant'])
        
        # Create scatter plot
        plt.figure(figsize=(14, 8))
        
        colors = ['red' if sig else 'gray' for sig in significance]
        
        plt.scatter(range(len(effect_sizes)), effect_sizes, c=colors, alpha=0.7)
        
        # Add effect size thresholds
        plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small Effect')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect')
        
        plt.xlabel('Comparisons')
        plt.ylabel("Cohen's d (Effect Size)")
        plt.title('Effect Size Analysis Across All Comparisons')
        plt.xticks(range(len(comparison_names)), comparison_names, rotation=90)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
    
    def _plot_quantum_advantage(self, results: Dict[str, Any]):
        """Plot quantum advantage visualization."""
        
        if 'summary' not in results:
            return
        
        summary = results['summary']
        
        # Create quantum advantage summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Win rate
        win_rate = summary['overall_quantum_advantage']['win_rate']
        
        ax1.bar(['Quantum Advantage'], [win_rate], color='blue', alpha=0.7)
        ax1.axhline(y=0.5, color='red', linestyle='--', label='Chance Level')
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Overall Quantum Advantage Win Rate')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Significant improvements
        improvements = summary['overall_quantum_advantage']['significant_improvements']
        
        if improvements:
            datasets = [imp['dataset'] for imp in improvements]
            improvement_values = [imp['improvement'] * 100 for imp in improvements]  # Convert to percentage
            
            ax2.bar(datasets, improvement_values, color='green', alpha=0.7)
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Statistically Significant Quantum Improvements')
            ax2.set_xticklabels(datasets, rotation=45)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Statistically\nSignificant Improvements', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Statistically Significant Quantum Improvements')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quantum_advantage.png', dpi=300, bbox_inches='tight')


def main():
    """Run comprehensive quantum vs classical comparison."""
    print("🚀 Comprehensive Quantum vs Classical ML Comparison Framework")
    print("=" * 70)
    
    # Configuration
    config = ComparisonConfig(
        n_runs=8,  # Reduced for demo
        confidence_level=0.95,
        random_seed=42
    )
    
    # Create test datasets
    print("📊 Generating Test Datasets...")
    
    datasets = {
        'Spiral_Classification': TensorDataset(*ExperimentalDataset.spiral_classification(
            n_samples=800, n_classes=3, input_dim=8
        )),
        'Optical_Channel': TensorDataset(*ExperimentalDataset.optical_channel_dataset(
            n_samples=800, n_channels=6, snr_db=15
        )),
        'Photonic_Interference': TensorDataset(*ExperimentalDataset.photonic_interference_dataset(
            n_samples=800, n_modes=6
        ))
    }
    
    # Initialize comparison framework
    comparison = QuantumClassicalComparison(config, output_dir="comprehensive_comparison_results")
    
    # Run comprehensive comparison
    print("\n🔬 Running Comprehensive Comparison...")
    results = comparison.run_comprehensive_comparison(datasets)
    
    # Generate plots
    comparison.generate_comparison_plots(results)
    
    # Print summary results
    print("\n📈 COMPARISON RESULTS SUMMARY")
    print("=" * 40)
    
    summary = results.get('summary', {})
    
    if 'overall_quantum_advantage' in summary:
        qa = summary['overall_quantum_advantage']
        print(f"🏆 Quantum Win Rate: {qa['win_rate']:.1%}")
        print(f"🔬 Significant Improvements: {len(qa['significant_improvements'])}")
        
        if qa['significant_improvements']:
            print("\n✅ Statistically Significant Quantum Advantages:")
            for imp in qa['significant_improvements']:
                print(f"  • {imp['dataset']}: {imp['improvement']:.1%} improvement (p={imp['p_value']:.4f})")
    
    if 'key_findings' in summary:
        print("\n🎯 Key Findings:")
        for finding in summary['key_findings']:
            print(f"  {finding}")
    
    if 'recommendations' in summary:
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
    
    print(f"\n📁 Complete results saved to: {comparison.output_dir}")
    print("📊 Plots and statistical analysis generated")
    print("🎉 Comprehensive comparison completed successfully!")


if __name__ == "__main__":
    main()