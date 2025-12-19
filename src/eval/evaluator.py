"""Evaluation metrics and utilities for TSP optimization."""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils.core import compute_tour_length, compute_optimality_gap


class TSPEvaluator:
    """Evaluator for TSP optimization results."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def evaluate_tour(
        self,
        coords: torch.Tensor,
        predicted_tour: List[int],
        optimal_tour: Optional[List[int]] = None,
        method_name: str = "unknown"
    ) -> Dict[str, float]:
        """Evaluate a single TSP tour.
        
        Args:
            coords: Node coordinates [n_nodes, 2]
            predicted_tour: Predicted tour
            optimal_tour: Optimal tour (if available)
            method_name: Name of the method
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Compute tour length
        predicted_length = compute_tour_length(coords, predicted_tour)
        
        metrics = {
            'tour_length': predicted_length,
            'method': method_name
        }
        
        # Compute optimality gap if optimal tour is provided
        if optimal_tour is not None:
            optimal_length = compute_tour_length(coords, optimal_tour)
            optimality_gap = compute_optimality_gap(predicted_length, optimal_length)
            metrics.update({
                'optimal_length': optimal_length,
                'optimality_gap': optimality_gap
            })
        
        # Store results
        if method_name not in self.results:
            self.results[method_name] = []
        self.results[method_name].append(metrics)
        
        return metrics
    
    def evaluate_batch(
        self,
        coords_batch: List[torch.Tensor],
        predicted_tours: List[List[int]],
        optimal_tours: Optional[List[List[int]]] = None,
        method_name: str = "unknown"
    ) -> Dict[str, float]:
        """Evaluate a batch of TSP tours.
        
        Args:
            coords_batch: List of coordinate tensors
            predicted_tours: List of predicted tours
            optimal_tours: List of optimal tours (if available)
            method_name: Name of the method
            
        Returns:
            Dictionary of aggregated metrics
        """
        batch_metrics = []
        
        for i, (coords, pred_tour) in enumerate(zip(coords_batch, predicted_tours)):
            opt_tour = optimal_tours[i] if optimal_tours else None
            metrics = self.evaluate_tour(coords, pred_tour, opt_tour, method_name)
            batch_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(batch_metrics)
        aggregated['method'] = method_name
        aggregated['n_instances'] = len(batch_metrics)
        
        return aggregated
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple instances.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            if key == 'method':
                continue
            
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def compare_methods(self) -> Dict[str, Dict[str, float]]:
        """Compare all evaluated methods.
        
        Returns:
            Dictionary of method comparisons
        """
        comparison = {}
        
        for method_name, method_results in self.results.items():
            aggregated = self._aggregate_metrics(method_results)
            comparison[method_name] = aggregated
        
        return comparison
    
    def plot_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot comparison of methods.
        
        Args:
            save_path: Path to save plot
        """
        comparison = self.compare_methods()
        
        if not comparison:
            print("No results to plot")
            return
        
        # Extract data for plotting
        methods = list(comparison.keys())
        tour_lengths = [comparison[method]['tour_length_mean'] for method in methods]
        tour_lengths_std = [comparison[method]['tour_length_std'] for method in methods]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Tour length comparison
        bars = ax1.bar(methods, tour_lengths, yerr=tour_lengths_std, capsize=5)
        ax1.set_title('Average Tour Length Comparison')
        ax1.set_ylabel('Tour Length')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, length in zip(bars, tour_lengths):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + tour_lengths_std[methods.index(bar.get_label())],
                    f'{length:.2f}', ha='center', va='bottom')
        
        # Optimality gap comparison (if available)
        if any('optimality_gap_mean' in comparison[method] for method in methods):
            optimality_gaps = []
            optimality_gaps_std = []
            
            for method in methods:
                if 'optimality_gap_mean' in comparison[method]:
                    optimality_gaps.append(comparison[method]['optimality_gap_mean'])
                    optimality_gaps_std.append(comparison[method]['optimality_gap_std'])
                else:
                    optimality_gaps.append(0)
                    optimality_gaps_std.append(0)
            
            bars = ax2.bar(methods, optimality_gaps, yerr=optimality_gaps_std, capsize=5)
            ax2.set_title('Optimality Gap Comparison')
            ax2.set_ylabel('Optimality Gap (%)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, gap in zip(bars, optimality_gaps):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + optimality_gaps_std[methods.index(bar.get_label())],
                        f'{gap:.1f}%', ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No optimality gap data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Optimality Gap Comparison')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate evaluation report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Report string
        """
        comparison = self.compare_methods()
        
        report = "TSP Optimization Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        for method_name, metrics in comparison.items():
            report += f"Method: {method_name}\n"
            report += "-" * 20 + "\n"
            report += f"Number of instances: {metrics.get('n_instances', 'N/A')}\n"
            report += f"Average tour length: {metrics.get('tour_length_mean', 'N/A'):.4f} ± {metrics.get('tour_length_std', 'N/A'):.4f}\n"
            report += f"Min tour length: {metrics.get('tour_length_min', 'N/A'):.4f}\n"
            report += f"Max tour length: {metrics.get('tour_length_max', 'N/A'):.4f}\n"
            
            if 'optimality_gap_mean' in metrics:
                report += f"Average optimality gap: {metrics['optimality_gap_mean']:.2f}% ± {metrics['optimality_gap_std']:.2f}%\n"
                report += f"Min optimality gap: {metrics['optimality_gap_min']:.2f}%\n"
                report += f"Max optimality gap: {metrics['optimality_gap_max']:.2f}%\n"
            
            report += "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results = {}


class ConvergenceAnalyzer:
    """Analyze training convergence."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.training_history = {}
    
    def add_training_history(
        self,
        method_name: str,
        train_losses: List[float],
        val_losses: List[float]
    ) -> None:
        """Add training history for a method.
        
        Args:
            method_name: Name of the method
            train_losses: Training losses
            val_losses: Validation losses
        """
        self.training_history[method_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """Plot training convergence.
        
        Args:
            save_path: Path to save plot
        """
        if not self.training_history:
            print("No training history to plot")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Training losses
        plt.subplot(1, 2, 1)
        for method_name, history in self.training_history.items():
            plt.plot(history['train_losses'], label=f'{method_name} (train)', alpha=0.7)
        plt.title('Training Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Validation losses
        plt.subplot(1, 2, 2)
        for method_name, history in self.training_history.items():
            plt.plot(history['val_losses'], label=f'{method_name} (val)', alpha=0.7)
        plt.title('Validation Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def analyze_convergence(self) -> Dict[str, Dict[str, float]]:
        """Analyze convergence characteristics.
        
        Returns:
            Dictionary of convergence metrics
        """
        analysis = {}
        
        for method_name, history in self.training_history.items():
            train_losses = history['train_losses']
            val_losses = history['val_losses']
            
            # Compute convergence metrics
            final_train_loss = train_losses[-1] if train_losses else 0
            final_val_loss = val_losses[-1] if val_losses else 0
            min_train_loss = min(train_losses) if train_losses else 0
            min_val_loss = min(val_losses) if val_losses else 0
            
            # Convergence rate (slope of last 10% of training)
            if len(train_losses) > 10:
                last_10_percent = int(0.1 * len(train_losses))
                train_slope = np.polyfit(range(last_10_percent), train_losses[-last_10_percent:], 1)[0]
                val_slope = np.polyfit(range(last_10_percent), val_losses[-last_10_percent:], 1)[0]
            else:
                train_slope = 0
                val_slope = 0
            
            analysis[method_name] = {
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'min_train_loss': min_train_loss,
                'min_val_loss': min_val_loss,
                'train_convergence_rate': train_slope,
                'val_convergence_rate': val_slope,
                'n_epochs': len(train_losses)
            }
        
        return analysis
