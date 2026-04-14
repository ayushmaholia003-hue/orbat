"""Evaluation metrics and visualization for ORBAT classification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    top_k_accuracy_score
)
from typing import Dict, List, Tuple
import json


class ORBATEvaluator:
    """Comprehensive evaluation for ORBAT classification system."""
    
    def __init__(self, class_names: List[str] = None):
        """Initialize evaluator.
        
        Args:
            class_names: List of unit_id names for display
        """
        self.class_names = class_names
        self.metrics = {}
    
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: np.ndarray = None,
        confidence_scores: np.ndarray = None
    ) -> Dict:
        """Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (for top-k accuracy)
            confidence_scores: Confidence scores
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Top-K accuracy
        if y_proba is not None:
            for k in [3, 5]:
                if y_proba.shape[1] >= k:
                    metrics[f'top_{k}_accuracy'] = top_k_accuracy_score(
                        y_true, y_proba, k=k
                    )
        
        # Per-class metrics
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Confidence calibration
        if confidence_scores is not None:
            metrics['confidence_stats'] = {
                'mean': float(confidence_scores.mean()),
                'std': float(confidence_scores.std()),
                'min': float(confidence_scores.min()),
                'max': float(confidence_scores.max())
            }
            
            # Accuracy by confidence bins
            bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
            bin_accuracies = []
            for i in range(len(bins) - 1):
                mask = (confidence_scores >= bins[i]) & (confidence_scores < bins[i+1])
                if mask.sum() > 0:
                    acc = (y_true[mask] == y_pred[mask]).mean()
                    bin_accuracies.append({
                        'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                        'accuracy': float(acc),
                        'count': int(mask.sum())
                    })
            metrics['confidence_calibration'] = bin_accuracies
        
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrix(
        self, 
        cm: np.ndarray = None, 
        save_path: str = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix (uses stored if None)
            save_path: Path to save figure
            figsize: Figure size
        """
        if cm is None:
            cm = self.metrics.get('confusion_matrix')
        
        if cm is None:
            raise ValueError("No confusion matrix available")
        
        plt.figure(figsize=figsize)
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else 'auto',
            yticklabels=self.class_names if self.class_names else 'auto',
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_distribution(
        self, 
        confidence_scores: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """Plot confidence score distribution for correct vs incorrect predictions.
        
        Args:
            confidence_scores: Confidence scores
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        correct = y_true == y_pred
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(
            confidence_scores[correct], 
            bins=30, 
            alpha=0.6, 
            label='Correct Predictions',
            color='green',
            edgecolor='black'
        )
        plt.hist(
            confidence_scores[~correct], 
            bins=30, 
            alpha=0.6, 
            label='Incorrect Predictions',
            color='red',
            edgecolor='black'
        )
        
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Confidence Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.metrics:
            print("No metrics available. Run evaluate() first.")
            return
        
        print("=" * 60)
        print("ORBAT CLASSIFICATION EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nOverall Accuracy: {self.metrics['accuracy']:.4f}")
        
        if 'top_3_accuracy' in self.metrics:
            print(f"Top-3 Accuracy: {self.metrics['top_3_accuracy']:.4f}")
        if 'top_5_accuracy' in self.metrics:
            print(f"Top-5 Accuracy: {self.metrics['top_5_accuracy']:.4f}")
        
        if 'confidence_stats' in self.metrics:
            print("\nConfidence Statistics:")
            stats = self.metrics['confidence_stats']
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        if 'confidence_calibration' in self.metrics:
            print("\nConfidence Calibration:")
            for bin_info in self.metrics['confidence_calibration']:
                print(f"  {bin_info['bin']}: Accuracy={bin_info['accuracy']:.4f}, Count={bin_info['count']}")
        
        print("\n" + "=" * 60)
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            else:
                metrics_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
