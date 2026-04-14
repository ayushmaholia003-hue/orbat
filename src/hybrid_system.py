"""Hybrid prediction system combining classification and similarity matching."""

import numpy as np
from typing import Dict, List, Tuple
from .models import ClassificationModel, SimilarityModel


class HybridORBATSystem:
    """Combines classification and similarity for robust predictions."""
    
    def __init__(
        self, 
        classification_model: ClassificationModel,
        similarity_model: SimilarityModel,
        alpha: float = 0.6
    ):
        """Initialize hybrid system.
        
        Args:
            classification_model: Trained classification model
            similarity_model: Trained similarity model
            alpha: Weight for classification score (1-alpha for similarity)
        """
        self.clf_model = classification_model
        self.sim_model = similarity_model
        self.alpha = alpha
    
    def predict(
        self, 
        X: np.ndarray, 
        top_k: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Hybrid prediction combining both approaches.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            top_k: Number of candidates from classification
            
        Returns:
            predictions: Predicted unit labels
            confidence_scores: Confidence for each prediction
            details: Dictionary with detailed prediction info
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        confidence_scores = np.zeros(n_samples)
        details_list = []
        
        # Step 1: Get top-k candidates from classification model
        top_k_classes, top_k_probs = self.clf_model.get_top_k_predictions(X, k=top_k)
        
        # Step 2: For each sample, refine with similarity
        for i in range(n_samples):
            sample = X[i:i+1]
            candidates = top_k_classes[i]
            clf_probs = top_k_probs[i]
            
            # Compute similarity scores to candidates
            sim_scores = self.sim_model.compute_similarity_to_candidates(
                sample[0], candidates
            )
            
            # Normalize similarity scores
            if sim_scores.sum() > 0:
                sim_scores_norm = sim_scores / sim_scores.sum()
            else:
                sim_scores_norm = np.ones_like(sim_scores) / len(sim_scores)
            
            # Combine scores: weighted average
            combined_scores = self.alpha * clf_probs + (1 - self.alpha) * sim_scores_norm
            
            # Select best candidate
            best_idx = np.argmax(combined_scores)
            predictions[i] = candidates[best_idx]
            confidence_scores[i] = combined_scores[best_idx]
            
            # Store details
            details_list.append({
                'candidates': candidates.tolist(),
                'classification_probs': clf_probs.tolist(),
                'similarity_scores': sim_scores.tolist(),
                'combined_scores': combined_scores.tolist(),
                'selected_index': int(best_idx)
            })
        
        details = {
            'per_sample': details_list,
            'alpha': self.alpha
        }
        
        return predictions, confidence_scores, details
    
    def predict_single(
        self, 
        X: np.ndarray, 
        top_k: int = 3
    ) -> Tuple[int, float, Dict]:
        """Predict for a single sample.
        
        Args:
            X: Single feature vector (1, n_features)
            top_k: Number of candidates
            
        Returns:
            prediction: Predicted unit label
            confidence: Confidence score
            details: Prediction details
        """
        predictions, confidences, details = self.predict(X, top_k)
        return predictions[0], confidences[0], details['per_sample'][0]
    
    def calibrate_confidence(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Calibrate confidence scores using validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Calibration metrics
        """
        predictions, confidences, _ = self.predict(X_val)
        
        # Compute accuracy at different confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        calibration_metrics = {}
        
        for thresh in thresholds:
            mask = confidences >= thresh
            if mask.sum() > 0:
                accuracy = (predictions[mask] == y_val[mask]).mean()
                coverage = mask.mean()
                calibration_metrics[f'accuracy_at_{thresh}'] = accuracy
                calibration_metrics[f'coverage_at_{thresh}'] = coverage
        
        # Expected Calibration Error (simplified)
        correct = (predictions == y_val).astype(float)
        bins = np.linspace(0, 1, 11)
        ece = 0
        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                avg_confidence = confidences[mask].mean()
                avg_accuracy = correct[mask].mean()
                ece += mask.sum() / len(confidences) * abs(avg_confidence - avg_accuracy)
        
        calibration_metrics['expected_calibration_error'] = ece
        
        return calibration_metrics
