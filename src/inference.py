"""Inference interface for ORBAT classification system."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import joblib
from .preprocessing import ORBATPreprocessor
from .hybrid_system import HybridORBATSystem


class ORBATPredictor:
    """High-level interface for ORBAT predictions."""
    
    def __init__(
        self,
        preprocessor: ORBATPreprocessor,
        hybrid_system: HybridORBATSystem,
        hierarchy_map: Dict[str, str] = None
    ):
        """Initialize predictor.
        
        Args:
            preprocessor: Fitted preprocessor
            hybrid_system: Trained hybrid system
            hierarchy_map: Mapping from unit_id to hierarchy string
        """
        self.preprocessor = preprocessor
        self.hybrid_system = hybrid_system
        self.hierarchy_map = hierarchy_map or {}
    
    def predict(
        self, 
        input_data: Dict[str, Any],
        return_details: bool = False
    ) -> Dict[str, Any]:
        """Predict unit for new observation.
        
        Args:
            input_data: Dictionary with observation features
            return_details: Whether to include detailed prediction info
            
        Returns:
            Prediction result with unit, hierarchy, and confidence
        """
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocess
        X = self.preprocessor.transform(df)
        
        # Predict
        pred_encoded, confidence, details = self.hybrid_system.predict_single(X)
        
        # Decode prediction
        predicted_unit = self.preprocessor.inverse_transform_target([pred_encoded])[0]
        
        # Get hierarchy
        hierarchy = self.hierarchy_map.get(predicted_unit, "Unknown")
        
        # Build result
        result = {
            "predicted_unit": str(predicted_unit),
            "hierarchy": hierarchy,
            "confidence_score": float(confidence)
        }
        
        if return_details:
            # Decode candidate units
            candidates_encoded = details['candidates']
            candidates = self.preprocessor.inverse_transform_target(candidates_encoded)
            
            result['details'] = {
                'candidate_units': [str(u) for u in candidates],
                'classification_probabilities': details['classification_probs'],
                'similarity_scores': details['similarity_scores'],
                'combined_scores': details['combined_scores']
            }
        
        return result
    
    def predict_batch(
        self, 
        input_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict for multiple observations.
        
        Args:
            input_data: List of observation dictionaries
            
        Returns:
            List of prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame(input_data)
        
        # Preprocess
        X = self.preprocessor.transform(df)
        
        # Predict
        predictions, confidences, _ = self.hybrid_system.predict(X)
        
        # Decode predictions
        predicted_units = self.preprocessor.inverse_transform_target(predictions)
        
        # Build results
        results = []
        for unit, conf in zip(predicted_units, confidences):
            unit_str = str(unit)
            results.append({
                "predicted_unit": unit_str,
                "hierarchy": self.hierarchy_map.get(unit_str, "Unknown"),
                "confidence_score": float(conf)
            })
        
        return results
    
    def save(self, filepath: str):
        """Save complete predictor to disk."""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'ORBATPredictor':
        """Load predictor from disk."""
        return joblib.load(filepath)


def build_hierarchy_map(df: pd.DataFrame) -> Dict[str, str]:
    """Build hierarchy mapping from dataset.
    
    Args:
        df: DataFrame with unit_id and hierarchy_level columns
        
    Returns:
        Dictionary mapping unit_id to hierarchy string
    """
    hierarchy_map = {}
    
    if 'unit_id' in df.columns and 'hierarchy_level' in df.columns:
        for unit_id in df['unit_id'].unique():
            unit_data = df[df['unit_id'] == unit_id]
            hierarchy_levels = unit_data['hierarchy_level'].unique()
            
            # Build hierarchy string (simplified)
            if len(hierarchy_levels) == 1:
                hierarchy_map[str(unit_id)] = str(hierarchy_levels[0])
            else:
                # If multiple levels, create hierarchy chain
                level_order = ['HQ', 'Regiment', 'Brigade', 'Battalion', 'Company', 'Section']
                present_levels = [l for l in level_order if l in hierarchy_levels]
                hierarchy_map[str(unit_id)] = ' → '.join(present_levels)
    
    return hierarchy_map
