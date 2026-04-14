"""Classification and similarity-based models for ORBAT prediction."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, List, Dict
import joblib


class ClassificationModel:
    """XGBoost/LightGBM classifier optimized for resource-based identification."""
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """Initialize classification model with resource-focused parameters.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            **kwargs: Model hyperparameters
        """
        self.model_type = model_type
        
        if model_type == 'xgboost':
            # Optimized for resource-based identification
            default_params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'max_depth': 4,  # Simpler trees for resource focus
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.9,
                'colsample_bytree': 1.0,  # Use all features
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 0.1,  # L2 regularization
                'random_state': 42
            }
            default_params.update(kwargs)
            self.model = xgb.XGBClassifier(**default_params)
            
        elif model_type == 'lightgbm':
            default_params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_leaves': 15,  # Simpler model
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.9,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42
            }
            default_params.update(kwargs)
            self.model = lgb.LGBMClassifier(**default_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the classification model with feature importance focus.
        
        Args:
            X_train: Training features [equipment_score, latitude, longitude]
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        # Create feature weights: equipment_score gets high weight, location gets low weight
        sample_weight = None
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                sample_weight=sample_weight,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        return self.model.predict(X)
    
    def get_top_k_predictions(self, X: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k predictions with probabilities.
        
        Args:
            X: Feature matrix
            k: Number of top predictions
            
        Returns:
            top_classes: (n_samples, k) array of class indices
            top_probs: (n_samples, k) array of probabilities
        """
        proba = self.predict_proba(X)
        top_k_idx = np.argsort(proba, axis=1)[:, -k:][:, ::-1]
        top_k_proba = np.take_along_axis(proba, top_k_idx, axis=1)
        return top_k_idx, top_k_proba
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'ClassificationModel':
        """Load model from disk."""
        return joblib.load(filepath)


class SimilarityModel:
    """Location-based similarity for tiebreaking when resources are similar."""
    
    def __init__(self, metric: str = 'euclidean', n_neighbors: int = 5):
        """Initialize similarity model focused on geographic distance.
        
        Args:
            metric: 'euclidean' (for lat/long distance) or 'cosine'
            n_neighbors: Number of neighbors for KNN
        """
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.reference_embeddings = None
        self.reference_labels = None
        self.knn = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Store reference embeddings and labels.
        
        Args:
            X: Feature embeddings [equipment_score, latitude, longitude]
            y: Corresponding labels
        """
        self.reference_embeddings = X
        self.reference_labels = y
        
        # For location-based tiebreaking, focus on lat/long (columns 1,2)
        # Weight the features: equipment_score gets lower weight in similarity
        weighted_X = X.copy()
        weighted_X[:, 0] = weighted_X[:, 0] * 0.1  # Low weight for equipment_score
        weighted_X[:, 1] = weighted_X[:, 1] * 1.0  # High weight for latitude
        weighted_X[:, 2] = weighted_X[:, 2] * 1.0  # High weight for longitude
        
        # Initialize KNN for efficient similarity search
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm='auto'
        )
        self.knn.fit(weighted_X)
        self.weighted_reference = weighted_X
    
    def compute_similarity_to_candidates(
        self, 
        X: np.ndarray, 
        candidate_labels: np.ndarray
    ) -> np.ndarray:
        """Compute location-based similarity to specific candidate units.
        
        Focus on geographic distance when equipment scores are similar.
        
        Args:
            X: Query embedding (single sample) [equipment_score, lat, long]
            candidate_labels: Array of candidate unit labels
            
        Returns:
            Similarity scores for each candidate (higher = more similar)
        """
        # Weight the query the same way
        weighted_query = X.copy()
        weighted_query[0] = weighted_query[0] * 0.1  # Low weight for equipment_score
        weighted_query[1] = weighted_query[1] * 1.0  # High weight for latitude  
        weighted_query[2] = weighted_query[2] * 1.0  # High weight for longitude
        
        # Find indices of candidates in reference set
        candidate_mask = np.isin(self.reference_labels, candidate_labels)
        candidate_embeddings = self.weighted_reference[candidate_mask]
        candidate_labels_filtered = self.reference_labels[candidate_mask]
        
        if self.metric == 'euclidean':
            # Calculate Euclidean distances (lower = more similar)
            distances = np.sqrt(np.sum((candidate_embeddings - weighted_query) ** 2, axis=1))
            # Convert to similarity: similarity = 1 / (1 + distance)
            similarities = 1 / (1 + distances)
        else:
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(weighted_query.reshape(1, -1), candidate_embeddings)[0]
        
        # Map back to original candidate order
        scores = np.zeros(len(candidate_labels))
        for i, label in enumerate(candidate_labels):
            idx = np.where(candidate_labels_filtered == label)[0]
            if len(idx) > 0:
                scores[i] = similarities[idx[0]]
        
        return scores
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'SimilarityModel':
        """Load model from disk."""
        return joblib.load(filepath)
