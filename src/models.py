"""Classification and similarity-based models for ORBAT prediction."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import lightgbm as lgb
from typing import Tuple, List, Dict
import joblib


class ClassificationModel:
    """XGBoost/LightGBM classifier for unit prediction."""
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """Initialize classification model.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            **kwargs: Model hyperparameters
        """
        self.model_type = model_type
        
        if model_type == 'xgboost':
            default_params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            default_params.update(kwargs)
            self.model = xgb.XGBClassifier(**default_params)
            
        elif model_type == 'lightgbm':
            default_params = {
                'objective': 'multiclass',
                'metric': 'multi_logloss',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            default_params.update(kwargs)
            self.model = lgb.LGBMClassifier(**default_params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
    
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
    """Similarity-based matching using embeddings."""
    
    def __init__(self, metric: str = 'cosine', n_neighbors: int = 5):
        """Initialize similarity model.
        
        Args:
            metric: 'cosine' or 'euclidean'
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
            X: Feature embeddings
            y: Corresponding labels
        """
        self.reference_embeddings = X
        self.reference_labels = y
        
        # Initialize KNN for efficient similarity search
        metric = 'cosine' if self.metric == 'cosine' else 'euclidean'
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=metric,
            algorithm='auto'
        )
        self.knn.fit(X)
    
    def find_similar(self, X: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Find k most similar units.
        
        Args:
            X: Query embeddings
            k: Number of similar units to return
            
        Returns:
            similar_labels: (n_samples, k) array of similar unit labels
            similarity_scores: (n_samples, k) array of similarity scores
        """
        if self.metric == 'cosine':
            # Cosine similarity
            similarities = cosine_similarity(X, self.reference_embeddings)
            top_k_idx = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
            top_k_scores = np.take_along_axis(similarities, top_k_idx, axis=1)
        else:
            # Euclidean distance (convert to similarity)
            distances = euclidean_distances(X, self.reference_embeddings)
            # Convert distance to similarity: similarity = 1 / (1 + distance)
            similarities = 1 / (1 + distances)
            top_k_idx = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
            top_k_scores = np.take_along_axis(similarities, top_k_idx, axis=1)
        
        similar_labels = self.reference_labels[top_k_idx]
        return similar_labels, top_k_scores
    
    def compute_similarity_to_candidates(
        self, 
        X: np.ndarray, 
        candidate_labels: np.ndarray
    ) -> np.ndarray:
        """Compute similarity scores to specific candidate units.
        
        Args:
            X: Query embedding (single sample)
            candidate_labels: Array of candidate unit labels
            
        Returns:
            Similarity scores for each candidate
        """
        # Find indices of candidates in reference set
        candidate_mask = np.isin(self.reference_labels, candidate_labels)
        candidate_embeddings = self.reference_embeddings[candidate_mask]
        candidate_labels_filtered = self.reference_labels[candidate_mask]
        
        if self.metric == 'cosine':
            similarities = cosine_similarity(X.reshape(1, -1), candidate_embeddings)[0]
        else:
            distances = euclidean_distances(X.reshape(1, -1), candidate_embeddings)[0]
            similarities = 1 / (1 + distances)
        
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
