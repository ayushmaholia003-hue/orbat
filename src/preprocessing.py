"""Data preprocessing and feature engineering for ORBAT classification."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import joblib


class ORBATPreprocessor:
    """Handles data cleaning, encoding, and normalization."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.feature_names = None
        self.categorical_features = [
            'dominant_equipment_type'
        ]
        self.numerical_features = [
            'personnel_count', 'latitude', 'longitude', 
            'equipment_score', 'total_equipment_count'
        ]
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform training data.
        
        Args:
            df: DataFrame with all features and unit_id target
            
        Returns:
            X: Transformed feature matrix
            y: Encoded target labels
        """
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode target variable
        y = self.target_encoder.fit_transform(df['unit_id'])
        
        # Encode categorical features
        encoded_features = []
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                encoded_features.append(encoded.reshape(-1, 1))
        
        # Extract numerical features
        numerical_data = []
        for col in self.numerical_features:
            if col in df.columns:
                numerical_data.append(df[col].values.reshape(-1, 1))
        
        # Combine all features
        if encoded_features and numerical_data:
            X_combined = np.hstack(encoded_features + numerical_data)
        elif encoded_features:
            X_combined = np.hstack(encoded_features)
        else:
            X_combined = np.hstack(numerical_data)
        
        # Normalize numerical features
        X = self.scaler.fit_transform(X_combined)
        
        # Store feature names for reference
        self.feature_names = (
            self.categorical_features[:len(encoded_features)] + 
            self.numerical_features[:len(numerical_data)]
        )
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor.
        
        Args:
            df: DataFrame with features (no target)
            
        Returns:
            X: Transformed feature matrix
        """
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Encode categorical features
        encoded_features = []
        for col in self.categorical_features:
            if col in df.columns:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                encoded = le.transform(df[col])
                encoded_features.append(encoded.reshape(-1, 1))
        
        # Extract numerical features
        numerical_data = []
        for col in self.numerical_features:
            if col in df.columns:
                numerical_data.append(df[col].values.reshape(-1, 1))
        
        # Combine all features
        if encoded_features and numerical_data:
            X_combined = np.hstack(encoded_features + numerical_data)
        elif encoded_features:
            X_combined = np.hstack(encoded_features)
        else:
            X_combined = np.hstack(numerical_data)
        
        # Normalize
        X = self.scaler.transform(X_combined)
        
        return X
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Fill numerical missing values with median
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        return df
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to original unit_ids."""
        return self.target_encoder.inverse_transform(y_encoded)
    
    def save(self, filepath: str):
        """Save preprocessor to disk."""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'ORBATPreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(filepath)


def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets ensuring no data leakage.
    
    Args:
        df: Full dataset
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        train_df, test_df
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['unit_id'] if 'unit_id' in df.columns else None
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
