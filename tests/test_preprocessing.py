"""Unit tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import ORBATPreprocessor, prepare_train_test_split


def create_sample_dataframe():
    """Create sample ORBAT dataframe for testing."""
    return pd.DataFrame({
        'equipment_type': ['tank', 'radar', 'artillery', 'tank'],
        'equipment_count': [10, 3, 8, 12],
        'communication_frequency': [150.0, 200.0, 120.0, np.nan],
        'mobility': ['mobile', 'static', 'mobile', 'mobile'],
        'hierarchy_level': ['Battalion', 'HQ', 'Regiment', 'Battalion'],
        'communication_degree': [5, 10, 5, 6],
        'location_x': [500.0, 100.0, 600.0, 450.0],
        'location_y': [300.0, 200.0, 400.0, 320.0],
        'cluster_id': [2, 0, 3, 2],
        'unit_id': ['UNIT_001', 'UNIT_002', 'UNIT_003', 'UNIT_001']
    })


def test_preprocessor_fit_transform():
    """Test preprocessor fit and transform."""
    df = create_sample_dataframe()
    preprocessor = ORBATPreprocessor()
    
    X, y = preprocessor.fit_transform(df)
    
    # Check shapes
    assert X.shape[0] == len(df)
    assert X.shape[1] > 0
    assert len(y) == len(df)
    
    # Check no NaN values
    assert not np.isnan(X).any()
    
    # Check target encoding
    assert len(np.unique(y)) == df['unit_id'].nunique()


def test_preprocessor_transform():
    """Test preprocessor transform on new data."""
    df_train = create_sample_dataframe()
    df_test = create_sample_dataframe().iloc[:2]
    
    preprocessor = ORBATPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)
    
    # Check same number of features
    assert X_train.shape[1] == X_test.shape[1]
    
    # Check no NaN values
    assert not np.isnan(X_test).any()


def test_preprocessor_handles_missing_values():
    """Test missing value handling."""
    df = create_sample_dataframe()
    preprocessor = ORBATPreprocessor()
    
    X, y = preprocessor.fit_transform(df)
    
    # Should have no NaN values after preprocessing
    assert not np.isnan(X).any()


def test_preprocessor_handles_unseen_categories():
    """Test handling of unseen categorical values."""
    df_train = create_sample_dataframe()
    preprocessor = ORBATPreprocessor()
    preprocessor.fit_transform(df_train)
    
    # Create test data with unseen category
    df_test = pd.DataFrame({
        'equipment_type': ['drone'],  # Unseen category
        'equipment_count': [5],
        'communication_frequency': [180.0],
        'mobility': ['mobile'],
        'hierarchy_level': ['Company'],
        'communication_degree': [4],
        'location_x': [400.0],
        'location_y': [250.0],
        'cluster_id': [1]
    })
    
    # Should not raise error
    X_test = preprocessor.transform(df_test)
    assert X_test.shape[0] == 1


def test_inverse_transform_target():
    """Test inverse transformation of target labels."""
    df = create_sample_dataframe()
    preprocessor = ORBATPreprocessor()
    
    X, y = preprocessor.fit_transform(df)
    y_original = preprocessor.inverse_transform_target(y)
    
    # Check if inverse transform recovers original labels
    assert all(y_original == df['unit_id'].values)


def test_train_test_split():
    """Test train/test split function."""
    df = create_sample_dataframe()
    
    train_df, test_df = prepare_train_test_split(df, test_size=0.25, random_state=42)
    
    # Check split proportions
    assert len(train_df) + len(test_df) == len(df)
    assert len(test_df) == 1  # 25% of 4 samples
    
    # Check no overlap
    train_indices = set(train_df.index)
    test_indices = set(test_df.index)
    assert len(train_indices.intersection(test_indices)) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
