"""Unit tests for models module."""

import pytest
import numpy as np
from src.models import ClassificationModel, SimilarityModel


def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 5, 100)
    return X, y


def test_classification_model_xgboost():
    """Test XGBoost classification model."""
    X, y = create_sample_data()
    
    model = ClassificationModel(model_type='xgboost', n_estimators=10)
    model.train(X, y)
    
    # Test prediction
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert all(p in np.unique(y) for p in predictions)
    
    # Test probability prediction
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), len(np.unique(y)))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classification_model_lightgbm():
    """Test LightGBM classification model."""
    X, y = create_sample_data()
    
    model = ClassificationModel(model_type='lightgbm', n_estimators=10)
    model.train(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == len(X)


def test_classification_model_top_k():
    """Test top-K predictions."""
    X, y = create_sample_data()
    
    model = ClassificationModel(model_type='xgboost', n_estimators=10)
    model.train(X, y)
    
    top_classes, top_probs = model.get_top_k_predictions(X, k=3)
    
    assert top_classes.shape == (len(X), 3)
    assert top_probs.shape == (len(X), 3)
    
    # Check probabilities are sorted descending
    for i in range(len(X)):
        assert all(top_probs[i][j] >= top_probs[i][j+1] for j in range(2))


def test_similarity_model_cosine():
    """Test similarity model with cosine metric."""
    X, y = create_sample_data()
    
    model = SimilarityModel(metric='cosine', n_neighbors=5)
    model.fit(X, y)
    
    # Test finding similar units
    similar_labels, scores = model.find_similar(X[:5], k=3)
    
    assert similar_labels.shape == (5, 3)
    assert scores.shape == (5, 3)
    
    # Check scores are in valid range
    assert np.all(scores >= -1) and np.all(scores <= 1)


def test_similarity_model_euclidean():
    """Test similarity model with Euclidean metric."""
    X, y = create_sample_data()
    
    model = SimilarityModel(metric='euclidean', n_neighbors=5)
    model.fit(X, y)
    
    similar_labels, scores = model.find_similar(X[:5], k=3)
    
    assert similar_labels.shape == (5, 3)
    assert scores.shape == (5, 3)


def test_similarity_to_candidates():
    """Test similarity computation to specific candidates."""
    X, y = create_sample_data()
    
    model = SimilarityModel(metric='cosine')
    model.fit(X, y)
    
    # Test with specific candidates
    candidates = np.array([0, 1, 2])
    scores = model.compute_similarity_to_candidates(X[0], candidates)
    
    assert len(scores) == len(candidates)
    assert all(isinstance(s, (int, float, np.number)) for s in scores)


def test_model_save_load(tmp_path):
    """Test model saving and loading."""
    X, y = create_sample_data()
    
    # Train and save
    model = ClassificationModel(model_type='xgboost', n_estimators=10)
    model.train(X, y)
    
    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))
    
    # Load and test
    loaded_model = ClassificationModel.load(str(save_path))
    
    # Compare predictions
    pred_original = model.predict(X)
    pred_loaded = loaded_model.predict(X)
    
    assert np.array_equal(pred_original, pred_loaded)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
