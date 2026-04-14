"""Unit tests for hybrid system."""

import pytest
import numpy as np
from src.models import ClassificationModel, SimilarityModel
from src.hybrid_system import HybridORBATSystem


def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 5, 100)
    X_test = np.random.randn(20, 10)
    y_test = np.random.randint(0, 5, 20)
    return X_train, y_train, X_test, y_test


def test_hybrid_system_prediction():
    """Test hybrid system prediction."""
    X_train, y_train, X_test, y_test = create_sample_data()
    
    # Train models
    clf_model = ClassificationModel(model_type='xgboost', n_estimators=10)
    clf_model.train(X_train, y_train)
    
    sim_model = SimilarityModel(metric='cosine')
    sim_model.fit(X_train, y_train)
    
    # Create hybrid system
    hybrid = HybridORBATSystem(clf_model, sim_model, alpha=0.6)
    
    # Predict
    predictions, confidences, details = hybrid.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert len(confidences) == len(X_test)
    assert all(0 <= c <= 1 for c in confidences)


def test_hybrid_system_single_prediction():
    """Test single sample prediction."""
    X_train, y_train, X_test, y_test = create_sample_data()
    
    clf_model = ClassificationModel(model_type='xgboost', n_estimators=10)
    clf_model.train(X_train, y_train)
    
    sim_model = SimilarityModel(metric='cosine')
    sim_model.fit(X_train, y_train)
    
    hybrid = HybridORBATSystem(clf_model, sim_model, alpha=0.6)
    
    # Single prediction
    pred, conf, details = hybrid.predict_single(X_test[:1])
    
    assert isinstance(pred, (int, np.integer))
    assert 0 <= conf <= 1
    assert 'candidates' in details
    assert 'combined_scores' in details


def test_hybrid_system_alpha_values():
    """Test different alpha values."""
    X_train, y_train, X_test, y_test = create_sample_data()
    
    clf_model = ClassificationModel(model_type='xgboost', n_estimators=10)
    clf_model.train(X_train, y_train)
    
    sim_model = SimilarityModel(metric='cosine')
    sim_model.fit(X_train, y_train)
    
    # Test different alpha values
    for alpha in [0.0, 0.5, 1.0]:
        hybrid = HybridORBATSystem(clf_model, sim_model, alpha=alpha)
        predictions, confidences, _ = hybrid.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(0 <= c <= 1 for c in confidences)


def test_confidence_calibration():
    """Test confidence calibration."""
    X_train, y_train, X_test, y_test = create_sample_data()
    
    clf_model = ClassificationModel(model_type='xgboost', n_estimators=10)
    clf_model.train(X_train, y_train)
    
    sim_model = SimilarityModel(metric='cosine')
    sim_model.fit(X_train, y_train)
    
    hybrid = HybridORBATSystem(clf_model, sim_model, alpha=0.6)
    
    # Calibrate
    calibration_metrics = hybrid.calibrate_confidence(X_test, y_test)
    
    assert 'expected_calibration_error' in calibration_metrics
    assert calibration_metrics['expected_calibration_error'] >= 0


def test_hybrid_details_structure():
    """Test structure of prediction details."""
    X_train, y_train, X_test, y_test = create_sample_data()
    
    clf_model = ClassificationModel(model_type='xgboost', n_estimators=10)
    clf_model.train(X_train, y_train)
    
    sim_model = SimilarityModel(metric='cosine')
    sim_model.fit(X_train, y_train)
    
    hybrid = HybridORBATSystem(clf_model, sim_model, alpha=0.6)
    
    _, _, details = hybrid.predict(X_test[:1], top_k=3)
    
    sample_details = details['per_sample'][0]
    
    assert 'candidates' in sample_details
    assert 'classification_probs' in sample_details
    assert 'similarity_scores' in sample_details
    assert 'combined_scores' in sample_details
    assert 'selected_index' in sample_details
    
    assert len(sample_details['candidates']) == 3
    assert len(sample_details['classification_probs']) == 3
    assert len(sample_details['similarity_scores']) == 3
    assert len(sample_details['combined_scores']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
