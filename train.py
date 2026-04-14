"""Training pipeline for ORBAT classification system."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

from src.preprocessing import ORBATPreprocessor, prepare_train_test_split
from src.models import ClassificationModel, SimilarityModel
from src.hybrid_system import HybridORBATSystem
from src.evaluation import ORBATEvaluator
from src.inference import ORBATPredictor, build_hierarchy_map


def train_orbat_system(
    data_path: str,
    output_dir: str = 'models',
    test_size: float = 0.2,
    val_size: float = 0.1,
    model_type: str = 'xgboost',
    similarity_metric: str = 'cosine',
    alpha: float = 0.6,
    random_state: int = 42
):
    """Complete training pipeline.
    
    Args:
        data_path: Path to CSV dataset
        output_dir: Directory to save models
        test_size: Test set proportion
        val_size: Validation set proportion
        model_type: 'xgboost' or 'lightgbm'
        similarity_metric: 'cosine' or 'euclidean'
        alpha: Weight for classification in hybrid system
        random_state: Random seed
    """
    print("=" * 70)
    print("ORBAT CLASSIFICATION SYSTEM - TRAINING PIPELINE")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load data
    print(f"\n[1/8] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Number of unique units: {df['unit_id'].nunique()}")
    
    # Split data
    print(f"\n[2/8] Splitting data (test={test_size}, val={val_size})...")
    train_val_df, test_df = prepare_train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    # Further split train into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = prepare_train_test_split(
        train_val_df, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Preprocessing
    print(f"\n[3/8] Preprocessing data...")
    preprocessor = ORBATPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df), preprocessor.target_encoder.transform(val_df['unit_id'])
    X_test, y_test = preprocessor.transform(test_df), preprocessor.target_encoder.transform(test_df['unit_id'])
    
    print(f"  Feature dimension: {X_train.shape[1]}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    
    # Train classification model
    print(f"\n[4/8] Training {model_type} classification model...")
    clf_model = ClassificationModel(model_type=model_type)
    clf_model.train(X_train, y_train, X_val, y_val)
    print("  Classification model trained successfully")
    
    # Train similarity model
    print(f"\n[5/8] Training similarity model (metric={similarity_metric})...")
    sim_model = SimilarityModel(metric=similarity_metric, n_neighbors=5)
    sim_model.fit(X_train, y_train)
    print("  Similarity model trained successfully")
    
    # Create hybrid system
    print(f"\n[6/8] Building hybrid system (alpha={alpha})...")
    hybrid_system = HybridORBATSystem(clf_model, sim_model, alpha=alpha)
    
    # Calibrate on validation set
    print("  Calibrating confidence scores...")
    calibration_metrics = hybrid_system.calibrate_confidence(X_val, y_val)
    print(f"  Expected Calibration Error: {calibration_metrics['expected_calibration_error']:.4f}")
    
    # Evaluate on test set
    print(f"\n[7/8] Evaluating on test set...")
    predictions, confidences, _ = hybrid_system.predict(X_test)
    
    # Get probabilities for top-k accuracy
    proba = clf_model.predict_proba(X_test)
    
    # Evaluate
    class_names = [str(c) for c in preprocessor.target_encoder.classes_]
    evaluator = ORBATEvaluator(class_names=class_names)
    metrics = evaluator.evaluate(y_test, predictions, proba, confidences)
    
    evaluator.print_summary()
    
    # Save models and artifacts
    print(f"\n[8/8] Saving models to {output_dir}/...")
    preprocessor.save(output_path / 'preprocessor.pkl')
    clf_model.save(output_path / 'classification_model.pkl')
    sim_model.save(output_path / 'similarity_model.pkl')
    
    # Build hierarchy map
    hierarchy_map = build_hierarchy_map(df)
    
    # Create predictor
    predictor = ORBATPredictor(preprocessor, hybrid_system, hierarchy_map)
    predictor.save(output_path / 'orbat_predictor.pkl')
    
    # Save metrics
    evaluator.save_metrics(output_path / 'evaluation_metrics.json')
    
    # Save configuration
    config = {
        'model_type': model_type,
        'similarity_metric': similarity_metric,
        'alpha': alpha,
        'test_size': test_size,
        'val_size': val_size,
        'random_state': random_state,
        'n_features': int(X_train.shape[1]),
        'n_classes': int(len(np.unique(y_train))),
        'calibration_metrics': {k: float(v) for k, v in calibration_metrics.items()}
    }
    
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nSaved artifacts:")
    print(f"  - {output_path / 'orbat_predictor.pkl'} (main predictor)")
    print(f"  - {output_path / 'preprocessor.pkl'}")
    print(f"  - {output_path / 'classification_model.pkl'}")
    print(f"  - {output_path / 'similarity_model.pkl'}")
    print(f"  - {output_path / 'evaluation_metrics.json'}")
    print(f"  - {output_path / 'config.json'}")
    
    return predictor, evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ORBAT classification system')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'lightgbm'])
    parser.add_argument('--similarity', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--alpha', type=float, default=0.6, help='Classification weight')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    train_orbat_system(
        data_path=args.data,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size,
        model_type=args.model,
        similarity_metric=args.similarity,
        alpha=args.alpha,
        random_state=args.seed
    )
