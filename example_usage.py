"""Example usage of ORBAT classification system."""

from src.inference import ORBATPredictor
import json


def example_single_prediction():
    """Example: Single observation prediction."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Observation Prediction")
    print("=" * 70)
    
    # Load trained predictor
    predictor = ORBATPredictor.load('models/orbat_predictor.pkl')
    
    # New observation
    observation = {
        'personnel_count': 600,
        'latitude': 45.2345,
        'longitude': 67.8901,
        'equipment_score': 250,
        'total_equipment_count': 50,
        'dominant_equipment_type': 1  # tank
    }
    
    print("\nInput Observation:")
    print(json.dumps(observation, indent=2))
    
    # Predict
    result = predictor.predict(observation, return_details=True)
    
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))


def example_batch_prediction():
    """Example: Batch prediction."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Prediction")
    print("=" * 70)
    
    # Load trained predictor
    predictor = ORBATPredictor.load('models/orbat_predictor.pkl')
    
    # Multiple observations
    observations = [
        {
            'personnel_count': 300,
            'latitude': 50.0000,
            'longitude': 70.0000,
            'equipment_score': 120,
            'total_equipment_count': 30,
            'dominant_equipment_type': 3  # radar
        },
        {
            'personnel_count': 500,
            'latitude': 42.5000,
            'longitude': 65.2500,
            'equipment_score': 180,
            'total_equipment_count': 40,
            'dominant_equipment_type': 2  # artillery
        }
    ]
    
    print(f"\nProcessing {len(observations)} observations...")
    
    # Predict
    results = predictor.predict_batch(observations)
    
    print("\nBatch Results:")
    for i, result in enumerate(results):
        print(f"\nObservation {i+1}:")
        print(f"  Predicted Unit: {result['predicted_unit']}")
        print(f"  Hierarchy: {result['hierarchy']}")
        print(f"  Confidence: {result['confidence_score']:.4f}")


def example_confidence_analysis():
    """Example: Analyzing prediction confidence."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Confidence Analysis")
    print("=" * 70)
    
    # Load trained predictor
    predictor = ORBATPredictor.load('models/orbat_predictor.pkl')
    
    # Test with varying confidence scenarios
    test_cases = [
        {
            'name': 'Heavy Armor Unit (Tank Battalion)',
            'data': {
                'personnel_count': 600,
                'latitude': 45.0000,
                'longitude': 68.0000,
                'equipment_score': 280,
                'total_equipment_count': 55,
                'dominant_equipment_type': 1  # tank
            }
        },
        {
            'name': 'Artillery Unit',
            'data': {
                'personnel_count': 500,
                'latitude': 42.5000,
                'longitude': 75.0000,
                'equipment_score': 200,
                'total_equipment_count': 45,
                'dominant_equipment_type': 2  # artillery
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        result = predictor.predict(case['data'], return_details=True)
        print(f"  Predicted: {result['predicted_unit']}")
        print(f"  Confidence: {result['confidence_score']:.4f}")
        print(f"  Top Candidates: {result['details']['candidate_units']}")
        print(f"  Combined Scores: {[f'{s:.3f}' for s in result['details']['combined_scores']]}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ORBAT CLASSIFICATION SYSTEM - USAGE EXAMPLES")
    print("=" * 70)
    print("\nNote: Run train.py first to generate the model files.")
    print("Example: python train.py --data data/orbat_training.csv")
    
    try:
        example_single_prediction()
        example_batch_prediction()
        example_confidence_analysis()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70 + "\n")
        
    except FileNotFoundError:
        print("\nError: Model files not found. Please run training first:")
        print("  python generate_realistic_data.py")
        print("  python train.py --data data/orbat_training.csv")
