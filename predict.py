"""Inference script for ORBAT classification."""

import argparse
import json
import pandas as pd
from pathlib import Path
from src.inference import ORBATPredictor


def predict_single(model_path: str, input_data: dict, verbose: bool = True):
    """Predict unit for a single observation.
    
    Args:
        model_path: Path to saved predictor
        input_data: Dictionary with observation features
        verbose: Whether to print detailed output
    """
    # Load predictor
    predictor = ORBATPredictor.load(model_path)
    
    # Predict
    result = predictor.predict(input_data, return_details=verbose)
    
    if verbose:
        print("=" * 60)
        print("ORBAT PREDICTION RESULT")
        print("=" * 60)
        print(f"\nPredicted Unit: {result['predicted_unit']}")
        print(f"Hierarchy: {result['hierarchy']}")
        print(f"Confidence Score: {result['confidence_score']:.4f}")
        
        if 'details' in result:
            print("\nDetailed Analysis:")
            print(f"  Candidate Units: {result['details']['candidate_units']}")
            print(f"  Classification Probs: {[f'{p:.4f}' for p in result['details']['classification_probabilities']]}")
            print(f"  Similarity Scores: {[f'{s:.4f}' for s in result['details']['similarity_scores']]}")
            print(f"  Combined Scores: {[f'{c:.4f}' for c in result['details']['combined_scores']]}")
        
        print("=" * 60)
    
    return result


def predict_batch(model_path: str, input_file: str, output_file: str = None):
    """Predict units for multiple observations from CSV file.
    
    Args:
        model_path: Path to saved predictor
        input_file: Path to CSV file with observations
        output_file: Path to save predictions (optional)
    """
    # Load predictor
    predictor = ORBATPredictor.load(model_path)
    
    # Load input data
    df = pd.read_csv(input_file)
    
    # Check if it has observation_id column (test format)
    has_obs_id = 'observation_id' in df.columns
    
    # Get feature columns (exclude observation_id and expected_unit if present)
    feature_cols = [c for c in df.columns if c not in ['observation_id', 'expected_unit', 'unit_id']]
    input_data = df[feature_cols].to_dict('records')
    
    # Predict
    results = predictor.predict_batch(input_data)
    
    print(f"Processed {len(results)} observations")
    
    # Create output dataframe
    if has_obs_id:
        output_df = df.copy()
        output_df['predicted_unit'] = [r['predicted_unit'] for r in results]
        output_df['confidence_score'] = [r['confidence_score'] for r in results]
        output_df['hierarchy'] = [r['hierarchy'] for r in results]
        
        # Rename expected_unit to actual_unit if it exists and has values
        if 'expected_unit' in output_df.columns:
            output_df = output_df.rename(columns={'expected_unit': 'actual_unit'})
    else:
        # Create new dataframe with results
        output_df = pd.DataFrame(results)
        # Add original features
        for col in feature_cols:
            output_df[col] = df[col]
    
    # Save results
    if output_file:
        output_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(f"\nSample predictions:")
        print(output_df.head(10))
    
    return output_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORBAT prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to saved predictor')
    parser.add_argument('--input', type=str, help='Input CSV file for batch prediction')
    parser.add_argument('--output', type=str, help='Output CSV file for batch prediction')
    
    # Single prediction arguments
    parser.add_argument('--personnel-count', type=int, help='Personnel count')
    parser.add_argument('--latitude', type=float, help='Latitude')
    parser.add_argument('--longitude', type=float, help='Longitude')
    parser.add_argument('--equipment-score', type=int, help='Equipment score')
    parser.add_argument('--total-equipment', type=int, help='Total equipment count')
    parser.add_argument('--dominant-equipment', type=int, help='Dominant equipment type (0-4)')
    
    args = parser.parse_args()
    
    if args.input:
        # Batch prediction from CSV
        predict_batch(args.model, args.input, args.output)
    else:
        # Single prediction
        input_data = {}
        if args.personnel_count is not None:
            input_data['personnel_count'] = args.personnel_count
        if args.latitude is not None:
            input_data['latitude'] = args.latitude
        if args.longitude is not None:
            input_data['longitude'] = args.longitude
        if args.equipment_score is not None:
            input_data['equipment_score'] = args.equipment_score
        if args.total_equipment is not None:
            input_data['total_equipment_count'] = args.total_equipment
        if args.dominant_equipment is not None:
            input_data['dominant_equipment_type'] = args.dominant_equipment
        
        if not input_data:
            print("Error: No input data provided. Use --input for batch or provide feature arguments.")
            print("\nExample:")
            print("  python predict.py --model models/orbat_predictor.pkl \\")
            print("    --personnel-count 600 --latitude 45.5 --longitude 67.8 \\")
            print("    --equipment-score 250 --total-equipment 50 --dominant-equipment 1")
        else:
            predict_single(args.model, input_data, verbose=True)
