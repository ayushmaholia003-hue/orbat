"""Generate realistic ORBAT datasets with military intelligence features."""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)


def calculate_equipment_score(equipment_counts):
    """Calculate weighted equipment score based on combat power.
    
    Weights reflect combat effectiveness:
    - tank: 5 (heavy armor, high firepower)
    - artillery: 4 (long-range fire support)
    - missile: 4 (precision strike capability)
    - radar: 3 (force multiplier, situational awareness)
    - infantry: 2 (basic combat unit)
    """
    weights = {
        'tank': 5,
        'artillery': 4,
        'missile': 4,
        'radar': 3,
        'infantry': 2
    }
    
    score = sum(equipment_counts.get(eq_type, 0) * weight 
                for eq_type, weight in weights.items())
    return score


def get_dominant_equipment(equipment_counts):
    """Get the most prominent equipment type.
    
    Returns:
        0 = infantry
        1 = tank
        2 = artillery
        3 = radar
        4 = missile
    """
    if not equipment_counts:
        return 0
    
    dominant = max(equipment_counts.items(), key=lambda x: x[1])[0]
    
    mapping = {
        'infantry': 0,
        'tank': 1,
        'artillery': 2,
        'radar': 3,
        'missile': 4
    }
    
    return mapping.get(dominant, 0)


def generate_training_dataset(n_units: int = 15, samples_per_unit: int = 40):
    """Generate realistic ORBAT training dataset.
    
    Features (based on military intelligence):
    - personnel_count: Total number of soldiers
    - latitude: Geographical latitude
    - longitude: Geographical longitude
    - equipment_score: Weighted combat power score
    - total_equipment_count: Total equipment items
    - dominant_equipment_type: Primary equipment category (0-4)
    - unit_id: Target variable (unit identifier)
    
    Args:
        n_units: Number of unique military units
        samples_per_unit: Number of observations per unit
    """
    data = []
    
    # Define unit archetypes with realistic characteristics
    unit_archetypes = [
        # Heavy Armor Units (Tank-heavy)
        {
            'type': 'armor_battalion',
            'personnel_mean': 600,
            'personnel_std': 50,
            'equipment': {'tank': (30, 5), 'infantry': (100, 10), 'artillery': (8, 2)},
            'dominant': 'tank'
        },
        # Artillery Units
        {
            'type': 'artillery_battalion',
            'personnel_mean': 500,
            'personnel_std': 40,
            'equipment': {'artillery': (24, 4), 'infantry': (80, 10), 'radar': (4, 1)},
            'dominant': 'artillery'
        },
        # Infantry Units
        {
            'type': 'infantry_battalion',
            'personnel_mean': 800,
            'personnel_std': 60,
            'equipment': {'infantry': (150, 15), 'tank': (10, 2), 'artillery': (6, 2)},
            'dominant': 'infantry'
        },
        # Missile Units
        {
            'type': 'missile_battalion',
            'personnel_mean': 400,
            'personnel_std': 30,
            'equipment': {'missile': (20, 3), 'radar': (6, 1), 'infantry': (60, 8)},
            'dominant': 'missile'
        },
        # Reconnaissance Units (Radar-heavy)
        {
            'type': 'recon_battalion',
            'personnel_mean': 300,
            'personnel_std': 25,
            'equipment': {'radar': (12, 2), 'infantry': (50, 8), 'tank': (5, 1)},
            'dominant': 'radar'
        }
    ]
    
    for unit_id in range(n_units):
        # Assign archetype to unit
        archetype = unit_archetypes[unit_id % len(unit_archetypes)]
        
        # Base location for this unit (operational area)
        # Latitude: 30-50 (realistic military operational zones)
        # Longitude: 40-80
        base_latitude = np.random.uniform(30, 50)
        base_longitude = np.random.uniform(40, 80)
        
        for _ in range(samples_per_unit):
            # Personnel count with variation
            personnel_count = max(50, int(np.random.normal(
                archetype['personnel_mean'],
                archetype['personnel_std']
            )))
            
            # Generate equipment counts
            equipment_counts = {}
            for eq_type, (mean, std) in archetype['equipment'].items():
                count = max(0, int(np.random.normal(mean, std)))
                if count > 0:
                    equipment_counts[eq_type] = count
            
            # Calculate derived features
            equipment_score = calculate_equipment_score(equipment_counts)
            total_equipment_count = sum(equipment_counts.values())
            dominant_equipment_type = get_dominant_equipment(equipment_counts)
            
            # Location with operational movement (±0.5 degrees)
            latitude = base_latitude + np.random.normal(0, 0.5)
            longitude = base_longitude + np.random.normal(0, 0.5)
            
            # Clip to realistic ranges
            latitude = np.clip(latitude, 25, 55)
            longitude = np.clip(longitude, 35, 85)
            
            data.append({
                'personnel_count': personnel_count,
                'latitude': round(latitude, 4),
                'longitude': round(longitude, 4),
                'equipment_score': equipment_score,
                'total_equipment_count': total_equipment_count,
                'dominant_equipment_type': dominant_equipment_type,
                'unit_id': f'UNIT_{unit_id:03d}'
            })
    
    df = pd.DataFrame(data)
    
    # Add some realistic missing values (3% missing)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
    df.loc[missing_indices, 'personnel_count'] = np.nan
    
    return df


def generate_test_dataset(n_samples: int = 25):
    """Generate test dataset with observations to classify.
    
    This represents new, unseen observations that need unit identification.
    """
    data = []
    
    # Generate diverse test cases
    for i in range(n_samples):
        # Random personnel count (50-1000)
        personnel_count = np.random.randint(50, 1000)
        
        # Random location
        latitude = round(np.random.uniform(25, 55), 4)
        longitude = round(np.random.uniform(35, 85), 4)
        
        # Random equipment composition
        equipment_counts = {
            'tank': np.random.randint(0, 40),
            'artillery': np.random.randint(0, 30),
            'missile': np.random.randint(0, 25),
            'radar': np.random.randint(0, 15),
            'infantry': np.random.randint(20, 200)
        }
        
        # Calculate features
        equipment_score = calculate_equipment_score(equipment_counts)
        total_equipment_count = sum(equipment_counts.values())
        dominant_equipment_type = get_dominant_equipment(equipment_counts)
        
        data.append({
            'observation_id': f'OBS_{i+1:03d}',
            'personnel_count': personnel_count,
            'latitude': latitude,
            'longitude': longitude,
            'equipment_score': equipment_score,
            'total_equipment_count': total_equipment_count,
            'dominant_equipment_type': dominant_equipment_type,
            'expected_unit': ''  # To be filled after prediction
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    print("=" * 80)
    print("GENERATING REALISTIC ORBAT DATASETS")
    print("Military Intelligence Features")
    print("=" * 80)
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate training dataset
    print("\n[1/2] Generating training dataset...")
    train_df = generate_training_dataset(n_units=15, samples_per_unit=40)
    train_path = data_dir / 'orbat_training.csv'
    train_df.to_csv(train_path, index=False)
    
    print(f"✓ Training dataset saved: {train_path}")
    print(f"  - Shape: {train_df.shape}")
    print(f"  - Units: {train_df['unit_id'].nunique()}")
    print(f"  - Samples per unit: ~{len(train_df) // train_df['unit_id'].nunique()}")
    print(f"\n  Features:")
    print(f"    • personnel_count: {train_df['personnel_count'].min():.0f} - {train_df['personnel_count'].max():.0f}")
    print(f"    • latitude: {train_df['latitude'].min():.2f} - {train_df['latitude'].max():.2f}")
    print(f"    • longitude: {train_df['longitude'].min():.2f} - {train_df['longitude'].max():.2f}")
    print(f"    • equipment_score: {train_df['equipment_score'].min():.0f} - {train_df['equipment_score'].max():.0f}")
    print(f"    • total_equipment_count: {train_df['total_equipment_count'].min():.0f} - {train_df['total_equipment_count'].max():.0f}")
    print(f"    • dominant_equipment_type: {sorted(train_df['dominant_equipment_type'].unique())}")
    
    # Generate test dataset
    print("\n[2/2] Generating test dataset...")
    test_df = generate_test_dataset(n_samples=25)
    test_path = data_dir / 'orbat_test.csv'
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Test dataset saved: {test_path}")
    print(f"  - Shape: {test_df.shape}")
    print(f"  - Observations: {len(test_df)}")
    
    print("\n" + "=" * 80)
    print("DATASET PREVIEW")
    print("=" * 80)
    
    print("\nTraining Data (first 5 rows):")
    print(train_df.head())
    
    print("\nTraining Data - Unit Distribution:")
    unit_stats = train_df.groupby('unit_id').agg({
        'personnel_count': 'mean',
        'equipment_score': 'mean',
        'dominant_equipment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).round(0)
    print(unit_stats.head(10))
    
    print("\nTest Data (first 5 rows):")
    print(test_df.head())
    
    print("\n" + "=" * 80)
    print("FEATURE ENCODING REFERENCE")
    print("=" * 80)
    print("\ndominant_equipment_type:")
    print("  0 = infantry")
    print("  1 = tank")
    print("  2 = artillery")
    print("  3 = radar")
    print("  4 = missile")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Train the model:")
    print(f"   python train.py --data {train_path}")
    print("\n2. Make predictions on test data:")
    print(f"   python predict.py --model models/orbat_predictor.pkl --input {test_path} --output data/predictions.csv")
    print("\n" + "=" * 80)
