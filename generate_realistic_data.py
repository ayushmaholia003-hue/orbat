"""Generate large realistic ORBAT dataset focused on resource identification."""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)


def generate_large_orbat_dataset(n_units: int = 50, samples_per_unit: int = 100):
    """Generate large realistic ORBAT dataset.
    
    Core concept: Units are identified primarily by resources (equipment_score),
    with location (lat/long) as tiebreaker when resources are similar.
    
    Features (only essential):
    - equipment_score: Primary identifier (high weight)
    - latitude: Tiebreaker for similar resources
    - longitude: Tiebreaker for similar resources
    - unit_name: Target variable (realistic military names)
    
    Args:
        n_units: Number of unique military units
        samples_per_unit: Number of observations per unit
    """
    
    # Realistic military unit names
    unit_types = [
        'Troop', 'Squadron', 'Section', 'Regiment', 'Battalion', 
        'Company', 'Platoon', 'Brigade', 'Division', 'Corps'
    ]
    
    # Generate unit names
    unit_names = []
    for i in range(n_units):
        unit_type = np.random.choice(unit_types)
        number = (i % 20) + 1  # Numbers 1-20
        unit_names.append(f"{unit_type}{number}")
    
    data = []
    
    for unit_idx in range(n_units):
        unit_name = unit_names[unit_idx]
        
        # Each unit has a characteristic equipment_score (primary identifier)
        # Spread units across different resource levels to avoid clustering
        base_equipment_score = 100 + (unit_idx * 15) + np.random.randint(-20, 20)
        
        # Each unit operates in a specific geographic area
        base_latitude = 30 + (unit_idx % 10) * 2.5 + np.random.uniform(-1, 1)
        base_longitude = 40 + (unit_idx % 8) * 5 + np.random.uniform(-2, 2)
        
        for sample_idx in range(samples_per_unit):
            # Equipment score with noise (primary feature - high weight)
            # 80% of time stays close to base, 20% has more variation
            if np.random.random() < 0.8:
                equipment_score = base_equipment_score + np.random.normal(0, 10)
            else:
                equipment_score = base_equipment_score + np.random.normal(0, 25)
            
            equipment_score = max(50, int(equipment_score))  # Minimum realistic value
            
            # Location with operational movement (tiebreaker features)
            # Units move within operational area (±2 degrees typical)
            latitude = base_latitude + np.random.normal(0, 1.5)
            longitude = base_longitude + np.random.normal(0, 2.0)
            
            # Clip to realistic military operational ranges
            latitude = np.clip(latitude, 25, 55)
            longitude = np.clip(longitude, 35, 85)
            
            data.append({
                'equipment_score': equipment_score,
                'latitude': round(latitude, 4),
                'longitude': round(longitude, 4),
                'unit_name': unit_name
            })
    
    df = pd.DataFrame(data)
    
    # Add realistic noise to prevent overfitting
    # 5% of samples get random noise
    noise_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
    
    for idx in noise_indices:
        # Add equipment score noise
        df.loc[idx, 'equipment_score'] += np.random.randint(-50, 50)
        df.loc[idx, 'equipment_score'] = max(50, df.loc[idx, 'equipment_score'])
        
        # Add location noise
        df.loc[idx, 'latitude'] += np.random.uniform(-5, 5)
        df.loc[idx, 'longitude'] += np.random.uniform(-5, 5)
        df.loc[idx, 'latitude'] = np.clip(df.loc[idx, 'latitude'], 25, 55)
        df.loc[idx, 'longitude'] = np.clip(df.loc[idx, 'longitude'], 35, 85)
    
    # Add some missing values (3% missing)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
    df.loc[missing_indices, 'equipment_score'] = np.nan
    
    return df


def generate_test_dataset(n_samples: int = 50):
    """Generate test dataset with new observations to classify."""
    
    data = []
    
    for i in range(n_samples):
        # Random equipment scores across the range
        equipment_score = np.random.randint(100, 900)
        
        # Random locations across operational zones
        latitude = round(np.random.uniform(25, 55), 4)
        longitude = round(np.random.uniform(35, 85), 4)
        
        data.append({
            'observation_id': f'OBS_{i+1:03d}',
            'equipment_score': equipment_score,
            'latitude': latitude,
            'longitude': longitude,
            'predicted_unit': ''  # To be filled after prediction
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    print("=" * 80)
    print("GENERATING LARGE REALISTIC ORBAT DATASET")
    print("Resource-focused identification with location tiebreaker")
    print("=" * 80)
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate large training dataset
    print("\n[1/2] Generating large training dataset...")
    train_df = generate_large_orbat_dataset(n_units=50, samples_per_unit=100)
    train_path = data_dir / 'orbat_training.csv'
    train_df.to_csv(train_path, index=False)
    
    print(f"✓ Training dataset saved: {train_path}")
    print(f"  - Shape: {train_df.shape}")
    print(f"  - Units: {train_df['unit_name'].nunique()}")
    print(f"  - Samples per unit: ~{len(train_df) // train_df['unit_name'].nunique()}")
    print(f"  - Equipment score range: {train_df['equipment_score'].min():.0f} - {train_df['equipment_score'].max():.0f}")
    print(f"  - Latitude range: {train_df['latitude'].min():.2f} - {train_df['latitude'].max():.2f}")
    print(f"  - Longitude range: {train_df['longitude'].min():.2f} - {train_df['longitude'].max():.2f}")
    
    # Generate test dataset
    print("\n[2/2] Generating test dataset...")
    test_df = generate_test_dataset(n_samples=50)
    test_path = data_dir / 'orbat_test.csv'
    test_df.to_csv(test_path, index=False)
    
    print(f"✓ Test dataset saved: {test_path}")
    print(f"  - Shape: {test_df.shape}")
    print(f"  - Observations: {len(test_df)}")
    
    print("\n" + "=" * 80)
    print("DATASET PREVIEW")
    print("=" * 80)
    
    print("\nTraining Data (first 10 rows):")
    print(train_df.head(10))
    
    print(f"\nUnique Units ({train_df['unit_name'].nunique()}):")
    print(sorted(train_df['unit_name'].unique())[:20], "...")
    
    print("\nUnit Statistics (sample):")
    unit_stats = train_df.groupby('unit_name').agg({
        'equipment_score': ['mean', 'std'],
        'latitude': 'mean',
        'longitude': 'mean'
    }).round(2)
    print(unit_stats.head(10))
    
    print("\nTest Data (first 5 rows):")
    print(test_df.head())
    
    print("\n" + "=" * 80)
    print("TRAINING CONCEPT")
    print("=" * 80)
    print("\n1. PRIMARY IDENTIFICATION: equipment_score (high weight)")
    print("   - Units have characteristic resource levels")
    print("   - Main feature for unit classification")
    print("\n2. TIEBREAKER: latitude + longitude (low weight)")
    print("   - When equipment_score is similar between units")
    print("   - Choose closest unit geographically")
    print("\n3. NOISE ADDED:")
    print("   - 5% samples have random noise")
    print("   - 3% missing values")
    print("   - Prevents overfitting")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Train the model:")
    print(f"   python train.py --data {train_path}")
    print("\n2. Make predictions:")
    print(f"   python predict.py --model models/orbat_predictor.pkl --input {test_path}")
    print("\n" + "=" * 80)