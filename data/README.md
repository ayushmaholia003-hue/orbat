# ORBAT Dataset Directory

This directory contains ORBAT (Order of Battle) datasets for training and testing.

## Dataset Format

The system uses **6 military intelligence features** that are commonly available in real-world ORBAT data.

### Required Columns (Training Data)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `personnel_count` | numerical | Total number of soldiers | 50-1000+ |
| `latitude` | numerical | Geographical latitude | 25.0-55.0 |
| `longitude` | numerical | Geographical longitude | 35.0-85.0 |
| `equipment_score` | numerical | Weighted combat power score | 50-500+ |
| `total_equipment_count` | numerical | Total equipment items | 10-200+ |
| `dominant_equipment_type` | categorical | Primary equipment (0-4) | 0=infantry, 1=tank, 2=artillery, 3=radar, 4=missile |
| `unit_id` | target | Unit identifier (what to predict) | UNIT_001, UNIT_002, etc. |

### Training Data Example

```csv
personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,unit_id
600,45.2345,67.8901,250,50,1,UNIT_001
500,42.5000,75.0000,200,45,2,UNIT_002
300,50.0000,70.0000,120,30,3,UNIT_003
```

### Test Data Format (For Predictions)

Test data has the same features but includes `observation_id` and `expected_unit` columns:

```csv
observation_id,personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,expected_unit
OBS_001,600,45.5,68.0,280,55,1,
OBS_002,300,51.2,71.3,120,30,3,
```

The `expected_unit` column is left empty and will be filled with predictions.

## Feature Calculations

### equipment_score
Weighted aggregate based on combat power:
```
equipment_score = tank×5 + artillery×4 + missile×4 + radar×3 + infantry×2
```

Example:
- 30 tanks, 8 artillery, 100 infantry
- Score = 30×5 + 8×4 + 100×2 = 150 + 32 + 200 = 382

### dominant_equipment_type
The equipment category with highest count:
```
0 = infantry (most common)
1 = tank (armored units)
2 = artillery (fire support)
3 = radar (reconnaissance)
4 = missile (precision strike)
```

## Generating Datasets

To generate realistic training and test datasets:

```bash
python generate_realistic_data.py
```

This creates:
- `orbat_training.csv` - 600 samples from 15 units (for training)
- `orbat_test.csv` - 25 observations (for testing predictions)

## Using Your Own Data

### Preparing Training Data

1. Collect unit observations with the 6 features
2. Calculate `equipment_score` and `dominant_equipment_type` from raw equipment data
3. Save as CSV with required columns
4. Ensure at least 30-40 observations per unit

### Preparing Test Data

1. Collect new observations you want to classify
2. Calculate the same 6 features
3. Add `observation_id` column for tracking
4. Add empty `expected_unit` column
5. Save as CSV

## Example Workflow

```bash
# 1. Generate sample data (or use your own)
python generate_realistic_data.py

# 2. Train model
python train.py --data data/orbat_training.csv

# 3. Make predictions
python predict.py --model models/orbat_predictor.pkl \
  --input data/orbat_test.csv \
  --output data/predictions.csv

# 4. View results
cat data/predictions.csv
```

## Data Quality Tips

1. **Consistent units** - Use same measurement units throughout
2. **Realistic ranges** - Keep values within typical military ranges
3. **Location accuracy** - GPS coordinates should be accurate to 4 decimal places
4. **Equipment scores** - Verify calculations are correct
5. **Balanced data** - Similar number of observations per unit

For detailed feature specifications, see [SIMPLIFIED_FEATURES.md](../SIMPLIFIED_FEATURES.md)
