# ORBAT Classification - Military Intelligence Features

## Overview

The system uses **6 realistic military intelligence features** that are commonly available in real-world ORBAT data. Each feature captures specific military signals that help identify and classify units.

## ✅ Features (6 Total)

### Input Features (What You Provide)

| # | Feature | Type | Description | Range | Military Signal |
|---|---------|------|-------------|-------|-----------------|
| 1 | **personnel_count** | Numerical | Total number of soldiers | 50-1000+ | Unit scale/strength, hierarchy level |
| 2 | **latitude** | Numerical | Geographical latitude | 25-55° | Operational zone, deployment area |
| 3 | **longitude** | Numerical | Geographical longitude | 35-85° | Operational zone, deployment area |
| 4 | **equipment_score** | Numerical | Weighted combat power | 50-500+ | Combat capability, firepower intensity |
| 5 | **total_equipment_count** | Numerical | Total equipment items | 10-200+ | Resource volume, unit size |
| 6 | **dominant_equipment_type** | Categorical | Primary equipment category | 0-4 | Unit specialization/role |

### Target Variable (What You Want to Predict)

| Feature | Type | Description |
|---------|------|-------------|
| **unit_id** | Target | Unit identifier to predict | UNIT_001, UNIT_002, etc. |

---

## 📊 Feature Details

### 1. personnel_count
**What it is:** Total number of soldiers in the unit

**Signal captured:**
- Unit scale / strength
- Helps distinguish hierarchy levels (platoon vs battalion)

**Why it matters:** Larger formations (e.g., battalions) have significantly higher manpower than smaller ones (sections, platoons).

**Typical ranges:**
- Platoon: 30-50
- Company: 100-200
- Battalion: 400-800
- Brigade: 3000-5000

---

### 2. latitude & longitude
**What they are:** Geographical coordinates of the unit

**Signal captured:**
- Operational zone / deployment area
- Spatial clustering of units

**Why it matters:** Units are usually deployed in specific regions, so location acts as a strong contextual feature.

**Note:** Risk of overfitting if units don't move, but still very useful in ORBAT mapping.

**Typical ranges:**
- Latitude: 25-55° (realistic operational zones)
- Longitude: 35-85°

---

### 3. equipment_score
**What it is:** A weighted aggregate score of all equipment

**Calculation example:**
```
equipment_score = tank×5 + artillery×4 + missile×4 + radar×3 + infantry×2
```

**Weights reflect combat effectiveness:**
- Tank: 5 (heavy armor, high firepower)
- Artillery: 4 (long-range fire support)
- Missile: 4 (precision strike capability)
- Radar: 3 (force multiplier, situational awareness)
- Infantry: 2 (basic combat unit)

**Signal captured:**
- Combat power / capability
- Firepower intensity

**Why it matters:** Different unit types have distinct combat strength profiles.

---

### 4. total_equipment_count
**What it is:** Total number of all equipment items

**Calculation:**
```
total_equipment_count = sum of all equipment
```

**Signal captured:**
- Resource volume
- Unit size (indirectly)

**Why it matters:** Helps differentiate:
- Light units (low count: 10-30)
- Heavy units (high count: 100-200)

---

### 5. dominant_equipment_type
**What it is:** The most prominent equipment category in the unit

**Encoding:**
```
0 = infantry
1 = tank
2 = artillery
3 = radar
4 = missile
```

**Signal captured:**
- Unit specialization / role

**Why it matters:** Two units can have same score but:
- One is tank-heavy (armored unit)
- Other is radar-heavy (reconnaissance unit)

👉 This feature resolves that ambiguity.

---

## 📋 Dataset Files

### 1. Training Dataset (`orbat_training.csv`)

**Purpose:** Train the model to recognize unit patterns

**Format:**
```csv
personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,unit_id
600,45.2345,67.8901,250,50,1,UNIT_001
500,42.5000,75.0000,200,45,2,UNIT_002
300,50.0000,70.0000,120,30,3,UNIT_003
```

**Generated Data:**
- 15 unique units (5 archetypes × 3 instances)
- 40 observations per unit
- 600 total samples
- Realistic unit archetypes:
  - Heavy Armor (tank-heavy)
  - Artillery Battalion
  - Infantry Battalion
  - Missile Battalion
  - Reconnaissance (radar-heavy)

---

### 2. Test Dataset (`orbat_test.csv`)

**Purpose:** Test predictions on new, unseen observations

**Format:**
```csv
observation_id,personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,expected_unit
OBS_001,600,45.5000,68.0000,280,55,1,
OBS_002,300,51.2000,71.3000,120,30,3,
```

**Generated Data:**
- 25 test observations
- `observation_id` for tracking
- `expected_unit` column (empty, to be filled with predictions)

---

## 🚀 Quick Start

### Step 1: Generate Datasets

```bash
python generate_realistic_data.py
```

**Output:**
- `data/orbat_training.csv` (600 samples, 15 units)
- `data/orbat_test.csv` (25 observations)

### Step 2: Train the Model

```bash
python train.py --data data/orbat_training.csv
```

### Step 3: Make Predictions

**Batch prediction on test file:**
```bash
python predict.py --model models/orbat_predictor.pkl \
  --input data/orbat_test.csv \
  --output data/predictions.csv
```

**Single prediction:**
```bash
python predict.py --model models/orbat_predictor.pkl \
  --personnel-count 600 \
  --latitude 45.5 \
  --longitude 68.0 \
  --equipment-score 250 \
  --total-equipment 50 \
  --dominant-equipment 1
```

---

## 📝 Using Your Own Data

### Training Data Format

Create a CSV file with these columns:

```csv
personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,unit_id
600,45.2345,67.8901,250,50,1,UNIT_001
600,45.3456,67.9012,260,52,1,UNIT_001
500,42.5000,75.0000,200,45,2,UNIT_002
```

**Requirements:**
- At least 30-40 observations per unit (more is better)
- Multiple units (at least 10-15 different units)
- Consistent feature values

**How to calculate features from raw data:**

If you have raw equipment counts:
```python
# Example: Unit has 30 tanks, 8 artillery, 100 infantry
equipment_score = 30*5 + 8*4 + 100*2 = 150 + 32 + 200 = 382
total_equipment_count = 30 + 8 + 100 = 138
dominant_equipment_type = 1  # tank (highest count)
```

---

### Test Data Format

Create a CSV file with observations you want to classify:

```csv
observation_id,personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,expected_unit
OBS_001,600,45.5,68.0,280,55,1,
OBS_002,300,51.2,71.3,120,30,3,
```

**Notes:**
- `observation_id` is optional but helpful for tracking
- `expected_unit` column is optional (leave empty for predictions)

---

## 🎯 Prediction Output

After running predictions, you get a CSV with:

```csv
observation_id,personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,predicted_unit,confidence_score,hierarchy
OBS_001,600,45.5,68.0,280,55,1,UNIT_001,0.87,Battalion
OBS_002,300,51.2,71.3,120,30,3,UNIT_003,0.92,Reconnaissance
```

**New columns added:**
- `predicted_unit` - The predicted unit identifier
- `confidence_score` - Prediction confidence (0.0-1.0)
- `hierarchy` - Unit hierarchy level (if available)

---

## 💡 Tips for Best Results

1. **More training data is better** - Aim for 40-50 observations per unit
2. **Consistent patterns** - Units should have characteristic equipment/locations
3. **Balanced data** - Similar number of observations for each unit
4. **Realistic features** - Use actual intelligence data when available
5. **Location variation** - Include some movement (±0.5° typical)
6. **Equipment diversity** - Different units should have distinct equipment profiles

---

## 🔧 Feature Engineering Guide

### Calculating equipment_score

If you have raw equipment data:

```python
def calculate_equipment_score(equipment_dict):
    """
    equipment_dict = {
        'tank': 30,
        'artillery': 8,
        'infantry': 100,
        'radar': 4,
        'missile': 0
    }
    """
    weights = {
        'tank': 5,
        'artillery': 4,
        'missile': 4,
        'radar': 3,
        'infantry': 2
    }
    
    score = sum(equipment_dict.get(eq, 0) * weights.get(eq, 0) 
                for eq in weights.keys())
    return score
```

### Determining dominant_equipment_type

```python
def get_dominant_equipment(equipment_dict):
    """Returns 0-4 based on highest count"""
    if not equipment_dict:
        return 0
    
    dominant = max(equipment_dict.items(), key=lambda x: x[1])[0]
    
    mapping = {
        'infantry': 0,
        'tank': 1,
        'artillery': 2,
        'radar': 3,
        'missile': 4
    }
    
    return mapping.get(dominant, 0)
```

---

## 📊 Example Workflow

```bash
# 1. Generate sample data (or use your own)
python generate_realistic_data.py

# 2. Train the model
python train.py --data data/orbat_training.csv

# 3. Make predictions on test data
python predict.py --model models/orbat_predictor.pkl \
  --input data/orbat_test.csv \
  --output data/predictions.csv

# 4. View results
head -20 data/predictions.csv
```

---

## ✅ Benefits of These Features

1. **Realistic** - All features commonly available in military intelligence
2. **Interpretable** - Clear military meaning for each feature
3. **Effective** - Captures key unit characteristics
4. **Scalable** - Works with various unit types and sizes
5. **Robust** - Handles missing data and variations
6. **Actionable** - Easy to collect from field reports

---

## 🎓 Unit Archetypes (Reference)

| Archetype | Personnel | Equipment Score | Total Equipment | Dominant Type |
|-----------|-----------|-----------------|-----------------|---------------|
| Heavy Armor | 600 | 250-300 | 140-160 | 1 (tank) |
| Artillery | 500 | 180-220 | 110-130 | 2 (artillery) |
| Infantry | 800 | 300-350 | 160-180 | 0 (infantry) |
| Missile | 400 | 150-180 | 80-100 | 4 (missile) |
| Reconnaissance | 300 | 100-140 | 60-80 | 3 (radar) |

---

**Ready to use your own data?** Just create a CSV file with the 6 required features and run the training!
