# ORBAT Classification System

A machine learning system for military unit identification based on **resource levels** with **location-based tiebreaking**.

## Core Concept

1. **Primary Identification**: `equipment_score` (high weight)
   - Units are identified mainly by their resource/equipment levels
   
2. **Tiebreaker**: `latitude` + `longitude` (low weight)  
   - When multiple units have similar equipment scores
   - Choose the geographically closest unit

## Features

- **Resource-Focused**: Primary identification by equipment score
- **Location Tiebreaker**: Geographic distance for similar resources  
- **Large Dataset**: 5000 samples (50 units × 100 observations each)
- **Realistic Military Names**: Troop1, Squadron2, Regiment3, etc.
- **Noise Resistant**: Built-in noise to prevent overfitting
- **Simple & Fast**: Only 3 essential features

## Quick Start

**New to the system?** See **[GETTING_STARTED.md](GETTING_STARTED.md)** for a detailed walkthrough.

### Automated Setup

```bash
chmod +x quickstart.sh
./quickstart.sh
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate large realistic dataset
python generate_realistic_data.py

# Train the system
python train.py --data data/orbat_training.csv

# Make predictions on test data
python predict.py --model models/orbat_predictor.pkl \
  --input data/orbat_test.csv \
  --output data/predictions.csv
```

### Make Predictions

**Batch Prediction (CSV file):**
```bash
python predict.py \
  --model models/orbat_predictor.pkl \
  --input data/orbat_test.csv \
  --output data/predictions.csv
```

**Single Prediction:**
```bash
python predict.py \
  --model models/orbat_predictor.pkl \
  --equipment-score 250 \
  --latitude 45.5 \
  --longitude 68.0
```

**Python API:**
```python
from src.inference import ORBATPredictor

# Load trained predictor
predictor = ORBATPredictor.load('models/orbat_predictor.pkl')

# Predict (only 3 features needed)
observation = {
    'equipment_score': 250,  # Primary identifier
    'latitude': 45.2345,     # Tiebreaker
    'longitude': 67.8901     # Tiebreaker
}

result = predictor.predict(observation)
print(f"Unit: {result['predicted_unit']}")
print(f"Confidence: {result['confidence_score']:.3f}")
```

## Project Structure

```
├── data/                       # Dataset directory
├── src/
│   ├── preprocessing.py        # Data cleaning and feature engineering
│   ├── models.py               # Classification and similarity models
│   ├── hybrid_system.py        # Hybrid prediction system
│   ├── evaluation.py           # Metrics and evaluation
│   └── inference.py            # Prediction interface
├── tests/                      # Unit tests
├── notebooks/
│   └── training_pipeline.ipynb # Interactive training notebook
├── train.py                    # Training script
├── predict.py                  # Prediction script
├── example_usage.py            # Usage examples
├── generate_sample_data.py     # Sample data generator
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── USAGE_GUIDE.md             # Detailed usage guide
└── ARCHITECTURE.md            # System architecture documentation
```

## Dataset Format

### Training Data (CSV)
```csv
equipment_score,latitude,longitude,unit_name
250,45.2345,67.8901,Troop1
180,42.5000,75.0000,Squadron2
400,50.0000,70.0000,Regiment3
```

**Features:**
- `equipment_score`: Resource level (primary identifier, high weight)
- `latitude`, `longitude`: GPS coordinates (tiebreaker, low weight)
- `unit_name`: Target variable (realistic military names)

### Test Data (CSV)
```csv
observation_id,equipment_score,latitude,longitude,predicted_unit
OBS_001,250,45.5,68.0,
OBS_002,180,51.2,71.3,
```

## System Architecture

### Hybrid Prediction Pipeline

```
Input Observation
       ↓
Data Preprocessing (encoding, scaling)
       ↓
Classification Model (XGBoost/LightGBM)
       ↓
Top-K Candidates (e.g., top 3 units)
       ↓
Similarity Matching (cosine/euclidean)
       ↓
Score Combination (α × classification + (1-α) × similarity)
       ↓
Final Prediction + Confidence Score
```

### Key Components

1. **Preprocessing**: Handles missing values, encodes categorical features, normalizes numerical features
2. **Classification Model**: Multi-class gradient boosting (XGBoost or LightGBM)
3. **Similarity Model**: Embedding-based matching with KNN
4. **Hybrid System**: Combines both approaches with configurable weights
5. **Evaluation**: Comprehensive metrics including top-K accuracy and confidence calibration

## Training Configuration

```bash
python train.py \
  --data data/orbat_dataset.csv \
  --output models \
  --test-size 0.2 \
  --val-size 0.1 \
  --model xgboost \
  --similarity cosine \
  --alpha 0.6 \
  --seed 42
```

**Parameters:**
- `--model`: xgboost or lightgbm
- `--similarity`: cosine or euclidean
- `--alpha`: Weight for classification (0.0-1.0, default: 0.6)

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Top-K Accuracy**: Accuracy when considering top-K predictions
- **Confusion Matrix**: Per-class performance visualization
- **Confidence Calibration**: Expected Calibration Error (ECE)
- **Per-Class Metrics**: Precision, recall, F1-score

## Testing

Run the test suite:

```bash
python run_tests.py
```

Or with pytest directly:

```bash
pytest tests/ -v
```

## Documentation

- **README.md**: This file - core concept and usage
- **notebooks/training_pipeline.ipynb**: Interactive training walkthrough

## Requirements

- Python 3.7+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Advanced Features

### Custom Model Configuration

```python
from src.models import ClassificationModel

clf_model = ClassificationModel(
    model_type='xgboost',
    max_depth=8,
    learning_rate=0.05,
    n_estimators=300
)
```

### Adjusting Hybrid Weights

```python
from src.hybrid_system import HybridORBATSystem

# More weight on classification
hybrid = HybridORBATSystem(clf_model, sim_model, alpha=0.7)

# More weight on similarity
hybrid = HybridORBATSystem(clf_model, sim_model, alpha=0.4)
```

### Confidence Thresholding

```python
result = predictor.predict(observation)

if result['confidence_score'] >= 0.8:
    print("High confidence prediction")
elif result['confidence_score'] >= 0.6:
    print("Medium confidence - review recommended")
else:
    print("Low confidence - manual verification required")
```

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.
