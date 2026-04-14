# ORBAT Classification System

An intelligent machine learning system for classifying military units (Order of Battle) using hybrid classification and similarity-based approaches.

## Features

- **Military Intelligence Features**: Uses 6 realistic features commonly available in ORBAT data
  - Personnel count, GPS coordinates, equipment score, total equipment, dominant equipment type
  - Based on actual military intelligence collection methods
- **Dual Model Architecture**: XGBoost/LightGBM classifier + Similarity matching
- **Hybrid Prediction**: Combines classification probabilities with similarity scores
- **Confidence Scoring**: Calibrated confidence metrics with ECE
- **CSV-Based Workflow**: Easy data upload and batch predictions
- **Production-Ready**: Comprehensive testing and documentation

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

# Generate realistic sample data
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
  --personnel-count 600 \
  --latitude 45.5 \
  --longitude 68.0 \
  --equipment-score 250 \
  --total-equipment 50 \
  --dominant-equipment 1
```

**Python API:**
```python
from src.inference import ORBATPredictor

# Load trained predictor
predictor = ORBATPredictor.load('models/orbat_predictor.pkl')

# Predict
observation = {
    'personnel_count': 600,
    'latitude': 45.2345,
    'longitude': 67.8901,
    'equipment_score': 250,
    'total_equipment_count': 50,
    'dominant_equipment_type': 1  # tank
}

result = predictor.predict(observation, return_details=True)
print(result)
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
personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,unit_id
600,45.2345,67.8901,250,50,1,UNIT_001
500,42.5000,75.0000,200,45,2,UNIT_002
```

**Features:**
- `personnel_count`: Total soldiers (50-1000+)
- `latitude`, `longitude`: GPS coordinates
- `equipment_score`: Weighted combat power (tank×5 + artillery×4 + missile×4 + radar×3 + infantry×2)
- `total_equipment_count`: Total equipment items
- `dominant_equipment_type`: Primary equipment (0=infantry, 1=tank, 2=artillery, 3=radar, 4=missile)
- `unit_id`: Target variable (what to predict)

### Test Data (CSV)
```csv
observation_id,personnel_count,latitude,longitude,equipment_score,total_equipment_count,dominant_equipment_type,expected_unit
OBS_001,600,45.5,68.0,280,55,1,
OBS_002,300,51.2,71.3,120,30,3,
```

**See [SIMPLIFIED_FEATURES.md](SIMPLIFIED_FEATURES.md) for detailed feature specification.**

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

- **[SIMPLIFIED_FEATURES.md](SIMPLIFIED_FEATURES.md)**: Feature specification and data format
- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Step-by-step tutorial for new users
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Comprehensive usage guide with examples
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and design decisions
- **[notebooks/training_pipeline.ipynb](notebooks/training_pipeline.ipynb)**: Interactive training walkthrough

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
