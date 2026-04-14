# ORBAT Classification System - Usage Guide

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd orbat-classification

# Install dependencies
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python generate_sample_data.py
```

This creates `data/orbat_sample.csv` with 20 units and 50 observations per unit.

### Train the System

```bash
python train.py --data data/orbat_sample.csv
```

This will:
- Train classification and similarity models
- Evaluate on test set
- Save models to `models/` directory

### Make Predictions

**Single Prediction:**
```bash
python predict.py \
  --model models/orbat_predictor.pkl \
  --equipment-type tank \
  --equipment-count 12 \
  --comm-freq 150.5 \
  --mobility mobile \
  --hierarchy Battalion \
  --comm-degree 6 \
  --location-x 450.2 \
  --location-y 320.8 \
  --cluster-id 2
```

**Batch Prediction:**
```bash
python predict.py \
  --model models/orbat_predictor.pkl \
  --input data/test_observations.json \
  --output results/predictions.json
```

## Detailed Usage

### 1. Data Preparation

Your dataset should be a CSV file with the following columns:

**Required Columns:**
- `equipment_type` (categorical): Type of equipment (tank, radar, artillery, etc.)
- `equipment_count` (numerical): Number of equipment units
- `communication_frequency` (numerical): Communication frequency in MHz
- `mobility` (categorical): static or mobile
- `hierarchy_level` (categorical): HQ, Regiment, Battalion, Company, Section
- `communication_degree` (numerical): Number of communication connections
- `location_x` (numerical): X coordinate
- `location_y` (numerical): Y coordinate
- `cluster_id` (numerical): Precomputed cluster identifier
- `unit_id` (target): Unit identifier (e.g., UNIT_001)

**Example CSV:**
```csv
equipment_type,equipment_count,communication_frequency,mobility,hierarchy_level,communication_degree,location_x,location_y,cluster_id,unit_id
tank,10,150.0,mobile,Battalion,5,500.0,300.0,2,UNIT_001
radar,3,200.0,static,HQ,10,100.0,200.0,0,UNIT_002
```

### 2. Training Configuration

**Basic Training:**
```bash
python train.py --data data/orbat_dataset.csv
```

**Advanced Training:**
```bash
python train.py \
  --data data/orbat_dataset.csv \
  --output models_v2 \
  --test-size 0.2 \
  --val-size 0.1 \
  --model xgboost \
  --similarity cosine \
  --alpha 0.6 \
  --seed 42
```

**Parameters:**
- `--data`: Path to CSV dataset (required)
- `--output`: Output directory for models (default: models)
- `--test-size`: Test set proportion (default: 0.2)
- `--val-size`: Validation set proportion (default: 0.1)
- `--model`: Classification model (xgboost or lightgbm, default: xgboost)
- `--similarity`: Similarity metric (cosine or euclidean, default: cosine)
- `--alpha`: Classification weight in hybrid system (default: 0.6)
- `--seed`: Random seed (default: 42)

### 3. Using the Python API

**Load and Predict:**
```python
from src.inference import ORBATPredictor

# Load trained predictor
predictor = ORBATPredictor.load('models/orbat_predictor.pkl')

# Single prediction
observation = {
    'equipment_type': 'tank',
    'equipment_count': 12,
    'communication_frequency': 150.5,
    'mobility': 'mobile',
    'hierarchy_level': 'Battalion',
    'communication_degree': 6,
    'location_x': 450.2,
    'location_y': 320.8,
    'cluster_id': 2
}

result = predictor.predict(observation, return_details=True)
print(f"Predicted Unit: {result['predicted_unit']}")
print(f"Confidence: {result['confidence_score']:.4f}")
```

**Batch Prediction:**
```python
observations = [
    {...},  # observation 1
    {...},  # observation 2
]

results = predictor.predict_batch(observations)
for result in results:
    print(f"{result['predicted_unit']}: {result['confidence_score']:.4f}")
```

### 4. Jupyter Notebook

Open and run the training notebook:
```bash
jupyter notebook notebooks/training_pipeline.ipynb
```

The notebook includes:
- Data exploration
- Step-by-step training
- Evaluation and visualization
- Interactive prediction testing

### 5. Evaluation and Metrics

After training, check the evaluation metrics:

```python
import json

# Load metrics
with open('models/evaluation_metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
```

**Visualize Results:**
```python
from src.evaluation import ORBATEvaluator
import joblib

# Load evaluator with saved metrics
evaluator = ORBATEvaluator()
evaluator.metrics = metrics

# Plot confusion matrix
evaluator.plot_confusion_matrix()

# Plot confidence distribution
evaluator.plot_confidence_distribution(confidences, y_true, y_pred)
```

## Advanced Usage

### Custom Model Configuration

**XGBoost with Custom Parameters:**
```python
from src.models import ClassificationModel

clf_model = ClassificationModel(
    model_type='xgboost',
    max_depth=8,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.9,
    colsample_bytree=0.9
)
```

**LightGBM Configuration:**
```python
clf_model = ClassificationModel(
    model_type='lightgbm',
    num_leaves=50,
    learning_rate=0.05,
    n_estimators=300
)
```

### Adjusting Hybrid System Weights

The `alpha` parameter controls the balance between classification and similarity:

- `alpha=1.0`: Pure classification (no similarity)
- `alpha=0.6`: Default (60% classification, 40% similarity)
- `alpha=0.5`: Equal weight
- `alpha=0.0`: Pure similarity (no classification)

**Experiment with different alphas:**
```python
from src.hybrid_system import HybridORBATSystem

# More weight on classification
hybrid_system = HybridORBATSystem(clf_model, sim_model, alpha=0.7)

# More weight on similarity
hybrid_system = HybridORBATSystem(clf_model, sim_model, alpha=0.4)
```

### Confidence Thresholding

Filter predictions by confidence:

```python
result = predictor.predict(observation)

if result['confidence_score'] >= 0.8:
    print(f"High confidence: {result['predicted_unit']}")
elif result['confidence_score'] >= 0.6:
    print(f"Medium confidence: {result['predicted_unit']}")
else:
    print(f"Low confidence - manual review recommended")
```

### Handling Unseen Features

The system handles unseen categorical values by mapping them to the most common category:

```python
# This will work even if 'drone' wasn't in training data
observation = {
    'equipment_type': 'drone',  # Unseen category
    # ... other features
}

result = predictor.predict(observation)
# System maps 'drone' to most common equipment type
```

## Troubleshooting

### Issue: Low Accuracy

**Solutions:**
1. Increase training data
2. Tune hyperparameters
3. Adjust alpha parameter
4. Try different model (xgboost vs lightgbm)
5. Add more features

### Issue: Poor Confidence Calibration

**Solutions:**
1. Increase validation set size
2. Adjust alpha parameter
3. Use confidence thresholding
4. Retrain with more diverse data

### Issue: Slow Inference

**Solutions:**
1. Reduce top-K candidates (default: 3)
2. Use smaller classification model
3. Reduce KNN neighbors
4. Batch predictions instead of single

### Issue: Memory Errors

**Solutions:**
1. Reduce training data size
2. Use LightGBM instead of XGBoost
3. Reduce max_depth parameter
4. Process in smaller batches

## Best Practices

1. **Data Quality:** Ensure clean, consistent data
2. **Feature Engineering:** Add domain-specific features
3. **Validation:** Always use separate validation set
4. **Monitoring:** Track confidence scores in production
5. **Retraining:** Periodically retrain with new data
6. **Versioning:** Save model versions with metadata
7. **Testing:** Test on diverse scenarios before deployment

## Example Workflows

### Workflow 1: Initial Development
```bash
# Generate sample data
python generate_sample_data.py

# Train with defaults
python train.py --data data/orbat_sample.csv

# Test predictions
python example_usage.py
```

### Workflow 2: Production Training
```bash
# Train with production data
python train.py \
  --data data/production_orbat.csv \
  --output models/production_v1 \
  --model xgboost \
  --alpha 0.65 \
  --seed 42

# Evaluate
python -c "
from src.evaluation import ORBATEvaluator
import json
with open('models/production_v1/evaluation_metrics.json') as f:
    metrics = json.load(f)
print(f'Accuracy: {metrics[\"accuracy\"]:.4f}')
"
```

### Workflow 3: Batch Processing
```bash
# Prepare input file (JSON array of observations)
cat > batch_input.json << EOF
[
  {"equipment_type": "tank", "equipment_count": 10, ...},
  {"equipment_type": "radar", "equipment_count": 3, ...}
]
EOF

# Run batch prediction
python predict.py \
  --model models/orbat_predictor.pkl \
  --input batch_input.json \
  --output batch_results.json

# View results
cat batch_results.json
```
