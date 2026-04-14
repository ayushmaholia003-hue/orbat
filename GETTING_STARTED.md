# Getting Started with ORBAT Classification System

Welcome! This guide will help you get up and running with the ORBAT Classification System in minutes.

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
1. Install the system
2. Generate sample data
3. Train a model
4. Make predictions
5. Evaluate results

**Time Required:** 15-30 minutes

---

## 📋 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- 2GB free disk space
- Basic command line knowledge

**Check your Python version:**
```bash
python --version
```

---

## 🚀 Quick Start (Automated)

The fastest way to get started:

```bash
# Make the script executable
chmod +x quickstart.sh

# Run the complete setup
./quickstart.sh
```

This will:
1. Install all dependencies
2. Generate sample data
3. Train a model
4. Run example predictions

**That's it!** Your system is ready to use.

---

## 📝 Step-by-Step Setup (Manual)

If you prefer to understand each step:

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**What this installs:**
- XGBoost & LightGBM (ML models)
- scikit-learn (preprocessing & metrics)
- pandas & numpy (data handling)
- matplotlib & seaborn (visualization)

### Step 2: Generate Sample Data

```bash
python generate_sample_data.py
```

**Output:** `data/orbat_sample.csv`
- 20 unique military units
- 50 observations per unit
- 1,000 total samples

**View the data:**
```bash
head -n 5 data/orbat_sample.csv
```

### Step 3: Train the Model

```bash
python train.py --data data/orbat_sample.csv
```

**What happens:**
- Data is split into train/val/test sets
- XGBoost classifier is trained
- Similarity model is built
- Hybrid system is created
- Model is evaluated on test set
- Everything is saved to `models/`

**Training time:** ~10-30 seconds

**Output files:**
```
models/
├── orbat_predictor.pkl          # Main predictor (use this!)
├── preprocessor.pkl
├── classification_model.pkl
├── similarity_model.pkl
├── evaluation_metrics.json
└── config.json
```

### Step 4: Make Predictions

**Single prediction:**
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

**Expected output:**
```
============================================================
ORBAT PREDICTION RESULT
============================================================

Predicted Unit: UNIT_001
Hierarchy: Battalion
Confidence Score: 0.8542

Detailed Analysis:
  Candidate Units: ['UNIT_001', 'UNIT_002', 'UNIT_003']
  Classification Probs: [0.4500, 0.3000, 0.1500]
  Similarity Scores: [0.9200, 0.7800, 0.6500]
  Combined Scores: [0.6500, 0.5100, 0.3700]
============================================================
```

### Step 5: Run Tests

```bash
python run_tests.py
```

**What this tests:**
- Data preprocessing
- Model training and prediction
- Hybrid system
- Confidence scoring

---

## 💻 Using the Python API

Create a file `my_prediction.py`:

```python
from src.inference import ORBATPredictor

# Load the trained model
predictor = ORBATPredictor.load('models/orbat_predictor.pkl')

# Define an observation
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

# Make prediction
result = predictor.predict(observation, return_details=True)

# Print results
print(f"Predicted Unit: {result['predicted_unit']}")
print(f"Hierarchy: {result['hierarchy']}")
print(f"Confidence: {result['confidence_score']:.4f}")

# Access detailed breakdown
if 'details' in result:
    print(f"\nTop Candidates: {result['details']['candidate_units']}")
    print(f"Combined Scores: {result['details']['combined_scores']}")
```

Run it:
```bash
python my_prediction.py
```

---

## 📊 Interactive Tutorial

For a hands-on learning experience:

```bash
jupyter notebook notebooks/training_pipeline.ipynb
```

This notebook includes:
- Data exploration
- Step-by-step training
- Visualization
- Interactive predictions

---

## 🎓 Next Steps

### Learn More
1. **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage instructions
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - How the system works
3. **[example_usage.py](example_usage.py)** - More code examples

### Use Your Own Data
1. Prepare your CSV file (see [data/README.md](data/README.md))
2. Train: `python train.py --data your_data.csv`
3. Predict: `python predict.py --model models/orbat_predictor.pkl ...`

### Customize the System
1. Adjust hyperparameters in `train.py`
2. Try different models (XGBoost vs LightGBM)
3. Tune the alpha parameter (classification vs similarity weight)
4. Add new features to preprocessing

---

## 🔧 Common Commands

```bash
# Using Makefile (recommended)
make install      # Install dependencies
make sample-data  # Generate sample data
make train        # Train model
make test         # Run tests
make predict      # Run example predictions
make clean        # Clean generated files
make all          # Complete pipeline

# Or use individual scripts
python generate_sample_data.py
python train.py --data data/orbat_sample.csv
python predict.py --model models/orbat_predictor.pkl ...
python run_tests.py
python example_usage.py
```

---

## ❓ Troubleshooting

### Issue: Import errors
**Solution:** Make sure you're in the project root directory and dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Model file not found
**Solution:** Train the model first:
```bash
python train.py --data data/orbat_sample.csv
```

### Issue: Data file not found
**Solution:** Generate sample data:
```bash
python generate_sample_data.py
```

### Issue: Tests failing
**Solution:** Ensure all dependencies are installed and you're using Python 3.7+:
```bash
python --version
pip install -r requirements.txt
python run_tests.py
```

---

## 📚 Documentation Index

- **[README.md](README.md)** - Project overview
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - This file
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Feature summary
- **[SYSTEM_FLOW.md](SYSTEM_FLOW.md)** - Visual diagrams
- **[INDEX.md](INDEX.md)** - Complete documentation index
- **[VERIFICATION_CHECKLIST.md](VERIFICATION_CHECKLIST.md)** - Requirements verification

---

## 🎉 Success!

You now have a working ORBAT classification system! 

**What you can do:**
- ✅ Train models on military unit data
- ✅ Make predictions with confidence scores
- ✅ Evaluate model performance
- ✅ Use both CLI and Python API
- ✅ Extend and customize the system

**Need help?** Check the [USAGE_GUIDE.md](USAGE_GUIDE.md) or [INDEX.md](INDEX.md) for more information.

---

## 🚀 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    QUICK REFERENCE                           │
├─────────────────────────────────────────────────────────────┤
│ Setup:        ./quickstart.sh                               │
│ Train:        python train.py --data data/orbat_sample.csv  │
│ Predict:      python predict.py --model models/...          │
│ Test:         python run_tests.py                           │
│ Examples:     python example_usage.py                       │
│ Notebook:     jupyter notebook notebooks/...                │
│ Clean:        make clean                                    │
│ Help:         python train.py --help                        │
└─────────────────────────────────────────────────────────────┘
```

Happy classifying! 🎯
