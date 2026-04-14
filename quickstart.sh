#!/bin/bash

# ORBAT Classification System - Quick Start Script

set -e

echo "=========================================="
echo "ORBAT Classification System - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install -r requirements.txt -q
echo "  Dependencies installed successfully"

# Generate sample data
echo ""
echo "[3/5] Generating realistic ORBAT datasets..."
python generate_realistic_data.py
echo "  Training data: data/orbat_training.csv"
echo "  Test data: data/orbat_test.csv"

# Train model
echo ""
echo "[4/5] Training ORBAT classification system..."
python train.py --data data/orbat_training.csv
echo "  Model training complete"

# Run example predictions
echo ""
echo "[5/5] Running example predictions..."
python example_usage.py

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - View models in: models/"
echo "  - Run tests: python run_tests.py"
echo "  - Make predictions: python predict.py --help"
echo "  - Open notebook: jupyter notebook notebooks/training_pipeline.ipynb"
echo ""
