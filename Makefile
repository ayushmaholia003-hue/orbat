# Makefile for ORBAT Classification System

.PHONY: help install test train predict clean sample-data

help:
	@echo "ORBAT Classification System - Available Commands"
	@echo "================================================"
	@echo "make install      - Install dependencies"
	@echo "make sample-data  - Generate sample dataset"
	@echo "make train        - Train model on sample data"
	@echo "make test         - Run test suite"
	@echo "make predict      - Run example predictions"
	@echo "make clean        - Clean generated files"
	@echo "make notebook     - Start Jupyter notebook"

install:
	pip install -r requirements.txt

sample-data:
	python generate_realistic_data.py

train: sample-data
	python train.py --data data/orbat_training.csv

test:
	python run_tests.py

predict:
	python example_usage.py

clean:
	rm -rf models/
	rm -rf data/*.csv
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	find . -name "*.pyc" -delete

notebook:
	jupyter notebook notebooks/training_pipeline.ipynb

all: install sample-data train test
	@echo "Complete pipeline executed successfully!"
