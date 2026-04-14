# ORBAT Classification System - Architecture

## System Overview

The ORBAT (Order of Battle) Classification System is a hybrid machine learning system that combines gradient boosting classification with similarity-based matching to predict military unit identifications from observational data.

## Architecture Components

### 1. Data Preprocessing (`src/preprocessing.py`)

**ORBATPreprocessor**
- Handles missing values (median for numerical, mode for categorical)
- Label encoding for categorical features
- Standard scaling for numerical features
- Maintains fitted encoders for consistent transformation
- Prevents data leakage between train/test sets

**Features Processed:**
- Categorical: equipment_type, mobility, hierarchy_level
- Numerical: equipment_count, communication_frequency, communication_degree, location_x, location_y, cluster_id

### 2. Classification Model (`src/models.py`)

**ClassificationModel**
- Supports XGBoost and LightGBM
- Multi-class classification with softmax probabilities
- Top-K prediction capability
- Hyperparameter tuning support

**Key Methods:**
- `train()`: Fit model with optional validation
- `predict_proba()`: Get class probabilities
- `get_top_k_predictions()`: Return top-K candidates with scores

### 3. Similarity Model (`src/models.py`)

**SimilarityModel**
- Embedding-based similarity matching
- Supports cosine similarity and Euclidean distance
- KNN for efficient nearest neighbor search
- Candidate-specific similarity computation

**Key Methods:**
- `fit()`: Store reference embeddings
- `find_similar()`: Find K most similar units
- `compute_similarity_to_candidates()`: Score specific candidates

### 4. Hybrid System (`src/hybrid_system.py`)

**HybridORBATSystem**
- Combines classification and similarity approaches
- Two-stage prediction pipeline
- Confidence calibration

**Prediction Pipeline:**
```
Input → Classification Model → Top-K Candidates
                                      ↓
                              Similarity Matching
                                      ↓
                              Score Combination
                                      ↓
                              Final Prediction + Confidence
```

**Score Combination:**
```
combined_score = α × classification_prob + (1-α) × similarity_score
```

Default α = 0.6 (60% classification, 40% similarity)

### 5. Evaluation (`src/evaluation.py`)

**ORBATEvaluator**
- Comprehensive metrics computation
- Visualization tools
- Confidence calibration analysis

**Metrics:**
- Accuracy (overall and top-K)
- Per-class precision, recall, F1
- Confusion matrix
- Confidence calibration (ECE)
- Accuracy by confidence bins

### 6. Inference (`src/inference.py`)

**ORBATPredictor**
- High-level prediction interface
- Single and batch prediction
- Hierarchy mapping
- Detailed prediction breakdown

**Output Format:**
```json
{
  "predicted_unit": "UNIT_001",
  "hierarchy": "HQ → Regiment → Battalion",
  "confidence_score": 0.87,
  "details": {
    "candidate_units": ["UNIT_001", "UNIT_002", "UNIT_003"],
    "classification_probabilities": [0.45, 0.30, 0.15],
    "similarity_scores": [0.92, 0.78, 0.65],
    "combined_scores": [0.65, 0.51, 0.37]
  }
}
```

## Training Pipeline

### Step-by-Step Process

1. **Data Loading**
   - Load CSV dataset
   - Validate required columns

2. **Data Splitting**
   - Train/Val/Test split (70/10/20 default)
   - Stratified by unit_id to maintain class balance

3. **Preprocessing**
   - Fit preprocessor on training data
   - Transform all splits consistently

4. **Classification Training**
   - Train XGBoost/LightGBM on encoded features
   - Use validation set for early stopping

5. **Similarity Training**
   - Store training embeddings
   - Build KNN index for efficient search

6. **Hybrid System Assembly**
   - Combine both models
   - Calibrate confidence on validation set

7. **Evaluation**
   - Compute metrics on test set
   - Generate visualizations

8. **Model Persistence**
   - Save all components
   - Store configuration and metrics

## Key Design Decisions

### Why Hybrid Approach?

1. **Classification Strengths:**
   - Learns complex feature interactions
   - Handles high-dimensional data well
   - Provides probabilistic outputs

2. **Similarity Strengths:**
   - Robust to unseen feature combinations
   - Interpretable (nearest neighbor logic)
   - Handles edge cases better

3. **Combined Benefits:**
   - Higher accuracy than either alone
   - More robust confidence estimates
   - Better generalization

### Confidence Scoring

The system uses a multi-faceted confidence approach:

1. **Classification Confidence:** Softmax probability of predicted class
2. **Similarity Confidence:** Normalized similarity score
3. **Combined Confidence:** Weighted average (α parameter)
4. **Calibration:** Adjusted based on validation performance

### Scalability Considerations

- **Feature Engineering:** Modular preprocessor for easy extension
- **Model Selection:** Pluggable classification backends
- **Batch Processing:** Vectorized operations for efficiency
- **Memory Management:** Efficient KNN indexing for large reference sets

## Advanced Features

### 1. Top-K Prediction
- Returns multiple candidate units
- Useful for human-in-the-loop systems
- Provides fallback options

### 2. Hierarchical Output
- Maps units to organizational hierarchy
- Provides context for predictions
- Supports multi-level classification

### 3. Confidence Calibration
- Expected Calibration Error (ECE)
- Accuracy by confidence bins
- Threshold-based filtering

### 4. Extensibility
- Easy to add new features
- Pluggable similarity metrics
- Configurable hybrid weights

## Performance Characteristics

### Computational Complexity

- **Training:** O(n × d × log(n)) for gradient boosting
- **Inference (Classification):** O(d × trees)
- **Inference (Similarity):** O(log(n) × d) with KNN index
- **Total Inference:** O(d × trees + k × d) where k is top-K

### Memory Requirements

- **Model Storage:** ~10-100 MB (depends on dataset size)
- **Reference Embeddings:** O(n × d) where n = training samples
- **Runtime:** Minimal overhead for single predictions

## Future Enhancements

1. **Graph Neural Networks:** Leverage communication network structure
2. **Hierarchical Classification:** Multi-level prediction (HQ → Regiment → Battalion)
3. **Online Learning:** Incremental updates with new observations
4. **Ensemble Methods:** Combine multiple classification models
5. **Feature Importance:** SHAP values for interpretability
6. **Active Learning:** Identify uncertain predictions for labeling
