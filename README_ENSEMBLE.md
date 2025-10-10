# ContextScale Enhanced: Deep Ensemble with Uncertainty Estimation

This enhanced version of ContextScale includes redesigned position score computation and deep ensemble training with comprehensive uncertainty estimation.

## Key Features

### 1. Exponential Position Score Formula
- **New Formula**: `position_score = exp(-β * neutral_prob) * (right_prob - left_prob)`
- **Benefits**: 
  - Reduces position scores when neutral confidence is high
  - More nuanced handling of political ambiguity
  - Configurable β parameter (default=1.0)

### 2. Deep Ensemble Training
- Trains multiple models with different random seeds
- Captures epistemic uncertainty (model disagreement)
- Improves robustness and generalization

### 3. Uncertainty Quantification
- **Epistemic Uncertainty**: Variance across ensemble predictions
- **Aleatoric Uncertainty**: Inherent data uncertainty  
- **Total Uncertainty**: Combined uncertainty estimate
- **Confidence Intervals**: 95% CI for position scores

## New Files

### `utils/uncertainty.py`
Core module containing:
- `compute_position_score_exponential()`: New position score formula
- `train_deep_ensemble()`: Deep ensemble training
- `ensemble_inference()`: Inference with uncertainty estimation
- `compute_epistemic_variance()`: Epistemic uncertainty calculation
- `compute_aleatoric_variance()`: Aleatoric uncertainty calculation

### Enhanced `utils/functions.py`
- Updated `scale_func()` with exponential option
- Backward compatibility with original approach

## Usage

### Basic Position Score Computation
```python
from utils.uncertainty import compute_position_score_exponential

# For 3-class sentiment: [left, neutral, right] probabilities
probs = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
scores = compute_position_score_exponential(probs, beta=1.0)
```

### Deep Ensemble Training
```python
from utils.uncertainty import train_deep_ensemble

# Define model factory
def create_model():
    return ContextScalePrediction(...)

# Train ensemble
ensemble_info = train_deep_ensemble(
    model_factory=create_model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    device=device,
    num_models=5,
    n_epochs=5
)
```

### Ensemble Inference
```python
from utils.uncertainty import ensemble_inference

results = ensemble_inference(
    models=ensemble_models,
    dataloader=test_dataloader,
    device=device,
    beta=1.0
)

# Results contain:
# - mean_position_scores: Average across ensemble
# - epistemic_variance: Model uncertainty
# - aleatoric_variance: Data uncertainty
# - total_variance: Combined uncertainty
```

## Configuration Parameters

Key parameters you can adjust:

- **beta** (float, default=1.0): Controls exponential gating strength
- **num_models** (int, default=5): Number of ensemble members
- **n_epochs** (int, default=5): Training epochs per model
- **lr** (float, default=2e-5): Learning rate

## Environment Variables

Set these before running to override defaults:
```bash
export ENSEMBLE_BETA=1.5
export ENSEMBLE_NUM_MODELS=7
export ENSEMBLE_EPOCHS=5
```

## Notebook Integration

The enhanced functionality is integrated into `03_main_analyses.ipynb`:

1. **Position Score Comparison**: Original vs exponential approaches
2. **Ensemble Training**: Automated training of multiple models
3. **Uncertainty Analysis**: Comprehensive uncertainty visualization
4. **Results Export**: Save results and uncertainty estimates

## Scientific Benefits

1. **Uncertainty-Aware Predictions**: Know when the model is uncertain
2. **Robust Position Scores**: Better handling of political ambiguity
3. **Model Reliability**: Ensemble reduces overfitting
4. **Reproducible Research**: Quantified uncertainty for scientific rigor

## File Structure

```
utils/
├── uncertainty.py       # New uncertainty estimation module
├── functions.py         # Enhanced with exponential support
└── models.py           # Unchanged

results/
├── models/ensemble/    # Ensemble model checkpoints
└── ensemble_uncertainty/ # Uncertainty analysis results
```

## Dependencies

All dependencies are the same as the original ContextScale project:
- torch
- transformers
- safetensors
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Citation

If you use this enhanced version, please cite both the original ContextScale work and mention the ensemble/uncertainty enhancements.