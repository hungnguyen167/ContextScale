# Training Script Usage Guide

## Improved train.py Script

The `train.py` script has been significantly improved to match the notebook's data loading and processing patterns. Here's how to use it:

### Features

1. **Exact Notebook Replication**: Follows the same data preprocessing pipeline as the notebook
2. **Intelligent Caching**: Checks for preprocessed data to avoid reprocessing 
3. **Complete Configuration**: Includes all ensemble parameters from the notebook
4. **Robust Error Handling**: Graceful shutdown and comprehensive error reporting
5. **Signal Handling**: Properly handles Ctrl+C and termination signals
6. **Memory Management**: Includes memory cleanup and CUDA cache clearing

### Basic Usage

```bash
# Simple training with defaults (5 models, 5 epochs, xlm-roberta-large)
python train.py

# Custom configuration
python train.py --num_models 3 --epochs 3 --lr 1e-5 --model_name xlm-roberta-base

# Large ensemble training
python train.py --num_models 10 --epochs 10 --batch_size 32 --beta 1.5
```

### Command Line Arguments

```
--num_models      Number of ensemble models (default: 5)
--num_splits      Number of data splits (default: 5) 
--epochs          Training epochs per model (default: 5)
--lr              Learning rate (default: 2e-5)
--beta            Beta for exponential position score (default: 1.0)
--model_name      Transformer model (default: xlm-roberta-large)
--save_dir        Output directory (default: results/models/ensemble_splits)
--lora            Use LoRA (default: True)
--batch_size      Batch size (default: 16)
--max_length      Max sequence length (default: 512)
```

### Running Under tmux/screen

For persistent training sessions that survive laptop closure:

#### Option 1: tmux (Recommended)
```bash
# Start tmux session
tmux new-session -d -s training

# Attach to session
tmux attach-session -t training

# Inside tmux, run training
python train.py --num_models 5 --epochs 5

# Detach (Ctrl+B then D) - training continues
# Later, reattach with:
tmux attach-session -t training
```

#### Option 2: screen
```bash
# Start screen session
screen -S training

# Run training
python train.py --num_models 5 --epochs 5

# Detach (Ctrl+A then D) - training continues  
# Reattach with:
screen -r training
```

### Data Loading Pattern

The script automatically:

1. **Checks for preprocessed data** in `data/temps/`
2. **If found**: Loads cached manifesto.csv and manifesto_regrouped.csv
3. **If not found**: 
   - Loads raw data from `data/r_outputs/pulled_manifestoes.csv`
   - Applies sentiment_code and topic_code functions
   - Groups texts using group_texts function
   - Saves preprocessed data for future use

### Output Structure

Training creates:
```
results/models/ensemble_splits/
├── ensemble_config.pkl           # Configuration used
├── model_ensemble_splits_0.safetensors
├── model_ensemble_splits_1.safetensors
├── model_ensemble_splits_2.safetensors
├── model_ensemble_splits_3.safetensors
└── model_ensemble_splits_4.safetensors

data/temps/
├── topic_labels                  # Pickled topic label mapping
├── sentiment_labels              # Pickled sentiment label mapping
├── manifesto.csv                 # Cached raw processed data
└── manifesto_regrouped.csv       # Cached regrouped data
```

### Expected Runtime

- **5 models, 5 epochs each**: ~2-4 hours on GPU
- **Memory usage**: ~8-12GB GPU RAM for xlm-roberta-large
- **Disk space**: ~2-3GB per model (with LoRA: ~100MB per model)

### Monitoring Progress

The script provides comprehensive logging:
- Dataset loading and preprocessing status
- Train/eval/test split summaries
- Per-epoch training progress
- Model saving confirmations
- Final completion summary

### Error Recovery

If training stops:
1. Check the last saved model number in the save directory
2. Models are saved after each epoch, so minimal progress is lost
3. You can modify the script to resume from a specific model if needed

This improved script makes ensemble training much more robust and suitable for long-running remote sessions!