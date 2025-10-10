#!/usr/bin/env python3
"""
Standalone training script for deep ensemble training with different data splits.
Designed to run under tmux/screen for persistent training sessions.

This script follows the exact data loading and processing pattern from the notebook:
- Loads raw manifesto data from data/r_outputs/pulled_manifestoes.csv
- Applies the same preprocessing, grouping, and encoding steps
- Creates proper datasets with stratified splits
- Trains ensembles using different data splits for better diversity

Usage:
    # Inside tmux session
    python train_improved.py --num_models 5 --epochs 5 --model_name xlm-roberta-large
    
    # With custom settings
    python train_improved.py --num_models 3 --epochs 3 --lr 1e-5 --beta 0.8 --batch_size 32
"""

import os
import sys
import json
import argparse
import signal
import time
import traceback
from pathlib import Path

# Add utils to path
sys.path.append('utils/')

import torch
import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
import pickle

# Import our custom modules
from utils.uncertainty import create_multiple_data_splits, train_deep_ensemble_with_different_splits
from utils.models import ContextScalePrediction
from utils.functions import sentiment_code, topic_code, group_texts, tokenize_function


def setup_environment():
    """Setup CUDA and random seeds for reproducibility."""
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pseudo-randomness for reproducibility
    seed_val = 1234
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    return device, seed_val


def load_and_prepare_data():
    """Load and prepare manifesto data following the notebook pattern."""
    print("Loading manifesto data...")
    
    # Check if preprocessed data exists
    manifesto_temp_path = 'data/temps/manifesto.csv'
    manifesto_regrouped_temp_path = 'data/temps/manifesto_regrouped.csv'
    
    if os.path.exists(manifesto_temp_path) and os.path.exists(manifesto_regrouped_temp_path):
        print("Found preprocessed data, loading...")
        manifesto = pd.read_csv(manifesto_temp_path, encoding='utf-8', dtype={2:'str',18: 'str'})
        manifesto_regrouped = pd.read_csv(manifesto_regrouped_temp_path, encoding='utf-8')
    else:
        print("Preprocessing raw data...")
        # Load raw data
        manifesto = pd.read_csv(os.path.join("data", "r_outputs","pulled_manifestoes.csv"), 
                               encoding="utf-8", 
                               dtype={2: 'str', 18:'str'})
        
        # Filter and prepare
        manifesto = manifesto[(manifesto.cmp_code.notna()) & 
                             ~(manifesto.cmp_code.isin(['H']))].reset_index(drop=True)
        
        manifesto['sentiment'] = manifesto['cmp_code'].apply(sentiment_code)
        manifesto['topic'] = manifesto['cmp_code'].apply(topic_code)
        manifesto['election'] = manifesto['date'].astype(str).str[:4]
        
        # Group texts
        results = group_texts(manifesto, 
                             ['countryname','election','party','cmp_code'], 'text', 
                             max_group_factor=5)
        
        manifesto_regrouped = pd.DataFrame(results)
        manifesto_regrouped = manifesto_regrouped.explode('text').reset_index(drop=True)
        
        # Process columns
        df_cols = manifesto_regrouped['labels'].str.split(';', expand=True)
        manifesto_regrouped = pd.concat([manifesto_regrouped, df_cols], axis=1)
        manifesto_regrouped.columns = ['text', 'idx', 'country','election', 'party', 'cmp_code']
        
        manifesto_regrouped['sentiment'] = manifesto_regrouped['cmp_code'].apply(sentiment_code)
        manifesto_regrouped['topic'] = manifesto_regrouped['cmp_code'].apply(topic_code)
        manifesto_regrouped = manifesto_regrouped.drop_duplicates().reset_index(drop=True)
        
        # Save preprocessed data
        os.makedirs('data/temps', exist_ok=True)
        manifesto_regrouped.to_csv(manifesto_regrouped_temp_path, encoding='utf-8', index=False)
        manifesto.to_csv(manifesto_temp_path, encoding='utf-8', index=False)
    
    # Prepare final dataset
    manifesto_reduced = manifesto_regrouped[['topic','sentiment','text']].reset_index(drop=True)
    manifesto_reduced['topic_sentiment'] = manifesto_reduced['topic'] + '_' + manifesto_reduced['sentiment']
    
    print(f"Dataset prepared: {len(manifesto_reduced)} samples")
    print("Topic distribution:")
    print(manifesto_reduced['topic'].value_counts())
    print("\nSentiment distribution:")
    print(manifesto_reduced['sentiment'].value_counts())
    
    return manifesto_reduced


def create_dataset(manifesto_reduced, seed_val):
    """Create and encode the dataset following notebook pattern."""
    print("Creating dataset...")
    
    manifesto_dataset = Dataset.from_pandas(manifesto_reduced)
    manifesto_dataset = manifesto_dataset.class_encode_column('topic')
    manifesto_dataset = manifesto_dataset.class_encode_column('sentiment')
    manifesto_dataset = manifesto_dataset.class_encode_column('topic_sentiment')
    
    # Save class labels
    os.makedirs('data/temps', exist_ok=True)
    
    topic_labels = manifesto_dataset.features['topic'].names
    with open('data/temps/topic_labels', 'wb') as fp:
        pickle.dump(topic_labels, fp)
    print(f"Topic labels: {topic_labels}")
    
    sentiment_labels = manifesto_dataset.features['sentiment'].names
    with open('data/temps/sentiment_labels', 'wb') as fp:
        pickle.dump(sentiment_labels, fp)
    print(f"Sentiment labels: {sentiment_labels}")
    
    return manifesto_dataset


def create_model_factory(model_name, num_topics, num_sentiments, lora=True):
    """Create model factory function following notebook pattern."""
    def create_model():
        return ContextScalePrediction(
            roberta_model=model_name, 
            num_topics=num_topics, 
            num_sentiments=num_sentiments,
            lora=lora,
            use_shared_attention=True  # Using shared attention architecture
        )
    return create_model


def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM gracefully."""
    print(f"\nReceived signal {signum}. Saving current progress and exiting gracefully...")
    # Add any cleanup code here
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Deep Ensemble Training with Different Data Splits')
    parser.add_argument('--num_models', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--num_splits', type=int, default=5, help='Number of data splits')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs per model')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for exponential position score')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', help='Model name')
    parser.add_argument('--save_dir', type=str, default='results/models/ensemble_splits', help='Save directory')
    parser.add_argument('--lora', action='store_true', default=True, help='Use LoRA')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("=" * 80)
        print("Starting Deep Ensemble Training Script")
        print("=" * 80)
        print(f"Configuration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print("=" * 80)
        
        # Setup environment
        device, seed_val = setup_environment()
        
        # Load and prepare data
        manifesto_reduced = load_and_prepare_data()
        
        # Create dataset
        manifesto_dataset = create_dataset(manifesto_reduced, seed_val)
        
        # Setup tokenizer and data collator
        print(f"Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        data_collator = DataCollatorWithPadding(tokenizer)
        
        # Calculate dataset dimensions
        num_topics = len(set(manifesto_reduced['topic']))
        num_sentiments = len(set(manifesto_reduced['sentiment']))
        print(f"Dataset dimensions: {num_topics} topics, {num_sentiments} sentiments")
        
        # Create ensemble configuration (following notebook pattern)
        ENSEMBLE_CONFIG = {
            'num_models': args.num_models,
            'beta': args.beta,
            'n_epochs': args.epochs,
            'lr': args.lr,
            'save_dir': args.save_dir,
            'model_prefix': 'model_ensemble_splits'
        }
        
        print("\nEnsemble Configuration:")
        for key, value in ENSEMBLE_CONFIG.items():
            print(f"  {key}: {value}")
        
        # Create model factory
        create_model = create_model_factory(args.model_name, num_topics, num_sentiments, args.lora)
        
        # Test model creation
        print("\nTesting model creation...")
        test_model = create_model()
        print(f"Model created successfully: {test_model.__class__.__name__}")
        del test_model  # Free memory
        
        # Create multiple data splits
        print(f"\nCreating {args.num_splits} different data splits from the original dataset...")
        print("Each split will have different train/eval/test samples for better ensemble diversity")
        
        data_splits = create_multiple_data_splits(
            original_dataset=manifesto_dataset,
            num_splits=args.num_splits,
            test_size=0.1,  # 10% for test
            eval_size=0.3,  # 30% of remaining data for eval
            stratify_column='topic_sentiment',
            base_seed=seed_val
        )
        
        print(f"\nSuccessfully created {len(data_splits)} different data splits!")
        print("Summary of splits:")
        for i, split in enumerate(data_splits):
            print(f"Split {i+1}: Train={len(split['train'])}, Eval={len(split['eval'])}, Test={len(split['test'])}, Seed={split['seed']}")
        
        # Save ensemble configuration
        os.makedirs(args.save_dir, exist_ok=True)
        config_path = os.path.join(args.save_dir, 'ensemble_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(ENSEMBLE_CONFIG, f)
        print(f"\nEnsemble configuration saved to: {config_path}")
        
        # Train ensemble with different splits
        print("\n" + "=" * 80)
        print("Starting Deep Ensemble Training with Different Data Splits")
        print(f"This will train {len(data_splits)} models using completely different train/eval/test splits")
        print(f"Each model will be trained for {args.epochs} epochs")
        print("=" * 80)
        
        train_deep_ensemble_with_different_splits(
            data_splits=data_splits,
            create_model_fn=create_model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            device=device,
            n_epochs=args.epochs,
            lr=args.lr,
            save_dir=args.save_dir,
            model_prefix='model_ensemble_splits',
            max_length=args.max_length,
            batch_size=args.batch_size
        )
        
        print("\n" + "=" * 80)
        print("🎉 Training completed successfully!")
        print(f"Models saved in: {args.save_dir}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()