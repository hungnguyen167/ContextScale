"""
Uncertainty estimation and improved position score computation for ContextScale.

This module provides:
1. Exponential position score computation with beta parameter
2. Deep ensemble training and inference
3. Epistemic and aleatoric uncertainty estimation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import time
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from transformers import get_linear_schedule_with_warmup
from safetensors.torch import save_file, load_file
import os
import copy

try:
    from .functions import train_loop, eval_loop
except Exception:
    from utils.functions import train_loop, eval_loop

def create_multiple_data_splits(
    original_dataset,
    num_splits: int = 5,
    test_size: float = 0.1,
    eval_size: float = 0.3,
    stratify_column: str = 'topic_sentiment',
    base_seed: int = 42
) -> List[Dict]:
    """
    Create multiple different train/eval/test splits from the original dataset.
    
    Args:
        original_dataset: The original dataset to split
        num_splits: Number of different splits to create
        test_size: Proportion of data for test set
        eval_size: Proportion of remaining data for eval set (after test split)
        stratify_column: Column to use for stratified splitting
        base_seed: Base random seed (each split gets base_seed + split_id)
    
    Returns:
        splits: List of dictionaries containing {'train': Dataset, 'eval': Dataset, 'test': Dataset}
    """
    splits = []
    
    for split_id in range(num_splits):
        seed = base_seed + split_id * 1000  # Use different seeds for each split
        print(f"Creating data split {split_id + 1}/{num_splits} with seed {seed}")
        
        # First split: train+eval vs test
        train_eval_test = original_dataset.train_test_split(
            test_size=test_size, 
            stratify_by_column=stratify_column, 
            seed=seed
        )
        
        # Second split: train vs eval
        train_eval_split = train_eval_test['train'].train_test_split(
            test_size=eval_size, 
            stratify_by_column=stratify_column, 
            seed=seed
        )
        
        # Create the split dictionary
        split_dict = {
            'train': train_eval_split['train'],
            'eval': train_eval_split['test'],  # Note: 'test' from second split becomes 'eval'
            'test': train_eval_test['test'],
            'seed': seed,
            'split_id': split_id
        }
        
        splits.append(split_dict)
        
        print(f"Split {split_id + 1}: Train={len(split_dict['train'])}, "
              f"Eval={len(split_dict['eval'])}, Test={len(split_dict['test'])}")
    
    return splits


def create_shuffled_dataloader(original_dataloader: DataLoader, seed: int) -> DataLoader:
    """
    Create a new DataLoader with different shuffling for ensemble training.
    
    Args:
        original_dataloader: Original DataLoader to copy
        seed: Random seed for shuffling
    
    Returns:
        new_dataloader: DataLoader with different shuffle order
    """
    # Set seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Create new DataLoader with same parameters but different shuffle
    new_dataloader = DataLoader(
        dataset=original_dataloader.dataset,
        batch_size=original_dataloader.batch_size,
        shuffle=True,  # Ensure shuffle is enabled
        collate_fn=original_dataloader.collate_fn,
        num_workers=getattr(original_dataloader, 'num_workers', 0),
        pin_memory=getattr(original_dataloader, 'pin_memory', False),
        generator=generator
    )
    
    return new_dataloader


def compute_position_score_exponential(probs: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Compute position scores using exponential gating approach.
    
    Args:
        probs: Array of shape (N, 3) with [left, neutral, right] probabilities
        beta: Exponential decay parameter for neutral gating (default=1.0)
    
    Returns:
        position_scores: Array of shape (N,) with scores in range [-1, 1]
        
    Formula:
        position_score = exp(-beta * neutral_prob) * (right_prob - left_prob)
    """
    if probs.shape[1] == 2:
        # Binary case: only left and right probabilities
        left_probs = probs[:, 0]
        right_probs = probs[:, 1]
        return right_probs - left_probs
    
    elif probs.shape[1] == 3:
        # Ternary case: left, neutral, right probabilities
        left_probs = probs[:, 0]
        neutral_probs = probs[:, 1]
        right_probs = probs[:, 2]
        
        # Exponential gating based on neutral confidence
        gating_factor = np.exp(-beta * neutral_probs)
        raw_score = right_probs - left_probs
        
        # Apply gating to reduce extreme scores when neutral is highly confident
        position_scores = gating_factor * raw_score
        
        return position_scores
    
    else:
        raise ValueError(f"Expected 2 or 3 sentiment classes, got {probs.shape[1]}")


def compute_epistemic_variance(ensemble_scores: List[np.ndarray]) -> np.ndarray:
    """
    Compute epistemic uncertainty as variance across ensemble predictions.
    
    Args:
        ensemble_scores: List of position score arrays from different models
    
    Returns:
        epistemic_var: Variance across ensemble members for each sample
    """
    # Stack scores from all ensemble members
    stacked_scores = np.stack(ensemble_scores, axis=0)  # Shape: (num_models, num_samples)
    
    # Compute variance across models (axis=0)
    epistemic_var = np.var(stacked_scores, axis=0)
    
    return epistemic_var


def compute_aleatoric_variance(probs_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute aleatoric uncertainty as average predictive variance.
    
    Args:
        probs_list: List of probability arrays from different models
    
    Returns:
        aleatoric_var: Average aleatoric variance for each sample
    """
    aleatoric_vars = []
    
    for probs in probs_list:
        # Compute predictive variance for each model
        # Variance of multinomial: p(1-p) for each class, then sum
        if probs.shape[1] == 2:
            # Binary case
            var_left = probs[:, 0] * (1 - probs[:, 0])
            var_right = probs[:, 1] * (1 - probs[:, 1])
            total_var = var_left + var_right
        elif probs.shape[1] == 3:
            # Ternary case
            var_left = probs[:, 0] * (1 - probs[:, 0])
            var_neutral = probs[:, 1] * (1 - probs[:, 1])
            var_right = probs[:, 2] * (1 - probs[:, 2])
            total_var = var_left + var_neutral + var_right
        else:
            raise ValueError(f"Expected 2 or 3 sentiment classes, got {probs.shape[1]}")
        
        aleatoric_vars.append(total_var)
    
    # Average across ensemble members
    mean_aleatoric_var = np.mean(aleatoric_vars, axis=0)
    
    return mean_aleatoric_var


def train_ensemble_member(
    model,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    device: torch.device,
    n_epochs: int = 5,
    lr: float = 2e-5,
    sentiment_var: str = 'sentiment',
    topic_var: str = 'topic',
    model_id: int = 0,
    save_path: Optional[str] = None,
    reshuffle_dataloader: bool = True,
    org_seed: int = 42
) -> Dict:
    """
    Train a single ensemble member.
    
    Args:
        model: The model to train
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader (optional, can be None)
        device: Device to train on
        n_epochs: Number of training epochs
        lr: Learning rate
        sentiment_var: Name of sentiment variable in batch
        topic_var: Name of topic variable in batch
        model_id: ID of this ensemble member
        save_path: Path to save model checkpoint
    
    Returns:
        training_info: Dictionary with training metrics
    """
    
    
    # Setup optimizer and scheduler
    total_steps = len(train_dataloader) * n_epochs
    warmup = total_steps * 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_training_steps=total_steps, 
        num_warmup_steps=warmup
    )
    
    # Loss functions
    criterion_sent = nn.CrossEntropyLoss()
    criterion_topic = nn.CrossEntropyLoss()
    
    training_info = {
        'model_id': model_id,
        'epochs': [],
        'train_times': []
    }
    
    print(f"Training ensemble member {model_id+1}")
    
    # Create shuffled dataloader for this ensemble member if requested
    if reshuffle_dataloader:
        shuffled_train_dataloader = create_shuffled_dataloader(train_dataloader, seed=org_seed + model_id)
        actual_train_dataloader = shuffled_train_dataloader
        print(f"Using shuffled data with seed {org_seed + model_id}")
    else:
        actual_train_dataloader = train_dataloader
    
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch+1}/{n_epochs}")
        
        # Train for one epoch
        timing_log = train_loop(
            actual_train_dataloader, 
            model, 
            optimizer, 
            scheduler, 
            device, 
            criterion_sent, 
            criterion_topic, 
            sentiment_var=sentiment_var,
            topic_var=topic_var, 
            timing_log=True
        )
        
        # Evaluate (only if eval_dataloader is provided)
        if eval_dataloader is not None:
            eval_loop(
                eval_dataloader, 
                model, 
                device, 
                criterion_sent, 
                criterion_topic, 
                sentiment_var=sentiment_var, 
                topic_var=topic_var
            )
        else:
            print("  Skipping evaluation (no eval_dataloader provided)")
        
        training_info['epochs'].append(epoch)
        training_info['train_times'].append(timing_log.get('epoch_time', 0))
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        state_dict = model.state_dict()
        save_file(state_dict, save_path)
        training_info['checkpoint_path'] = save_path
        print(f"Saved model checkpoint to {save_path}")
    
    return training_info


def train_deep_ensemble(
    model_base: Callable[[], torch.nn.Module],
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    device: torch.device,
    num_models: int = 5,
    n_epochs: int = 5,
    lr: float = 2e-5,
    sentiment_var: str = 'sentiment',
    topic_var: str = 'topic',
    save_dir: str = 'results/models/ensemble',
    model_prefix: str = 'model_ensemble',
    org_seed: int = 42
) -> List[Dict]:
    """
    Train a deep ensemble of models using the same data split but different shuffles.
    
    Args:
        model_factory: Function that returns a new model instance
        train_dataloader: Training data loader (will be reshuffled for each model)
        eval_dataloader: Evaluation data loader (optional, can be None)
        device: Device to train on
        num_models: Number of ensemble models to train
        n_epochs: Number of training epochs per model
        lr: Learning rate
        sentiment_var: Name of sentiment variable in batch
        topic_var: Name of topic variable in batch
        save_dir: Directory to save model checkpoints
        model_prefix: Prefix for model checkpoint filenames
        org_seed: Base seed for model initialization and data shuffling
    
    Returns:
        ensemble_info: List of training info for each model
    """
    
    ensemble_info = []
    
    for i in range(num_models):
        print(f"\n{'='*50}")
        print(f"Training ensemble member {i+1}/{num_models}")
        print(f"Using shuffled training data with seed {org_seed + i}")
        print(f"{'='*50}")
        
        # Set different random seed for each model (but consistent across runs)
        torch.manual_seed(org_seed + i)
        np.random.seed(org_seed + i)
        
        # Create model instance
        model = model_base.to(device)
        
        # Define save path
        save_path = os.path.join(save_dir, f"{model_prefix}_{i}.safetensors")
        
        # Train this ensemble member with shuffled data
        info = train_ensemble_member(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            device=device,
            n_epochs=n_epochs,
            lr=lr,
            sentiment_var=sentiment_var,
            topic_var=topic_var,
            model_id=i,
            save_path=save_path,
            reshuffle_dataloader=True,  # Shuffle training data for each model
            org_seed=org_seed
        )
        
        ensemble_info.append(info)
        
        # Clean up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*50}")
    print(f"Ensemble training completed!")
    print(f"Trained {num_models} models with different data shuffles")
    print(f"{'='*50}")
    
    return ensemble_info



def load_ensemble_models(
    model_base: Callable[[], torch.nn.Module],
    checkpoint_paths: List[str],
    device: torch.device
) -> List[torch.nn.Module]:
    """
    Load ensemble models from checkpoints.
    
    Args:
        model_factory: Function that returns a new model instance
        checkpoint_paths: List of paths to model checkpoints
        device: Device to load models on
    
    Returns:
        models: List of loaded models
    """
    models = []
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Loading ensemble member {i} from {checkpoint_path}")
        
        # Create new model instance
        model = model_base.to(device)
        
        # Load checkpoint
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
        model.eval()
        
        models.append(model)
    
    return models


def ensemble_inference(
    models: List[torch.nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    beta: float = 1.0,
    topic_label: Optional[str] = None,
    sentiment_label: Optional[str] = None,
    use_ground_truth_topic: bool = False,
    timing_log: bool = True
) -> Dict:
    """
    Run ensemble inference on a given dataloader.

    Args:
        models: List of ensemble models.
        dataloader: DataLoader to run inference on.
        device: Device to run inference on.
        beta: Beta parameter for exponential position score computation.
        topic_label: Name of topic label in batch (if available).
        sentiment_label: Name of sentiment label in batch (if available).
        use_ground_truth_topic: If True and topic_label is available, uses ground truth topics for scaling.
        timing_log: Whether to log timing information.

    Returns:
        results: Dictionary with ensemble predictions and uncertainty estimates.
    """
    # Collect predictions from all models
    all_pred_topics = []
    all_pred_sentiments = []
    all_sentiment_probs = []
    all_position_scores = []
    all_ground_truth_topics = []
    all_ground_truth_sentiments = []
    t0_local = time.time() if timing_log else None

    print(f"Running ensemble inference with {len(models)} models...")

    for i, model in enumerate(models):
        print(f"\nRunning inference with model {i+1}/{len(models)}")
        model.eval()
        
        # Storage for this model's predictions
        pred_topics = []
        pred_sentiments = []
        sentiment_probs = []
        true_topics_local = []
        true_sentiments_local = []
        t1_local = time.time() if timing_log else None
        with torch.no_grad():
            for batch_num, batch in enumerate(dataloader):
                if timing_log and t1_local is not None and (batch_num + 1) % 100 == 0:
                    elapsed = time.time() - t1_local
                    avg_batch_time = elapsed / (batch_num + 1)
                    total_batches = len(dataloader)
                    estimated_total_time = avg_batch_time * total_batches
                    estimated_remaining_time = estimated_total_time - elapsed
                    print(f"  Batch {batch_num+1}/{total_batches} | Elapsed: {elapsed:.2f}s, Remaining: {estimated_remaining_time:.2f}s")

                batch = {k: v.to(device) for k, v in batch.items()}
                topic_labels = batch.get(topic_label, None)
                sent_labels = batch.get(sentiment_label, None)

                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                # Store predictions
                pred_topic = outputs['logits_topic'].argmax(1)
                pred_sentiment = outputs['logits_sentiment'].argmax(1)
                sentiment_sigmoid = torch.sigmoid(outputs['logits_sentiment'])

                pred_topics.append(pred_topic)
                pred_sentiments.append(pred_sentiment)
                sentiment_probs.append(sentiment_sigmoid)

                # Store ground truth (only for first model to avoid duplicates)
                if i == 0:
                    if topic_labels is not None:
                        true_topics_local.append(topic_labels)
                    if sent_labels is not None:
                        true_sentiments_local.append(sent_labels)

        # Concatenate predictions for this model
        model_pred_topics = torch.cat(pred_topics, dim=0).cpu().detach().numpy()
        model_pred_sentiments = torch.cat(pred_sentiments, dim=0).cpu().detach().numpy()
        model_sentiment_probs = torch.cat(sentiment_probs, dim=0).cpu().detach().numpy()
        # Process ground truth topics (only for first model)
        ground_truth_topics_local = None
        if i == 0 and len(true_topics_local) > 0:
            ground_truth_topics_local = torch.cat(true_topics_local, dim=0).cpu().detach().numpy()
            


        # Determine which topic labels to use for position score computation
        if ground_truth_topics_local is not None and use_ground_truth_topic:
            topic_labels_for_scaling = ground_truth_topics_local
            print("  Using ground truth topic labels for position score computation")
        else:
            topic_labels_for_scaling = model_pred_topics
            print("  Using predicted topic labels for position score computation")

        ground_truth_sentiments_local = None
        
        if i == 0 and len(true_sentiments_local) > 0:
            ground_truth_sentiments_local = torch.cat(true_sentiments_local, dim=0).cpu().detach().numpy()

        # Determine which topic labels to use for position score computation
        if use_ground_truth_topic and ground_truth_topics_local is not None:
            topic_labels_for_scaling = ground_truth_topics_local
        else:
            topic_labels_for_scaling = model_pred_topics
        # Compute position scores for this model
        model_position_scores = compute_position_scores_by_topic(
            model_sentiment_probs,
            topic_labels_for_scaling,
            beta=beta
        )

        all_pred_topics.append(model_pred_topics)
        all_pred_sentiments.append(model_pred_sentiments)
        all_sentiment_probs.append(model_sentiment_probs)
        all_position_scores.append(model_position_scores)
        if i == 0:  # Only store ground truth once
            all_ground_truth_topics = ground_truth_topics_local if ground_truth_topics_local is not None else None
            all_ground_truth_sentiments = ground_truth_sentiments_local if ground_truth_sentiments_local is not None else None
    # Aggregate ensemble results
    print("\nComputing ensemble statistics...")

    if len(models) == 1:
        # Single model case - no aggregation across models needed
        mean_sentiment_probs = all_sentiment_probs[0]
        mean_position_scores = all_position_scores[0]
        ensemble_pred_topics = all_pred_topics[0]
        ensemble_pred_sentiments = all_pred_sentiments[0]
        
        # For single model, epistemic uncertainty is zero
        epistemic_var = np.zeros_like(mean_position_scores)
        aleatoric_var = compute_aleatoric_variance(all_sentiment_probs)
        total_var = aleatoric_var
    else:
        # Multiple models case - do ensemble aggregation
        mean_sentiment_probs = np.mean(all_sentiment_probs, axis=0)
        mean_position_scores = np.mean(all_position_scores, axis=0)
        epistemic_var = compute_epistemic_variance(all_position_scores)
        
        # Use majority vote for discrete predictions
        ensemble_pred_topics = np.array([
            np.bincount([all_pred_topics[i][j] for i in range(len(models))]).argmax()
            for j in range(len(all_pred_topics[0]))
        ])

        ensemble_pred_sentiments = np.array([
            np.bincount([all_pred_sentiments[i][j] for i in range(len(models))]).argmax()
            for j in range(len(all_pred_sentiments[0]))
        ])

        # Compute uncertainties
        aleatoric_var = compute_aleatoric_variance(all_sentiment_probs)
        total_var = epistemic_var + aleatoric_var

    results = {
        'mean_position_scores': mean_position_scores,
        'epistemic_variance': epistemic_var,
        'aleatoric_variance': aleatoric_var,
        'total_variance': total_var,
        'ensemble_pred_topics': ensemble_pred_topics,
        'ensemble_pred_sentiments': ensemble_pred_sentiments,
        'mean_sentiment_probs': mean_sentiment_probs,
        'individual_position_scores': all_position_scores,
        'individual_sentiment_probs': all_sentiment_probs,
        'individual_pred_topics': all_pred_topics,
        'individual_pred_sentiments': all_pred_sentiments,
        'used_ground_truth_topics': use_ground_truth_topic,
        'ground_truth_topics': all_ground_truth_topics,
        'ground_truth_sentiments': all_ground_truth_sentiments,
        'beta': beta,
        'num_models': len(models)
    }

    # Timing and summary
    total_time_local = time.time() - t0_local if timing_log and t0_local is not None else None
    if timing_log and total_time_local is not None:
        print(f"Total inference time: {total_time_local:.2f}s")
    
    if len(models) == 1:
        print("Single model inference - no epistemic uncertainty")
    else:
        used_gt = use_ground_truth_topic and len(all_ground_truth_topics) > 0 and all_ground_truth_topics[0] is not None
        print(f"Used {'ground truth' if used_gt else 'predicted'} topic labels for position scaling")
        print(f"Mean position score range: [{mean_position_scores.min():.3f}, {mean_position_scores.max():.3f}]")
        print(f"Mean epistemic variance: {epistemic_var.mean():.6f}")
        print(f"Mean aleatoric variance: {aleatoric_var.mean():.6f}")

    print("Ensemble inference completed!")
    
    # Clean up
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


def compute_position_scores_by_topic(
    sentiment_probs: np.ndarray,
    topic_predictions: np.ndarray,
    beta: float = 1.0
) -> np.ndarray:
    """
    Compute position scores grouped by topic using exponential approach.
    
    Args:
        sentiment_probs: Array of sentiment probabilities (N, num_sentiment_classes)
        topic_predictions: Array of topic predictions (N,)
        beta: Beta parameter for exponential computation
    
    Returns:
        position_scores: Array of position scores (N,)
    """
    position_scores = np.zeros(len(topic_predictions))
    original_indices = np.arange(len(topic_predictions))
    
    # Compute scores for each topic separately
    for topic_id in np.unique(topic_predictions):
        topic_mask = topic_predictions == topic_id
        topic_indices = original_indices[topic_mask]
        topic_probs = sentiment_probs[topic_mask]
        
        # Use the exponential approach for position scores
        topic_scores = compute_position_score_exponential(topic_probs, beta=beta)
        position_scores[topic_indices] = topic_scores
    
    return position_scores


def save_ensemble_results(
    results: Dict,
    save_path: str,
    include_individual_predictions: bool = False
) -> None:
    """
    Save ensemble results to file.
    
    Args:
        results: Results dictionary from ensemble_inference
        save_path: Path to save results
        include_individual_predictions: Whether to include individual model predictions
    """
    # Create a copy for saving
    save_results = results.copy()
    
    # Optionally exclude large individual prediction arrays
    if not include_individual_predictions:
        keys_to_remove = [
            'individual_position_scores',
            'individual_sentiment_probs',
            'individual_pred_topics',
            'individual_pred_sentiments'
        ]
        for key in keys_to_remove:
            save_results.pop(key, None)
    
    # Save to pickle
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(save_results, f)
    
    print(f"Ensemble results saved to {save_path}")


def create_ensemble_summary_dataframe(results: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame from ensemble results.
    
    Args:
        results: Results dictionary from ensemble_inference
    
    Returns:
        df: DataFrame with per-sample results
    """
    df = pd.DataFrame({
        'mean_position_score': results['mean_position_scores'],
        'epistemic_variance': results['epistemic_variance'],
        'aleatoric_variance': results['aleatoric_variance'],
        'total_variance': results['total_variance'],
        'ensemble_pred_topic': results['ensemble_pred_topics'],
        'ensemble_pred_sentiment': results['ensemble_pred_sentiments'],
        'ground_truth_topic': results['ground_truth_topics'] if results['ground_truth_topics'] is not None else [None]*len(results['mean_position_scores']),
        'ground_truth_sentiment': results['ground_truth_sentiments'] if results['ground_truth_sentiments'] is not None else [None]*len(results['mean_position_scores']),
    })
    

    # Add uncertainty statistics
    df['epistemic_std'] = np.sqrt(df['epistemic_variance'])
    df['aleatoric_std'] = np.sqrt(df['aleatoric_variance'])
    df['total_std'] = np.sqrt(df['total_variance'])
    
    # Add confidence intervals (approximate)
    df['position_score_lower_95'] = df['mean_position_score'] - 1.96 * df['epistemic_std']
    df['position_score_upper_95'] = df['mean_position_score'] + 1.96 * df['epistemic_std']
    
    return df