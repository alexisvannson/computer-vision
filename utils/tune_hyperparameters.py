"""
Hyperparameter tuning script using Optuna for Bayesian optimization.

This script performs hyperparameter search for CNN models to find optimal
configurations. It uses Optuna's TPESampler for intelligent search and
MedianPruner to stop unpromising trials early.

Usage:
    python utils/tune_hyperparameters.py --model cnn --n-trials 50
"""

import argparse
import os
import pickle
from typing import Tuple

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import utility functions from train_model
from train_model import load_config, create_model, get_transforms


def create_train_val_split(dataset, train_ratio=0.8, seed=42):
    """
    Create stratified train/validation split.

    Args:
        dataset: ImageFolder dataset
        train_ratio: Proportion of data for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        train_subset, val_subset: PyTorch Subset objects
    """
    # Get labels for stratification
    if hasattr(dataset, 'samples'):
        labels = [label for _, label in dataset.samples]
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        raise ValueError("Dataset must have 'samples' or 'targets' attribute")

    # Create stratified split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed
    )

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    print(f"Train size: {len(train_subset)}, Validation size: {len(val_subset)}")
    return train_subset, val_subset


def create_optimizer(opt_type, params, lr, weight_decay, momentum=0.9):
    """
    Create optimizer based on type.

    Args:
        opt_type: Optimizer type ('Adam', 'AdamW', or 'SGD')
        params: Model parameters
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum for SGD (default: 0.9)

    Returns:
        PyTorch optimizer instance
    """
    if opt_type == 'Adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'AdamW':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == 'SGD':
        return optim.SGD(
            params, lr=lr, weight_decay=weight_decay,
            momentum=momentum, nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def find_data_root(paths):
    """
    Find first existing path from a list of potential paths.

    Args:
        paths: List of potential paths or single path string

    Returns:
        First existing path

    Raises:
        FileNotFoundError: If no path exists
    """
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"None of the specified paths exist: {paths}")


def train_with_validation(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=25,
    patience=7,
    trial=None
):
    """
    Train model and return validation loss.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epochs: Maximum number of epochs (default: 25)
        patience: Early stopping patience (default: 7)
        trial: Optuna trial for pruning (optional)

    Returns:
        best_val_loss: Best validation loss achieved
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch, labels in val_loader:
                batch, labels = batch.to(device), labels.to(device)
                outputs = model(batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Report to Optuna for pruning
        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return best_val_loss


def objective(trial, base_config, data_root, device, model_name):
    """
    Optuna objective function to minimize validation loss.

    Args:
        trial: Optuna trial object
        base_config: Base configuration from YAML
        data_root: Path to dataset
        device: torch.device
        model_name: Name of the model being tuned

    Returns:
        val_loss: Validation loss (to minimize)
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    hidden_layers = trial.suggest_int('hidden_layers', 2, 5)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optimizer_type = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])

    # Conditional parameter for SGD
    momentum = 0.9  # default
    if optimizer_type == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)

    # Create model with suggested architecture
    model_params = base_config['model_params'].copy()
    model_params['hidden_dim'] = hidden_dim
    model_params['hidden_layers'] = hidden_layers

    # Create model using the same infrastructure as train_model.py
    config_with_params = {'model_params': model_params}
    model = create_model(model_name, config_with_params).to(device)

    # Load and split dataset
    transform = get_transforms(base_config['transforms'])
    full_dataset = datasets.ImageFolder(root=data_root, transform=transform)
    train_subset, val_subset = create_train_val_split(full_dataset)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Create optimizer
    optimizer = create_optimizer(
        optimizer_type,
        model.parameters(),
        lr,
        weight_decay,
        momentum
    )

    # Train and evaluate
    criterion = nn.CrossEntropyLoss()
    val_loss = train_with_validation(
        model, train_loader, val_loader, criterion, optimizer,
        device, epochs=25, patience=7, trial=trial
    )

    return val_loss


def create_best_config(best_params, base_config):
    """
    Create YAML config from best parameters.

    Args:
        best_params: Dictionary of best hyperparameters
        base_config: Base configuration to use as template

    Returns:
        Dictionary containing complete configuration
    """
    config = {
        'model_params': {
            'in_dim': base_config['model_params']['in_dim'],
            'out_dim': base_config['model_params']['out_dim'],
            'hidden_dim': best_params['hidden_dim'],
            'hidden_layers': best_params['hidden_layers'],
            'activation': base_config['model_params'].get('activation', 'ReLU'),
            'norm_type': base_config['model_params'].get('norm_type', 'BatchNorm2d')
        },
        'transforms': base_config['transforms'],
        'data': base_config['data'].copy(),
        'training': {
            'epochs': 50,  # Use full epochs for final training
            'patience': 10,
            'learning_rate': best_params['learning_rate'],
            'weight_decay': best_params['weight_decay'],
            'optimizer': best_params['optimizer'],
            'output_paths': [
                f'models/checkpoints/{base_config.get("model_name", "model")}_tuned',
                f'/content/drive/MyDrive/computer-vision/checkpoints/{base_config.get("model_name", "model")}_tuned'
            ]
        }
    }

    # Update batch size in data config
    config['data']['batch_size'] = best_params['batch_size']

    # Add momentum if SGD
    if best_params['optimizer'] == 'SGD':
        config['training']['momentum'] = best_params.get('momentum', 0.9)

    return config


def save_results(study, model_name, output_dir, base_config):
    """
    Save study results and generate visualizations.

    Args:
        study: Optuna study object
        model_name: Name of the model
        output_dir: Base output directory
        base_config: Base configuration used for tuning
    """
    # Create output directories
    model_output = os.path.join(output_dir, model_name)
    plots_dir = os.path.join(model_output, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Save study object
    study_path = os.path.join(model_output, 'study.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Saved study to {study_path}")

    # 2. Save results to CSV
    df = study.trials_dataframe()
    csv_path = os.path.join(model_output, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # 3. Save best hyperparameters as YAML
    best_params = study.best_params
    best_config = create_best_config(best_params, base_config)
    yaml_path = os.path.join(model_output, 'best_config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved best config to {yaml_path}")

    # 4. Generate visualizations
    print("Generating visualizations...")
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, 'optimization_history.png'))

        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, 'param_importances.png'))

        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image(os.path.join(plots_dir, 'parallel_coordinate.png'))

        # Slice plot
        fig = plot_slice(study)
        fig.write_image(os.path.join(plots_dir, 'slice_plot.png'))

        print(f"Generated plots in {plots_dir}")
    except Exception as e:
        print(f"Warning: Could not generate some visualizations: {e}")
        print("You may need to install kaleido: pip install kaleido")

    # Print best trial summary
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60)


def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., cnn)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--config', type=str, default=None,
                        help='Override default config path')
    parser.add_argument('--output-dir', type=str,
                        default='hyperparameter_tuning',
                        help='Output directory for results (default: hyperparameter_tuning)')
    args = parser.parse_args()

    # Load base config
    if args.config is None:
        config = load_config(args.model)
    else:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Add model name to config for reference
    config['model_name'] = args.model

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find dataset path
    data_root = find_data_root(config['data']['root'])
    print(f"Using data from: {data_root}")

    # Create study
    print(f"\nStarting hyperparameter tuning with {args.n_trials} trials...")
    print("Search space:")
    print("  hidden_dim: [32, 64, 128, 256]")
    print("  hidden_layers: [2, 3, 4, 5]")
    print("  learning_rate: [1e-5, 1e-2] (log scale)")
    print("  batch_size: [16, 32, 64, 128]")
    print("  weight_decay: [1e-6, 1e-3] (log scale)")
    print("  optimizer: ['Adam', 'AdamW', 'SGD']")
    print("  momentum: [0.8, 0.99] (only for SGD)")
    print()

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # Don't prune first 10 trials
            n_warmup_steps=5      # Wait 5 epochs before pruning
        )
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, data_root, device, args.model),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    # Save results
    save_results(study, args.model, args.output_dir, config)

    print(f"\nTo train with best hyperparameters:")
    print(f"  python utils/train_model.py --model {args.model} --config {args.output_dir}/{args.model}/best_config.yaml")


if __name__ == "__main__":
    main()
