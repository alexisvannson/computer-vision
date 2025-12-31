import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Model Registry: Maps model names to (module_path, class_name)
MODEL_REGISTRY = {
    "senet": ("models.SENET", "SENet"),
    "vit": ("models.VIT", "VisionTransformer"),
    "mlp": ("models.MLP", "MLP"),
    "cnn": ("models.CNN", "CNN"),
    "resnet": ("models.ResNet", "ResNet"),
}


def register_model(name, module_path, class_name):
    """Register a new model in the registry."""
    MODEL_REGISTRY[name.lower()] = (module_path, class_name)


def get_registered_models():
    """Get list of all registered model names."""
    return list(MODEL_REGISTRY.keys())


def calculate_class_weights(dataset_path):
    """Calculate class weights for handling imbalance."""
    class_counts = {}

    # Count samples per class
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count

    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = []

    for class_name in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_name])
        weights.append(weight)
        print(f"Class '{class_name}': {class_counts[class_name]} samples, weight: {weight:.4f}")

    return torch.FloatTensor(weights)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sample, label in val_loader:
            sample = sample.to(device)
            label = label.to(device)

            logits = model(sample)
            loss = criterion(logits, label)

            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience=10,
    output_path="weights",
    weights_name="final_model",
    start_weights=None,
):
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    if start_weights:
        model.load_state_dict(torch.load(start_weights, map_location=device))

    os.makedirs(output_path, exist_ok=True)
    print(f"Training model in {output_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_path, f"training_logs_{timestamp}.txt")

    with open(log_path, "w") as the_file:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        the_file.write(f"Training started at: {timestamp_str}\n")
        the_file.write(f"Epochs: {epochs}, Patience: {patience}\n")
        the_file.write(f"Output path: {output_path}\n")
        the_file.write(f"Device: {device}\n")
        the_file.write("-" * 50 + "\n")

    print('Start training with validation tracking')

    for epoch in range(epochs):
        checkpoint1 = time.time()
        epoch_loss = 0
        num_batches = 0
        model.train()

        for sample, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            sample = sample.to(device)
            label = label.to(device)

            logits = model(sample)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']

        checkpoint2 = time.time()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # Save logs
        with open(log_path, "a") as the_file:
            the_file.write(f"Epoch {epoch+1}/{epochs}\n")
            the_file.write(f"  Train Loss: {avg_train_loss:.4f}\n")
            the_file.write(f"  Val Loss: {val_loss:.4f}\n")
            the_file.write(f"  Val Acc: {val_acc:.2f}%\n")
            the_file.write(f"  Learning Rate: {current_lr:.6f}\n")
            time_mins = (checkpoint2 - checkpoint1) / 60
            the_file.write(f"  Time: {time_mins:.2f} minutes\n")

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(output_path, f"best_model_epoch{epoch+1}_acc{val_acc:.2f}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model! Accuracy: {val_acc:.2f}% - Saved: {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation accuracy: {best_val_acc:.2f}%")
            break

    # Save final model
    final_model_path = os.path.join(output_path, f"{weights_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    with open(log_path, "a") as the_file:
        the_file.write("-" * 50 + "\n")
        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        the_file.write(f"Training completed at: {completion_time}\n")
        the_file.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        the_file.write(f"Best validation loss: {best_val_loss:.4f}\n")

    return best_val_acc


def load_config(model_name):
    """Load configuration for a specific model."""
    config_path = os.path.join("config", f"{model_name.lower()}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_name, config):
    """Create model instance based on name and config using registry."""
    model_key = model_name.lower()

    if model_key not in MODEL_REGISTRY:
        available_models = ", ".join(get_registered_models())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

    module_path, class_name = MODEL_REGISTRY[model_key]
    module = __import__(module_path, fromlist=[class_name])
    ModelClass = getattr(module, class_name)
    model = ModelClass(**config["model_params"])

    return model


def get_transforms(config, augment=False):
    """Create transforms based on config with optional augmentation."""
    img_size = config.get("img_size", 224)

    if augment:
        # Training transforms with augmentation
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.05
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
    else:
        # Validation/test transforms - no augmentation
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]

    # Add normalization if specified
    if config.get("normalize", True):
        transform_list.append(
            transforms.Normalize(
                mean=config.get("mean", [0.485, 0.456, 0.406]),
                std=config.get("std", [0.229, 0.224, 0.225]),
            )
        )

    return transforms.Compose(transform_list)


def main():
    parser = argparse.ArgumentParser(description="Enhanced training script with validation and augmentation")
    parser.add_argument("--model", type=str, required=True, help="Model name to train")
    parser.add_argument("--config", type=str, default=None, help="Override default config file path")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    args = parser.parse_args()

    model_name = args.model

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(model_name)

    # Create model
    model = create_model(model_name, config)

    # Setup transforms
    transform_config = config.get("transforms", {})
    train_transform = get_transforms(transform_config, augment=args.augment)
    val_transform = get_transforms(transform_config, augment=False)

    data_config = config.get("data", {})
    root_path = data_config.get("root", "data/Dataset")

    # Handle case where root is a list
    if isinstance(root_path, list):
        for path in root_path:
            if os.path.exists(path) and os.path.isdir(path):
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    root_path = path
                    print(f"Using dataset path: {path}")
                    break
        else:
            root_path = root_path[0]

    # Calculate class weights for handling imbalance
    print("\n=== Class Distribution ===")
    class_weights = calculate_class_weights(root_path)
    print("=" * 30 + "\n")

    # Load full dataset with training transforms
    full_dataset = datasets.ImageFolder(root=root_path, transform=train_transform)

    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset_temp = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create validation dataset with validation transforms (no augmentation)
    # We need to apply val_transform to the validation split
    val_dataset_full = datasets.ImageFolder(root=root_path, transform=val_transform)
    val_indices = val_dataset_temp.indices
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    print(f"Dataset split: {train_size} training, {val_size} validation")

    # Create data loaders
    batch_size = data_config.get("batch_size", 32)
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Setup training components
    train_config = config.get("training", {})

    # Weighted loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer
    lr = train_config.get("learning_rate", 0.001)
    weight_decay = train_config.get("weight_decay", 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    # Handle output_path list
    output_path = train_config.get("output_paths", train_config.get("output_path", "models/checkpoints"))
    if isinstance(output_path, list):
        for path in output_path:
            parent = os.path.dirname(path) if os.path.dirname(path) else "."
            if os.path.exists(parent) and os.access(parent, os.W_OK):
                output_path = path
                break
        else:
            output_path = output_path[0]

    print(f"Output path: {output_path}")
    print(f"Data augmentation: {'ENABLED' if args.augment else 'DISABLED'}")
    print(f"Initial learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {batch_size}\n")

    # Train model
    best_acc = train(
        model=model,
        train_loader=trainloader,
        val_loader=valloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_path=output_path,
        weights_name=f"{model_name}_enhanced",
        epochs=train_config.get("epochs", 50),
        patience=train_config.get("patience", 10),
    )

    print(f"\n{'='*50}")
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
