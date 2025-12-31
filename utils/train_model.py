import argparse
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

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
    """
    Register a new model in the registry.

    Args:
        name: Model name (e.g., 'resnet')
        module_path: Python module path (e.g., 'models.ResNet')
        class_name: Class name in the module (e.g., 'ResNet50')

    Example:
        register_model('resnet', 'models.ResNet', 'ResNet50')
    """
    MODEL_REGISTRY[name.lower()] = (module_path, class_name)


def get_registered_models():
    """Get list of all registered model names."""
    return list(MODEL_REGISTRY.keys())


def train(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience=5,
    output_path="weights",
    weights_name="final_model",
    start_weights=None,
):
    best_loss = float("inf")
    patience_counter = 0

    if start_weights:
        model.load_state_dict(torch.load(start_weights, map_location=device))

    # Create the full output path directory structure
    os.makedirs(output_path, exist_ok=True)
    print(f"Training model in {output_path}")

    # Create a timestamped log file for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_path, f"training_logs_{timestamp}.txt")

    # Write training start info
    with open(log_path, "w") as the_file:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        the_file.write(f"Training started at: {timestamp_str}\n")
        the_file.write(f"Epochs: {epochs}, Patience: {patience}\n")
        the_file.write(f"Output path: {output_path}\n")
        the_file.write(f"Device: {device}\n")
        the_file.write("-" * 50 + "\n")

    print('start training')

    for epoch in range(epochs):
        checkpoint1 = time.time()
        epoch_loss = 0
        num_batches = 0
        model.train()
        for sample, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Move data to device
            sample = sample.to(device)
            label = label.to(device)

            logits = model(sample)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}")
        checkpoint2 = time.time()
        print(f"epoch: {epoch + 1} needed {checkpoint2 - checkpoint1} time")
        # Save training logs in the same directory as the weights
        with open(log_path, "a") as the_file:
            the_file.write(f"Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}\n")
            time_mins = (checkpoint2 - checkpoint1) / 60
            the_file.write(f"Epoch {epoch+1}/{epochs}, needed {time_mins:.2f} minutes\n")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model in the same directory as final model
            best_model_path = os.path.join(output_path, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model: {best_model_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model in the same directory as best models
    final_model_path = os.path.join(output_path, f"{weights_name}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Write training completion info to log
    with open(log_path, "a") as the_file:
        the_file.write("-" * 50 + "\n")
        completion_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        the_file.write(f"Training completed at: {completion_time}\n")
        the_file.write(f"Best loss achieved: {best_loss:.4f}\n")
        the_file.write(f"Final model saved: {final_model_path}\n")


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
        raise ValueError(f"Unknown model: {model_name}. " f"Available models: {available_models}")

    # Get module and class name from registry
    module_path, class_name = MODEL_REGISTRY[model_key]

    # Dynamically import the model class
    module = __import__(module_path, fromlist=[class_name])
    ModelClass = getattr(module, class_name)

    # Instantiate model with config parameters
    model = ModelClass(**config["model_params"])

    return model


def get_transforms(config):
    """Create transforms based on config."""
    transform_list = []

    # Resize if specified
    if "img_size" in config:
        transform_list.append(transforms.Resize((config["img_size"], config["img_size"])))

    transform_list.append(transforms.ToTensor())

    # Normalize if specified
    if "normalize" in config and config["normalize"]:
        transform_list.append(
            transforms.Normalize(
                mean=config.get("mean", [0.485, 0.456, 0.406]),
                std=config.get("std", [0.229, 0.224, 0.225]),
            )
        )

    return transforms.Compose(transform_list)


def main():
    parser = argparse.ArgumentParser(description="Train model script")
    parser.add_argument("--model", type=str, required=True, help="Model name to train")
    parser.add_argument(
        "--config", type=str, default=None, help="Override default config file path"
    )
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
    transform = get_transforms(config.get("transforms", {}))

    data_config = config.get("data", {})
    root_path = data_config.get("root", "data/Dataset")

    # Handle case where root is a list (for Colab/local compatibility)
    if isinstance(root_path, list):
        # Try to find the first path that exists
        for path in root_path:
            if os.path.exists(path):
                root_path = path
                break
        else:
            # If none exist, use the first one
            root_path = root_path[0]

    trainset = datasets.ImageFolder(
        root=root_path, transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=data_config.get("batch_size", 32),
        shuffle=data_config.get("shuffle", True),
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Setup training components
    train_config = config.get("training", {})
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config.get("learning_rate", 0.001))

    # Train model
    train(
        model=model,
        train_loader=trainloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_path=train_config.get("output_path", "models/checkpoints"),
        weights_name=model_name,
        epochs=train_config.get("epochs", 20),
        patience=train_config.get("patience", 5),
    )


if __name__ == "__main__":
    main()
