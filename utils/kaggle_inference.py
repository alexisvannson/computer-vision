import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import functions from train_model
from utils.train_model import create_model, get_transforms, load_config


def get_class_names():
    """
    Get class names in alphabetical order (matching ImageFolder's class_to_idx).
    Returns dict mapping index to class name.
    """
    classes = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
    return {i: class_name for i, class_name in enumerate(sorted(classes))}


def load_model_for_inference(model_name, weights_path, config, device):
    """
    Load model and weights for inference.

    Args:
        model_name: Name of the model (e.g., 'resnet', 'cnn')
        weights_path: Path to the .pth weights file
        config: Model configuration dict
        device: torch device

    Returns:
        model: Loaded model in eval mode
    """
    # Create model
    model = create_model(model_name, config)

    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded model: {model_name}")
    print(f"Loaded weights from: {weights_path}")
    print(f"Using device: {device}")

    return model


def run_inference(model, test_dir, transform, device, idx_to_class):
    """
    Run inference on all test images.

    Args:
        model: Trained model
        test_dir: Directory containing test images
        transform: Image transforms
        device: torch device
        idx_to_class: Dict mapping class index to class name

    Returns:
        predictions: List of (image_id, predicted_class) tuples
    """
    # Get all test images
    test_images = sorted(Path(test_dir).glob("*.png"))

    if len(test_images) == 0:
        raise ValueError(f"No .png images found in {test_dir}")

    print(f"Found {len(test_images)} test images")

    predictions = []

    with torch.no_grad():
        for img_path in tqdm(test_images, desc="Running inference"):
            # Extract image ID from filename (e.g., "10001.png" -> "10001")
            image_id = img_path.stem

            # Load and transform image
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            image_tensor = image_tensor.to(device)

            # Get prediction
            logits = model(image_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_class = idx_to_class[pred_idx]

            predictions.append((image_id, pred_class))

    return predictions


def create_submission_csv(predictions, output_path):
    """
    Create submission CSV file.

    Args:
        predictions: List of (image_id, predicted_class) tuples
        output_path: Path to output CSV file
    """
    # Convert to DataFrame
    df = pd.DataFrame(predictions, columns=["Id", "Label"])

    # Sort by Id (numerically)
    df["Id"] = df["Id"].astype(int)
    df = df.sort_values("Id")
    df["Id"] = df["Id"].astype(str)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"\nFirst few predictions:")
    print(df.head(10))


def main():
    parser = argparse.ArgumentParser(description="Kaggle inference script for emoji classification")
    parser.add_argument(
        "--model", type=str, required=True, help="Model name (e.g., resnet, cnn, mlp)"
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model weights (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (defaults to config/{model}.yaml)",
    )
    parser.add_argument(
        "--test-dir", type=str, default="data/test", help="Directory containing test images"
    )
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV file path")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.model)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model_for_inference(args.model, args.weights, config, device)

    # Get transforms
    transform = get_transforms(config.get("transforms", {}))

    # Get class mapping
    idx_to_class = get_class_names()

    # Run inference
    predictions = run_inference(model, args.test_dir, transform, device, idx_to_class)

    # Create submission CSV
    create_submission_csv(predictions, args.output)


if __name__ == "__main__":
    main()
