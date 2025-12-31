"""
Kaggle Inference with Test-Time Augmentation (TTA)
TTA can boost accuracy by 1-2% by averaging predictions from multiple augmented versions
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import functions from train_model
from utils.train_model import create_model, load_config


def get_class_names():
    """Get class names in alphabetical order matching ImageFolder."""
    classes = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
    return {i: class_name for i, class_name in enumerate(sorted(classes))}


def get_tta_transforms(img_size=224, normalize=True, mean=None, std=None):
    """
    Create multiple test-time augmentation transforms.

    Returns:
        List of transforms for TTA
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # Base normalization
    norm = transforms.Normalize(mean=mean, std=std) if normalize else transforms.Lambda(lambda x: x)

    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            norm
        ]),
        # Slight rotation left
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            norm
        ]),
        # Slight rotation right
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            norm
        ]),
        # Center crop with slight zoom
        transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm
        ]),
    ]

    return tta_transforms


def load_model_for_inference(model_name, weights_path, config, device):
    """Load model and weights for inference."""
    model = create_model(model_name, config)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded model: {model_name}")
    print(f"Loaded weights from: {weights_path}")
    print(f"Using device: {device}")

    return model


def run_inference_with_tta(model, test_dir, tta_transforms, device, idx_to_class, num_tta=5):
    """
    Run inference with Test-Time Augmentation.

    Args:
        model: Trained model
        test_dir: Directory containing test images
        tta_transforms: List of TTA transforms
        device: torch device
        idx_to_class: Dict mapping class index to class name
        num_tta: Number of TTA variations to use (max: len(tta_transforms))

    Returns:
        predictions: List of (image_id, predicted_class) tuples
    """
    test_images = sorted(Path(test_dir).glob("*.png"))

    if len(test_images) == 0:
        raise ValueError(f"No .png images found in {test_dir}")

    print(f"Found {len(test_images)} test images")
    print(f"Using {num_tta} TTA variations per image")

    # Limit TTA transforms to requested number
    tta_transforms = tta_transforms[:num_tta]

    predictions = []

    with torch.no_grad():
        for img_path in tqdm(test_images, desc="Running TTA inference"):
            image_id = img_path.stem
            image = Image.open(img_path).convert("RGB")

            # Collect predictions from all TTA variations
            all_logits = []

            for transform in tta_transforms:
                image_tensor = transform(image).unsqueeze(0).to(device)
                logits = model(image_tensor)
                all_logits.append(logits)

            # Average logits from all TTA variations
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            pred_idx = torch.argmax(avg_logits, dim=1).item()
            pred_class = idx_to_class[pred_idx]

            predictions.append((image_id, pred_class))

    return predictions


def run_inference_standard(model, test_dir, transform, device, idx_to_class):
    """Run standard inference without TTA."""
    test_images = sorted(Path(test_dir).glob("*.png"))

    if len(test_images) == 0:
        raise ValueError(f"No .png images found in {test_dir}")

    print(f"Found {len(test_images)} test images")

    predictions = []

    with torch.no_grad():
        for img_path in tqdm(test_images, desc="Running inference"):
            image_id = img_path.stem
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            logits = model(image_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_class = idx_to_class[pred_idx]

            predictions.append((image_id, pred_class))

    return predictions


def create_submission_csv(predictions, output_path):
    """Create submission CSV file."""
    df = pd.DataFrame(predictions, columns=["Id", "Label"])

    # Sort by Id numerically
    df["Id"] = df["Id"].astype(int)
    df = df.sort_values("Id")
    df["Id"] = df["Id"].astype(str)

    df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"\nFirst few predictions:")
    print(df.head(10))
    print(f"\nLast few predictions:")
    print(df.tail(10))

    # Show class distribution
    print(f"\nPrediction distribution:")
    print(df["Label"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser(description="Kaggle inference with optional TTA")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--test-dir", type=str, default="data/test", help="Test images directory")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--num-tta", type=int, default=5, help="Number of TTA variations (max 5)")

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

    # Get class mapping
    idx_to_class = get_class_names()

    # Get transform config
    transform_config = config.get("transforms", {})
    img_size = transform_config.get("img_size", 224)
    normalize = transform_config.get("normalize", True)
    mean = transform_config.get("mean", [0.485, 0.456, 0.406])
    std = transform_config.get("std", [0.229, 0.224, 0.225])

    # Run inference
    if args.tta:
        print("\n=== Test-Time Augmentation ENABLED ===")
        print("This will take longer but may improve accuracy by 1-2%\n")
        tta_transforms = get_tta_transforms(img_size, normalize, mean, std)
        predictions = run_inference_with_tta(
            model, args.test_dir, tta_transforms, device, idx_to_class, args.num_tta
        )
    else:
        print("\n=== Standard Inference (no TTA) ===")
        print("Use --tta flag to enable Test-Time Augmentation\n")
        # Standard single transform
        from utils.train_model import get_transforms
        transform = get_transforms(transform_config)
        predictions = run_inference_standard(model, args.test_dir, transform, device, idx_to_class)

    # Create submission CSV
    create_submission_csv(predictions, args.output)

    print("\n" + "=" * 60)
    print("Inference complete! Submit your CSV to Kaggle.")
    print("=" * 60)


if __name__ == "__main__":
    main()
