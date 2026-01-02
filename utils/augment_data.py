#!/usr/bin/env python3
"""
Offline data augmentation with class balancing for emoji classification.

This script generates augmented images to balance class distribution
in the emoji dataset, applying conservative transforms suitable for
emoji recognition tasks.
"""

import argparse
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Ensures same augmented dataset is generated each run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_dataset(source_dir):
    """
    Scan dataset directory and count images per class.

    Args:
        source_dir: Path to source dataset directory

    Returns:
        dict: {class_name: image_count}
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        sys.exit(1)

    class_counts = {}

    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue

        # Count PNG images only
        png_files = list(class_dir.glob('*.png'))

        if len(png_files) == 0:
            print(f"Warning: No PNG images found in {class_dir.name}/")
            continue

        class_counts[class_dir.name] = len(png_files)

    if len(class_counts) == 0:
        print(f"Error: No valid class directories with PNG images found in '{source_dir}'!")
        sys.exit(1)

    return class_counts


def calculate_augmentation_plan(class_counts, target_count=None):
    """
    Calculate how many augmented images needed per class to balance dataset.

    Args:
        class_counts: dict of {class_name: original_image_count}
        target_count: Target images per class (default: max class count)

    Returns:
        tuple: (augmentation_plan dict, target_count)
    """
    if target_count is None:
        target_count = max(class_counts.values())

    augmentation_plan = {}

    for class_name, count in class_counts.items():
        needed = max(0, target_count - count)

        if needed > 0:
            # How many augmented versions per original image
            augmentations_per_image = math.ceil(needed / count)
            total_augmented = needed
        else:
            augmentations_per_image = 0
            total_augmented = 0

        augmentation_plan[class_name] = {
            'original_count': count,
            'augmented_needed': total_augmented,
            'augmentations_per_image': augmentations_per_image
        }

    return augmentation_plan, target_count


def get_augmentation_transform():
    """
    Get conservative augmentation transform suitable for emoji classification.

    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05
        )
    ])


def augment_image_rgba(image_path, transform, output_path):
    """
    Augment RGBA image while preserving alpha channel.

    Args:
        image_path: Path to source image
        transform: torchvision transform to apply
        output_path: Path to save augmented image

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)

        # Handle different color modes - convert to RGBA
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Apply transforms (torchvision handles RGBA correctly for geometric ops)
        augmented = transform(img)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with PNG optimization
        augmented.save(output_path, 'PNG', optimize=True)

        return True

    except Exception as e:
        print(f"\nWarning: Failed to augment {image_path.name}: {e}")
        return False


def augment_class(class_name, class_info, source_dir, output_dir, transform, pbar):
    """
    Augment all images in a class directory.

    Args:
        class_name: Name of the class
        class_info: Dict with augmentation info for this class
        source_dir: Source dataset directory
        output_dir: Output directory for augmented images
        transform: Transform to apply
        pbar: tqdm progress bar
    """
    source_class_dir = Path(source_dir) / class_name
    output_class_dir = Path(output_dir) / class_name

    # Get list of source images
    source_images = sorted(source_class_dir.glob('*.png'))

    augmentations_per_image = class_info['augmentations_per_image']
    total_needed = class_info['augmented_needed']

    generated = 0

    # Cycle through source images and generate augmentations
    for img_idx in range(total_needed):
        # Select source image (cycle through if needed)
        source_img = source_images[img_idx % len(source_images)]

        # Determine augmentation number for this source image
        aug_num = (img_idx // len(source_images)) + 1

        # Create output filename
        output_name = f"{source_img.stem}_aug{aug_num}.png"
        output_path = output_class_dir / output_name

        # Augment image
        if augment_image_rgba(source_img, transform, output_path):
            generated += 1

        pbar.update(1)

    return generated


def create_combined_dataset(source_dir, augmented_dir, combined_dir):
    """
    Create combined dataset with original + augmented images.

    Args:
        source_dir: Original dataset directory
        augmented_dir: Augmented images directory
        combined_dir: Output combined directory
    """
    source_path = Path(source_dir)
    augmented_path = Path(augmented_dir)
    combined_path = Path(combined_dir)

    # Create combined directory
    combined_path.mkdir(parents=True, exist_ok=True)

    total_files = 0

    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        combined_class = combined_path / class_name
        combined_class.mkdir(parents=True, exist_ok=True)

        # Copy original images
        orig_count = 0
        for img_file in class_dir.glob('*.png'):
            shutil.copy2(img_file, combined_class / img_file.name)
            orig_count += 1

        # Copy augmented images (if any)
        aug_count = 0
        augmented_class = augmented_path / class_name
        if augmented_class.exists():
            for img_file in augmented_class.glob('*.png'):
                shutil.copy2(img_file, combined_class / img_file.name)
                aug_count += 1

        total = orig_count + aug_count
        total_files += total
        print(f"  {class_name:12s}: {orig_count:4d} orig + {aug_count:4d} aug = {total:4d} total")

    return total_files


def verify_augmentation(combined_dir, expected_per_class):
    """
    Verify that augmentation created balanced dataset.

    Args:
        combined_dir: Combined dataset directory
        expected_per_class: Expected number of images per class
    """
    combined_path = Path(combined_dir)

    all_balanced = True

    for class_dir in sorted(combined_path.iterdir()):
        if not class_dir.is_dir():
            continue

        count = len(list(class_dir.glob('*.png')))

        if count != expected_per_class:
            print(f"  WARNING: {class_dir.name} has {count} images (expected {expected_per_class})")
            all_balanced = False

    if all_balanced:
        print(f"  All classes balanced to {expected_per_class} images per class")

    return all_balanced


def print_augmentation_plan(class_counts, plan, target_count):
    """
    Print detailed augmentation plan.

    Args:
        class_counts: Original class counts
        plan: Augmentation plan
        target_count: Target images per class
    """
    print("\n" + "=" * 70)
    print("EMOJI DATASET OFFLINE AUGMENTATION")
    print("=" * 70)

    print(f"\nDataset Analysis:")
    print(f"  Total original images: {sum(class_counts.values())}")
    print(f"  Number of classes: {len(class_counts)}")
    print(f"  Target balanced count: {target_count} images per class")
    total_aug_needed = sum(p['augmented_needed'] for p in plan.values())
    print(f"  Total augmented needed: {total_aug_needed}")
    print(f"  Final dataset size: {target_count * len(class_counts)}")

    print("\n" + "━" * 70)
    print(f"{'Class':<12}  {'Original':<8}  {'Augmented':<10}  {'Per Image':<10}")
    print("━" * 70)

    for class_name in sorted(plan.keys()):
        info = plan[class_name]
        orig = info['original_count']
        aug = info['augmented_needed']
        per_img = info['augmentations_per_image']

        print(f"{class_name:<12}  {orig:<8}  {aug:<10}  {per_img}x")

    print("━" * 70)


def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(
        description='Offline data augmentation with class balancing for emoji dataset'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='data/Dataset',
        help='Source dataset directory (default: data/Dataset)'
    )

    parser.add_argument(
        '--output-augmented',
        type=str,
        default='data/Dataset_augmented',
        help='Output directory for augmented images only (default: data/Dataset_augmented)'
    )

    parser.add_argument(
        '--output-combined',
        type=str,
        default='data/Dataset_combined',
        help='Output directory for combined dataset (default: data/Dataset_combined)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--target-count',
        type=int,
        default=None,
        help='Target images per class (default: max class count)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print augmentation plan without generating images'
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Phase 1: Scan and plan
    print("\n[1/4] Scanning dataset...")
    class_counts = scan_dataset(args.source)
    plan, target_count = calculate_augmentation_plan(class_counts, args.target_count)

    # Print plan
    print_augmentation_plan(class_counts, plan, target_count)

    # If dry run, exit here
    if args.dry_run:
        print("\nDry run complete - no images generated.")
        return 0

    # Check if output directories exist
    output_augmented_path = Path(args.output_augmented)
    output_combined_path = Path(args.output_combined)

    if output_augmented_path.exists():
        print(f"\nWarning: Output directory '{args.output_augmented}' already exists!")
        response = input("Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return 1
        shutil.rmtree(output_augmented_path)

    if output_combined_path.exists():
        print(f"\nWarning: Output directory '{args.output_combined}' already exists!")
        response = input("Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return 1
        shutil.rmtree(output_combined_path)

    # Phase 2: Generate augmented images
    print("\n[2/4] Generating augmented images...")
    transform = get_augmentation_transform()
    total_to_augment = sum(p['augmented_needed'] for p in plan.values())

    with tqdm(total=total_to_augment, desc="Augmenting", unit="img") as pbar:
        for class_name, info in sorted(plan.items()):
            if info['augmented_needed'] > 0:
                augment_class(class_name, info, args.source, args.output_augmented, transform, pbar)

    # Phase 3: Create combined dataset
    print("\n[3/4] Creating combined dataset...")
    total_files = create_combined_dataset(args.source, args.output_augmented, args.output_combined)

    # Phase 4: Verification
    print("\n[4/4] Verifying results...")
    verify_augmentation(args.output_combined, target_count)

    print("\n" + "=" * 70)
    print("AUGMENTATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directories:")
    print(f"  Augmented only: {args.output_augmented}")
    print(f"  Combined dataset: {args.output_combined}")
    print(f"  Total files in combined: {total_files}")
    print(f"\nTo train with augmented data, update your config YAML:")
    print(f"  data:")
    print(f"    root: '{args.output_combined}'")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
