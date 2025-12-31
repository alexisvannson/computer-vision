"""
Transfer Learning Script for 95%+ Accuracy
Uses pretrained ResNet18/34/50 from ImageNet
"""
import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm


def calculate_class_weights(dataset):
    """Calculate class weights for handling imbalance."""
    class_counts = {}
    for _, label in dataset.imgs:
        class_counts[label] = class_counts.get(label, 0) + 1

    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]

    print("\n=== Class Distribution ===")
    for class_name, class_idx in dataset.class_to_idx.items():
        print(f"{class_name}: {class_counts[class_idx]} samples, weight: {weights[class_idx]:.4f}")
    print("=" * 30 + "\n")

    return torch.FloatTensor(weights)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return val_loss / len(val_loader), 100 * correct / total


def train_transfer_learning(
    model_name='resnet18',
    data_path='data/Dataset',
    batch_size=64,
    epochs=50,
    lr_backbone=1e-4,
    lr_head=1e-3,
    val_split=0.2,
    freeze_backbone=True,
    output_dir='models/checkpoints/transfer',
):
    """Train using transfer learning from pretrained ImageNet models."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_path, transform=train_transform)
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.classes}")

    # Calculate class weights
    class_weights = calculate_class_weights(full_dataset)

    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_indices_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create validation dataset with val transforms
    val_dataset_full = datasets.ImageFolder(root=data_path, transform=val_transform)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices_dataset.indices)

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # Load pretrained model
    print(f"\nLoading pretrained {model_name}...")
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Freeze backbone if requested
    if freeze_backbone:
        print("Freezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    model = model.to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer with different learning rates for backbone and head
    if freeze_backbone:
        # Only train the head
        optimizer = optim.Adam(model.fc.parameters(), lr=lr_head, weight_decay=1e-4)
    else:
        # Different LRs for backbone and head
        optimizer = optim.Adam([
            {'params': model.fc.parameters(), 'lr': lr_head},
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': lr_backbone}
        ], weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"training_log_{timestamp}.txt")

    with open(log_path, "w") as f:
        f.write(f"Transfer Learning Training\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Freeze backbone: {freeze_backbone}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"LR head: {lr_head}, LR backbone: {lr_backbone}\n")
        f.write("-" * 50 + "\n")

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Log to file
        with open(log_path, "a") as f:
            f.write(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, ")
            f.write(f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, LR={current_lr:.6f}\n")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = os.path.join(output_dir, f"{model_name}_best_acc{val_acc:.2f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best model saved! Accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break

        print("-" * 60)

    # Save final model
    final_path = os.path.join(output_dir, f"{model_name}_final.pth")
    torch.save(model.state_dict(), final_path)

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {output_dir}")
    print("=" * 60)

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning for Logo Classification")
    parser.add_argument("--model", type=str, default="resnet18",
                       choices=["resnet18", "resnet34", "resnet50"],
                       help="Pretrained model to use")
    parser.add_argument("--data", type=str, default="data/Dataset",
                       help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--lr_head", type=float, default=1e-3,
                       help="Learning rate for classification head")
    parser.add_argument("--lr_backbone", type=float, default=1e-4,
                       help="Learning rate for backbone (if not frozen)")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--unfreeze", action="store_true",
                       help="Unfreeze backbone for fine-tuning")
    parser.add_argument("--output", type=str, default="models/checkpoints/transfer",
                       help="Output directory for models")

    args = parser.parse_args()

    train_transfer_learning(
        model_name=args.model,
        data_path=args.data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        val_split=args.val_split,
        freeze_backbone=not args.unfreeze,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
