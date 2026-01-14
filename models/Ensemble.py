import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Literal


class Ensemble(nn.Module):
    """
    Ensemble model that combines predictions from multiple models.

    Supports different ensemble strategies:
    - 'average': Simple averaging of predictions
    - 'weighted': Weighted averaging based on provided weights
    - 'voting': Majority voting (for hard predictions)
    - 'soft_voting': Soft voting using probabilities
    """

    def __init__(
        self,
        models: List[nn.Module],
        strategy: Literal["average", "weighted", "voting", "soft_voting"] = "average",
        weights: List[float] | None = None,
    ):
        """
        Args:
            models: List of PyTorch models to ensemble
            strategy: Ensemble strategy ('average', 'weighted', 'voting', 'soft_voting')
            weights: Optional weights for each model (required for 'weighted' strategy)
        """
        super(Ensemble, self).__init__()

        if len(models) == 0:
            raise ValueError("At least one model is required for ensemble")

        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.num_models = len(models)

        # Setup weights
        if strategy == "weighted":
            if weights is None:
                raise ValueError("Weights must be provided for 'weighted' strategy")
            if len(weights) != len(models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")

            # Normalize weights to sum to 1
            weights_sum = sum(weights)
            self.weights = torch.tensor([w / weights_sum for w in weights])
        else:
            # Equal weights for other strategies
            self.weights = torch.ones(self.num_models) / self.num_models

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all models and combine predictions.

        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Combined predictions [B, num_classes]
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)

        # Stack predictions [num_models, B, num_classes]
        predictions = torch.stack(predictions)

        # Apply ensemble strategy
        if self.strategy == "average":
            return self._average_ensemble(predictions)
        elif self.strategy == "weighted":
            return self._weighted_ensemble(predictions)
        elif self.strategy == "voting":
            return self._voting_ensemble(predictions)
        elif self.strategy == "soft_voting":
            return self._soft_voting_ensemble(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _average_ensemble(self, predictions: Tensor) -> Tensor:
        """Simple averaging of predictions."""
        return predictions.mean(dim=0)

    def _weighted_ensemble(self, predictions: Tensor) -> Tensor:
        """Weighted averaging of predictions."""
        # Move weights to same device as predictions
        weights = self.weights.to(predictions.device).view(-1, 1, 1)
        weighted_preds = predictions * weights
        return weighted_preds.sum(dim=0)

    def _voting_ensemble(self, predictions: Tensor) -> Tensor:
        """
        Majority voting ensemble (hard voting).
        Takes the class with most votes across models.
        """
        # Get class predictions for each model
        class_preds = predictions.argmax(dim=2)  # [num_models, B]

        # Count votes for each class
        batch_size = class_preds.shape[1]
        num_classes = predictions.shape[2]

        votes = torch.zeros(batch_size, num_classes, device=predictions.device)
        for i in range(self.num_models):
            for b in range(batch_size):
                votes[b, class_preds[i, b]] += 1

        return votes

    def _soft_voting_ensemble(self, predictions: Tensor) -> Tensor:
        """
        Soft voting using averaged probabilities.
        Applies softmax to each model's logits, then averages probabilities.
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(predictions, dim=2)

        # Average probabilities
        avg_probs = probs.mean(dim=0)

        # Convert back to logits (optional, but maintains consistency)
        return torch.log(avg_probs + 1e-10)


class StackedEnsemble(nn.Module):
    """
    Stacked ensemble using a meta-learner.

    The base models make predictions, and a meta-model learns to combine them.
    This is more sophisticated than simple averaging or voting.
    """

    def __init__(
        self,
        base_models: List[nn.Module],
        num_classes: int,
        meta_hidden_dim: int = 128,
        freeze_base_models: bool = True,
    ):
        """
        Args:
            base_models: List of base models to ensemble
            num_classes: Number of output classes
            meta_hidden_dim: Hidden dimension for meta-learner
            freeze_base_models: Whether to freeze base models during training
        """
        super(StackedEnsemble, self).__init__()

        self.base_models = nn.ModuleList(base_models)
        self.num_models = len(base_models)
        self.num_classes = num_classes

        # Freeze base models if requested
        if freeze_base_models:
            for model in self.base_models:
                for param in model.parameters():
                    param.requires_grad = False

        # Meta-learner: takes concatenated predictions from all base models
        meta_input_dim = self.num_models * num_classes
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_hidden_dim, meta_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through base models and meta-learner.

        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Final predictions [B, num_classes]
        """
        # Get predictions from all base models
        base_predictions = []
        for model in self.base_models:
            with torch.no_grad() if not any(p.requires_grad for p in model.parameters()) else torch.enable_grad():
                pred = model(x)
                base_predictions.append(pred)

        # Concatenate all predictions [B, num_models * num_classes]
        stacked_preds = torch.cat(base_predictions, dim=1)

        # Pass through meta-learner
        final_pred = self.meta_learner(stacked_preds)

        return final_pred

    def unfreeze_base_models(self):
        """Unfreeze all base models for end-to-end training."""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = True

    def freeze_base_models(self):
        """Freeze base models (only train meta-learner)."""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    from CNN import CNN
    from ResNet import ResNet
    from VIT import ViT_Small

    # Example: Create ensemble of different models
    model1 = CNN(in_dim=3, out_dim=7, hidden_dim=64)
    model2 = ResNet(in_dim=3, out_dim=7, hidden_dim=64)
    model3 = ViT_Small(num_classes=7, img_size=64)

    # Simple averaging ensemble
    ensemble_avg = Ensemble([model1, model2, model3], strategy="average")

    # Weighted ensemble (give more weight to ViT)
    ensemble_weighted = Ensemble(
        [model1, model2, model3],
        strategy="weighted",
        weights=[0.2, 0.3, 0.5]
    )

    # Stacked ensemble with meta-learner
    ensemble_stacked = StackedEnsemble(
        [model1, model2, model3],
        num_classes=7,
        meta_hidden_dim=128
    )

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_avg = ensemble_avg.to(device)

    dummy_input = torch.randn(4, 3, 64, 64).to(device)
    output = ensemble_avg(dummy_input)

    print(f"Ensemble output shape: {output.shape}")
    print(f"Expected shape: [4, 7]")
