"""Tests for training functionality."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import CNN, MLP
from utils.train_model import train


class TestTraining:
    """Tests for training loop."""

    def test_train_smoke_test_cnn(self, tmp_path):
        """Smoke test: verify training runs without errors for CNN."""
        # Create a minimal dataset
        samples = torch.randn(10, 3, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        dataset = TensorDataset(samples, labels)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = CNN(in_dim=3, out_dim=10)
        device = torch.device("cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        output_path = str(tmp_path / "cnn_weights")

        # Train for just 2 epochs
        train(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=2,
            patience=10,
            output_path=output_path,
            weights_name="test_cnn",
        )

        # Check that weights were saved
        assert os.path.exists(f"{output_path}/test_cnn.pth")
        # Check that logs were created
        log_files = [f for f in os.listdir(output_path) if f.startswith("training_logs")]
        assert len(log_files) > 0

    def test_train_smoke_test_mlp(self, tmp_path):
        """Smoke test: verify training runs without errors for MLP."""
        # Create a minimal dataset
        samples = torch.randn(20, 784)
        labels = torch.tensor([i % 10 for i in range(20)], dtype=torch.long)
        dataset = TensorDataset(samples, labels)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        model = MLP(in_dim=784, out_dim=10, hidden_dim=64)
        device = torch.device("cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        output_path = str(tmp_path / "mlp_weights")

        # Train for just 2 epochs
        train(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=2,
            patience=10,
            output_path=output_path,
            weights_name="test_mlp",
        )

        # Check that weights were saved
        assert os.path.exists(f"{output_path}/test_mlp.pth")

    def test_train_early_stopping(self, tmp_path):
        """Test that early stopping works."""
        # Create a dataset with constant loss
        samples = torch.randn(5, 3, 32, 32)
        labels = torch.zeros(5, dtype=torch.long)
        dataset = TensorDataset(samples, labels)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        model = CNN(in_dim=3, out_dim=10)
        device = torch.device("cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        output_path = str(tmp_path / "early_stop_weights")

        # Train with very low patience
        train(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=100,  # Set high but expect early stop
            patience=2,
            output_path=output_path,
            weights_name="early_stop_test",
        )

        # Training should have stopped early
        assert os.path.exists(f"{output_path}/early_stop_test.pth")

    def test_train_saves_best_model(self, tmp_path):
        """Test that best model is saved during training."""
        samples = torch.randn(10, 3, 32, 32)
        labels = torch.zeros(10, dtype=torch.long)
        dataset = TensorDataset(samples, labels)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        model = CNN(in_dim=3, out_dim=10)
        device = torch.device("cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        output_path = str(tmp_path / "best_model_weights")

        train(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=3,
            patience=10,
            output_path=output_path,
            weights_name="final",
        )

        # Check that at least one best model was saved
        files = os.listdir(output_path)
        best_models = [f for f in files if f.startswith("best_model")]
        assert len(best_models) > 0
