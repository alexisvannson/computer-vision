"""Tests for neural network models."""

import pytest
import torch

from models import CNN, MLP


class TestCNN:
    """Tests for CNN model."""

    def test_cnn_initialization(self):
        """Test that CNN can be initialized."""
        model = CNN()
        assert model is not None

    def test_cnn_forward_pass(self):
        """Test CNN forward pass with correct input shape."""
        model = CNN()
        # Batch size=2, 3 channels (RGB), 32x32 image
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        # Should output 10 classes for batch of 2
        assert output.shape == (2, 10)

    def test_cnn_single_sample(self):
        """Test CNN with single sample."""
        model = CNN()
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output.shape == (1, 10)

    def test_cnn_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = CNN()
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestMLP:
    """Tests for MLP model."""

    def test_mlp_initialization(self):
        """Test that MLP can be initialized with custom parameters."""
        model = MLP(in_dim=784, out_dim=10, hidden_dim=128, hidden_layers=2)
        assert model is not None

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        model = MLP(in_dim=784, out_dim=10, hidden_dim=128)
        x = torch.randn(2, 784)
        output = model(x)
        assert output.shape == (2, 10)

    def test_mlp_with_image_input(self):
        """Test MLP flattens image input correctly."""
        model = MLP(in_dim=3 * 32 * 32, out_dim=10)
        # Image input: batch_size=2, 3 channels, 32x32
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_mlp_different_hidden_layers(self):
        """Test MLP with different number of hidden layers."""
        for n_layers in [1, 2, 3, 5]:
            model = MLP(in_dim=100, out_dim=10, hidden_dim=64, hidden_layers=n_layers)
            x = torch.randn(4, 100)
            output = model(x)
            assert output.shape == (4, 10)

    def test_mlp_activation_functions(self):
        """Test MLP with different activation functions."""
        for activation in ["ReLU", "Tanh", "GELU"]:
            model = MLP(in_dim=100, out_dim=10, activation=activation)
            x = torch.randn(2, 100)
            output = model(x)
            assert output.shape == (2, 10)

    def test_mlp_with_layer_norm(self):
        """Test MLP with layer normalization."""
        model = MLP(in_dim=100, out_dim=10, norm_type="LayerNorm")
        x = torch.randn(2, 100)
        output = model(x)
        assert output.shape == (2, 10)

    def test_mlp_with_batch_norm(self):
        """Test MLP with batch normalization."""
        model = MLP(in_dim=100, out_dim=10, norm_type="BatchNorm1d")
        x = torch.randn(2, 100)
        output = model(x)
        assert output.shape == (2, 10)

    def test_mlp_gradient_flow(self):
        """Test that gradients flow through the network."""
        model = MLP(in_dim=100, out_dim=10)
        x = torch.randn(2, 100)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestModelSaveLoad:
    """Tests for model saving and loading."""

    def test_cnn_save_load(self, tmp_path):
        """Test CNN state dict save and load."""
        model = CNN()
        save_path = tmp_path / "test_cnn.pth"

        # Save model
        torch.save(model.state_dict(), save_path)

        # Load into new model
        new_model = CNN()
        new_model.load_state_dict(torch.load(save_path))

        # Test that loaded model works
        x = torch.randn(1, 3, 32, 32)
        output = new_model(x)
        assert output.shape == (1, 10)

    def test_mlp_save_load(self, tmp_path):
        """Test MLP state dict save and load."""
        model = MLP(in_dim=784, out_dim=10, hidden_dim=128)
        save_path = tmp_path / "test_mlp.pth"

        # Save model
        torch.save(model.state_dict(), save_path)

        # Load into new model
        new_model = MLP(in_dim=784, out_dim=10, hidden_dim=128)
        new_model.load_state_dict(torch.load(save_path))

        # Test that loaded model works
        x = torch.randn(1, 784)
        output = new_model(x)
        assert output.shape == (1, 10)
