import torch
import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        activation: str = "ReLU",
        initializer: str | None = None,
        norm_type: str | None = "BatchNorm2d",
    ):
        """
        Flexible Convolutional Neural Network.
        """
        super(CNN, self).__init__()

        act_fn = getattr(nn, activation)()

        layers = []
        layers.append(nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1))
        if norm_type:
            layers.append(getattr(nn, norm_type)(hidden_dim))
        layers.append(act_fn)
        layers.append(nn.MaxPool2d(2, 2))  # Downsample by 2

        for _ in range(hidden_layers - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
            if norm_type:
                layers.append(getattr(nn, norm_type)(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.MaxPool2d(2, 2))  # Downsample by 2

        self.features = nn.Sequential(*layers)

        # AdaptiveAvgPool2d((1,1)) forces output to be (Batch, hidden_dim, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_dim, out_dim)

        # Optional Initialization
        if initializer is not None:
            init_fn = getattr(nn.init, initializer)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    init_fn(m.weight)

    def forward(self, x: Tensor):
        x = self.features(x)
        x = self.global_pool(x)  # Global Pooling (Batch, Channel, H, W) -> (Batch, Channel, 1, 1)
        x = torch.flatten(x, 1)  # Flatten (Batch, Channel, 1, 1) -> (Batch, Channel)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNN(
        in_dim=3,  # 3 for RGB
        out_dim=10,  # Number of classes in your dataset
        hidden_dim=128,  # Filters per layer
        hidden_layers=3,  # Depth
        norm_type="BatchNorm2d",  # Standard for CNNs
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model)
