import torch 
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):

        """
        Flexible Convolutional Neural Network.
        """

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()



import torch.nn as nn
from torch import Tensor

# Definition of a MLP
class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        activation: str = "ReLU",
        initializer: None | str = None,
        norm_type: None | str = "LayerNorm",
    ):
        """
        Flexible Multi-layer perceptron.
        """

        super(MLP, self).__init__()
        self.activation = getattr(nn, activation)()
        if initializer is not None:
            self.initializer = getattr(nn.init, initializer)
        layers = [nn.Linear(in_dim, hidden_dim), self.activation]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), self.activation]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "BatchNorm1d",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

        if initializer is not None:
            params = self.model.parameters()
            for param in params:
                if param.requires_grad and len(param.shape) > 1:
                    self.initializer(param)

    def forward(self, x: Tensor):
        x = x.view(x.size(0), -1)  # Flatten: (batch, channels*height*width)
        return self.model(x.float())