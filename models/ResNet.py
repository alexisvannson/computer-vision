import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """
    A standard ResNet block: Two 3x3 convolutions with a skip connection.
    """

    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, norm_type="BatchNorm2d"
    ):
        super(ResidualBlock, self).__init__()

        NormLayer = getattr(nn, norm_type)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = NormLayer(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = NormLayer(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # If the input shape/channels don't match the output,
        # apply the downsample layer to the identity so we can add them.
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip Connection
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_dim: int,  # Input channels (3 for RGB)
        out_dim: int,  # Number of classes
        hidden_dim: int = 64,  # Starting number of filters
        # How many blocks in each stage (Default: ResNet18-like)
        layers: list = [2, 2, 2, 2],
        norm_type: str = "BatchNorm2d",
    ):
        """
        Flexible ResNet Architecture.
        'layers' list controls depth.
        [2,2,2,2] is ~ResNet18. [3,4,6,3] is ~ResNet34.
        """
        super(ResNet, self).__init__()
        self.inplanes = hidden_dim
        self.norm_type = norm_type

        # -- Stem (Initial Entry) --
        # 7x7 Conv to reduce image size quickly
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = getattr(nn, norm_type)(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # -- Residual Stages --
        # Each "layer" here is actually a sequence of Residual Blocks
        self.layer1 = self._make_layer(hidden_dim, layers[0])
        self.layer2 = self._make_layer(hidden_dim * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(hidden_dim * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(hidden_dim * 8, layers[3], stride=2)

        # -- Classifier Head --
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_dim * 8, out_dim)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        """
        Builds a stage containing multiple ResidualBlocks.
        """
        downsample = None
        NormLayer = getattr(nn, self.norm_type)

        # Create a downsample layer if stride != 1 or channels change
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(planes),
            )

        layers = []
        # The first block handles the stride/downsampling
        layers.append(ResidualBlock(self.inplanes, planes, stride, downsample, self.norm_type))

        # Update current channel count
        self.inplanes = planes

        # The rest of the blocks just process features (stride=1)
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes, norm_type=self.norm_type))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
