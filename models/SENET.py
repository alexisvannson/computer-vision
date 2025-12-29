import torch
import torch.nn as nn
from torch import Tensor

from .ResNet import ResidualBlock, ResNet


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck in the excitation operation
        """
        super(SEBlock, self).__init__()

        # Squeeze: Global average pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two FC layers with ReLU and Sigmoid
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # Squeeze: [B, C, H, W] -> [B, C, 1, 1]
        squeeze = self.squeeze(x)

        # Flatten: [B, C, 1, 1] -> [B, C]
        squeeze = squeeze.view(batch, channels)

        # Excitation: [B, C] -> [B, C]
        excitation = self.excitation(squeeze)

        # Reshape: [B, C] -> [B, C, 1, 1]
        excitation = excitation.view(batch, channels, 1, 1)

        # Scale: Element-wise multiplication
        return x * excitation


class SEResidualBlock(ResidualBlock):
    """
    ResNet Residual Block with Squeeze-and-Excitation.
    Inherits from ResidualBlock and adds an SE module after the second convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        norm_type="BatchNorm2d",
        reduction=16,
    ):
        super(SEResidualBlock, self).__init__(
            in_channels, out_channels, stride, downsample, norm_type
        )

        # Add SE block after the second convolution
        self.se = SEBlock(out_channels, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE block
        out = self.se(out)

        # If the input shape/channels don't match the output,
        # apply the downsample layer to the identity so we can add them.
        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip Connection
        out += identity
        out = self.relu(out)

        return out


class SENet(ResNet):
    """
    SENet (Squeeze-and-Excitation Network).
    Inherits from ResNet and replaces standard residual blocks with SE-ResNet blocks.
    """

    def __init__(
        self,
        in_dim: int,  # Input channels (3 for RGB)
        out_dim: int,  # Number of classes
        hidden_dim: int = 64,  # Starting number of filters
        layers: list = [2, 2, 2, 2],  # How many blocks in each stage
        norm_type: str = "BatchNorm2d",
        reduction: int = 16,  # SE block reduction ratio
    ):
        """
        SENet Architecture.

        Args:
            in_dim: Number of input channels (3 for RGB)
            out_dim: Number of output classes
            hidden_dim: Starting number of filters
            layers: Number of blocks in each stage (e.g., [2,2,2,2] for SE-ResNet18)
            norm_type: Type of normalization layer
            reduction: Reduction ratio for SE blocks
        """
        self.reduction = reduction

        # Initialize the parent ResNet class
        # We'll override the layers after initialization
        super(SENet, self).__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            norm_type=norm_type,
        )

        # Rebuild layers with SE blocks
        self.inplanes = hidden_dim
        self.layer1 = self._make_se_layer(hidden_dim, layers[0])
        self.layer2 = self._make_se_layer(hidden_dim * 2, layers[1], stride=2)
        self.layer3 = self._make_se_layer(hidden_dim * 4, layers[2], stride=2)
        self.layer4 = self._make_se_layer(hidden_dim * 8, layers[3], stride=2)

    def _make_se_layer(self, planes, blocks, stride=1):
        """
        Builds a stage containing multiple SEResidualBlocks.
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
        layers.append(
            SEResidualBlock(
                self.inplanes, planes, stride, downsample, self.norm_type, self.reduction
            )
        )

        # Update current channel count
        self.inplanes = planes

        # The rest of the blocks just process features (stride=1)
        for _ in range(1, blocks):
            layers.append(
                SEResidualBlock(
                    self.inplanes, planes, norm_type=self.norm_type, reduction=self.reduction
                )
            )

        return nn.Sequential(*layers)


if __name__ == "__main__":
    # Create SE-ResNet18 (similar architecture to ResNet18 but with SE blocks)
    model = SENet(
        in_dim=3,
        out_dim=10,
        hidden_dim=64,
        layers=[2, 2, 2, 2],
        reduction=16,  # SE block reduction ratio
    )
