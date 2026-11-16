"""Model definitions for the MNIST experiments."""

from __future__ import annotations

import torch
from torch import nn


class MNISTMLP(nn.Module):
    """Fully connected classifier that returns logits for CrossEntropyLoss."""

    def __init__(self, hidden_sizes: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        input_size = 28 * 28
        for width in hidden_sizes:
            layers.extend((nn.Linear(input_size, width), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = width
        layers.append(nn.Linear(input_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class MNISTCNN(nn.Module):
    """Three convolution/pooling stages followed by a fully connected classifier."""

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        classifier: list[nn.Module] = [nn.Flatten(), nn.Linear(128 * 3 * 3, 128), nn.ReLU()]
        if dropout:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(128, 10))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.reshape(-1, 1, 28, 28)
        return self.classifier(self.features(images))


class MNISTDeepCNN(nn.Module):
    """Four convolution layers with spatial downsampling and a compact head."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        classifier: list[nn.Module] = [nn.Flatten()]
        if dropout:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(64, 10))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.reshape(-1, 1, 28, 28)
        return self.classifier(self.features(images))


class ResidualBlock(nn.Module):
    """Two-convolution residual block with a projection when dimensions differ."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


class MNISTResNet(nn.Module):
    """Residual CNN with six blocks and three feature-width stages."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResidualBlock(16, 16), ResidualBlock(16, 16))
        self.stage2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2), ResidualBlock(32, 32)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.reshape(-1, 1, 28, 28)
        outputs = self.stem(images)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        return self.classifier(self.pool(outputs).flatten(1))
