"""Model architectures for the lab: SimpleCNN and SmallResNet for CIFAR-10,
TabularMLP for the (later-phase) Adult Income dataset.

Opacus (DP-SGD) needs the per-sample gradient of every layer, which it gets
by hooking each module's forward/backward pass individually. BatchNorm
breaks this: its output for one example depends on every other example in
the batch, so there's no well-defined "per-sample" gradient for it.
GroupNorm normalizes each sample independently (splitting its channels into
groups), so it behaves identically whether or not DP-SGD is involved. We use
GroupNorm everywhere from the start, rather than training with BatchNorm and
running Opacus's ``ModuleValidator.fix()`` later — that autofix works, but
building DP-compatible models from day one means the *same* checkpoint code
path works for both standard and DP-SGD training in Phase 6, with nothing to
swap out.

For the same reason, **no activation here uses ``inplace=True``**. Opacus
attaches backward hooks whose outputs are views; mutating one in place makes
PyTorch refuse the backward pass outright ("a view ... is being modified
inplace"). The memory an inplace ReLU saves is irrelevant at this model size,
and tests/test_dp_training.py will fail loudly if one is reintroduced.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class SimpleCNN(nn.Module):
    """A small 4-conv-layer CNN for CIFAR-10 (32x32x3 -> 10 classes).

    Kept deliberately small (max 64 channels) so it trains in a couple of
    minutes on a free Colab GPU, or even on CPU for quick tests, and small
    enough that DP-SGD's per-sample gradient bookkeeping doesn't blow up
    memory.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.0, num_groups: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResidualBlock(nn.Module):
    """A basic pre-activation-free residual block using GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class SmallResNet(nn.Module):
    """A tiny ResNet (3 stages x 2 blocks, 16/32/64 channels) for CIFAR-10.

    This is a scaled-down ResNet-18-style architecture, not the real thing —
    real ResNet-18 is overkill for 32x32 images and for a Colab free-tier
    GPU. Channel counts are chosen to keep parameter count (and therefore
    Opacus per-sample gradient memory) small.
    """

    def __init__(self, num_classes: int = 10, num_groups: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 16),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16, stride=1, num_groups=num_groups),
            ResidualBlock(16, 16, stride=1, num_groups=num_groups),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2, num_groups=num_groups),
            ResidualBlock(32, 32, stride=1, num_groups=num_groups),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2, num_groups=num_groups),
            ResidualBlock(64, 64, stride=1, num_groups=num_groups),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TabularMLP(nn.Module):
    """A small MLP for tabular data (Adult Income, used from Phase 6 on).

    No normalization layers at all: plain Linear + ReLU + Dropout is already
    Opacus-compatible (each layer's output already depends only on its own
    input row), so there's nothing to swap out for DP-SGD here, unlike the
    CNNs above.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
    "small_resnet": SmallResNet,
    "tabular_mlp": TabularMLP,
}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Construct a model by its config-file name, e.g. ``build_model(name="simple_cnn", dropout=0.3)``.

    A registry dict (instead of ``eval()`` or a long if/elif chain) keeps
    config files decoupled from Python class names while staying trivial to
    extend — add one line here per new architecture.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Available: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name](**kwargs)
