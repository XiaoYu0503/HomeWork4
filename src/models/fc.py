"""Fully connected baseline network for MNIST."""
from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int], num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        dims: List[int] = [input_dim, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.net(x)
