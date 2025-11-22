"""Inference helpers for drawing predictions from a trained model."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

from src.models.fc import FullyConnectedNet


def load_model(checkpoint: Path, input_dim: int, hidden_dims: Tuple[int, ...], num_classes: int, dropout: float = 0.0, device: torch.device | None = None) -> FullyConnectedNet:
    device = device or torch.device("cpu")
    model = FullyConnectedNet(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes, dropout=dropout)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path, mean: float, std: float) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def predict_image(model: FullyConnectedNet, tensor: torch.Tensor, device: torch.device | None = None) -> Tuple[int, float]:
    device = device or torch.device("cpu")
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
    return pred.item(), conf.item()
