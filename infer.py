from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import yaml
from rich.console import Console
from torchvision import datasets

from src.data import emnist36, mnist
from src.inference.predict import load_model, predict_image, preprocess_image

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for MNIST baseline")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/step1_baseline.yaml", help="Model config for architecture params")
    parser.add_argument("--image", type=str, help="Optional path to an image file to classify")
    parser.add_argument("--index", type=int, default=0, help="Test dataset index to visualize if --image is not provided")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_dataset(cfg: Dict[str, Any]):
    dataset_cfg = cfg["dataset"]
    dataset_type = dataset_cfg.get("type", "mnist").lower()
    mean = dataset_cfg["normalization"]["mean"]
    std = dataset_cfg["normalization"]["std"]
    root = Path(dataset_cfg["data_root"])

    if dataset_type == "mnist":
        transform = mnist.build_transforms(mean, std)
        ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    elif dataset_type == "emnist36":
        transform = emnist36.build_transforms(mean, std)
        ds = datasets.EMNIST(
            root=root,
            split=dataset_cfg.get("split", "balanced"),
            train=False,
            download=True,
            transform=transform,
        )
        emnist36.remap_to_alphanumeric(ds)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return ds


def format_label(index: int, labels: Optional[Sequence[str]]) -> str:
    if labels is None:
        return str(index)
    if index < 0 or index >= len(labels):
        return str(index)
    return labels[index]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model_cfg = cfg["model"]
    model = load_model(
        checkpoint=Path(args.checkpoint),
        input_dim=model_cfg["input_dim"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.0),
        device=device,
    )

    label_names: Optional[Sequence[str]] = cfg.get("labels")
    dataset_cfg = cfg["dataset"]
    dataset_type = dataset_cfg.get("type", "mnist").lower()

    if args.image:
        tensor = preprocess_image(
            Path(args.image),
            dataset_cfg["normalization"]["mean"],
            dataset_cfg["normalization"]["std"],
        )
        if dataset_type == "emnist36":
            tensor = emnist36.adjust_tensor_orientation(tensor)
        label = None
    else:
        test_ds = _build_dataset(cfg)
        tensor, label = test_ds[args.index]
        tensor = tensor.unsqueeze(0)

    pred, confidence = predict_image(model, tensor, device=device)
    prediction_label = format_label(pred, label_names)
    message = f"Prediction: {prediction_label} (index {pred}) | Confidence: {confidence:.2%}"
    if label is not None:
        message += f" | Ground Truth: {format_label(int(label), label_names)}"
    console.print(message)


if __name__ == "__main__":
    main()
