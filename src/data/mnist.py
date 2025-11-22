"""MNIST data utilities for training and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    download: bool
    val_split: float
    mean: float
    std: float
    batch_size: int
    num_workers: int
    seed: int


@dataclass
class MNISTDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_transforms(mean: float, std: float) -> transforms.Compose:
    """Return the canonical MNIST transform pipeline."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def _split_train_val(dataset: torch.utils.data.Dataset, val_split: float, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be within (0, 1)")
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def create_dataloaders(cfg: DataConfig) -> MNISTDataLoaders:
    """Create train/val/test dataloaders for MNIST."""
    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    transform = build_transforms(cfg.mean, cfg.std)

    full_train = datasets.MNIST(
        root=data_root,
        train=True,
        download=cfg.download,
        transform=transform,
    )
    train_dataset, val_dataset = _split_train_val(full_train, cfg.val_split, cfg.seed)

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=cfg.download,
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }

    return MNISTDataLoaders(
        train=DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        val=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )
