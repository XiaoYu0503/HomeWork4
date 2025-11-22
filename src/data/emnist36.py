"""EMNIST pipeline remapped to 36 alphanumeric classes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from .mnist import MNISTDataLoaders

CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_NAMES)}


def _fix_emnist_orientation(tensor: torch.Tensor) -> torch.Tensor:
    # EMNIST samples are transposed; rotate 90 degrees and mirror to match MNIST orientation.
    spatial_dims = (-2, -1)
    tensor = torch.rot90(tensor, k=1, dims=spatial_dims)
    return torch.flip(tensor, dims=(spatial_dims[1],))


def adjust_tensor_orientation(tensor: torch.Tensor) -> torch.Tensor:
    """Expose the orientation correction for inference-time preprocessing."""
    if tensor.ndim == 3:
        return _fix_emnist_orientation(tensor)
    if tensor.ndim == 4:
        return _fix_emnist_orientation(tensor)
    raise ValueError("Expected a 3D (C,H,W) or 4D (N,C,H,W) tensor for orientation adjustment")


@dataclass(frozen=True)
class EMNISTConfig:
    data_root: str
    download: bool
    val_split: float
    mean: float
    std: float
    batch_size: int
    num_workers: int
    seed: int
    split: str = "balanced"


def build_transforms(mean: float, std: float) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(_fix_emnist_orientation),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def _split_train_val(dataset: Dataset, val_split: float, seed: int) -> Tuple[Dataset, Dataset]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be within (0, 1)")
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)


def _remap_dataset(dataset: datasets.EMNIST) -> None:
    original_classes = dataset.classes
    remap_table = torch.full((len(original_classes),), -1, dtype=torch.long)
    for idx, label in enumerate(original_classes):
        target_label = label.upper()
        if target_label not in CLASS_TO_INDEX:
            raise ValueError(f"Label {label} cannot be mapped to 36-class set")
        remap_table[idx] = CLASS_TO_INDEX[target_label]
    targets = dataset.targets
    if isinstance(targets, list):
        targets = torch.tensor(targets, dtype=torch.long)
    dataset.targets = remap_table[targets]
    dataset.classes = CLASS_NAMES
    dataset._class_to_idx = CLASS_TO_INDEX.copy()


def remap_to_alphanumeric(dataset: datasets.EMNIST) -> None:
    """Public helper so inference can mirror the training remap."""
    _remap_dataset(dataset)


def create_dataloaders(cfg: EMNISTConfig) -> MNISTDataLoaders:
    data_root = Path(cfg.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    transform = build_transforms(cfg.mean, cfg.std)

    train_dataset = datasets.EMNIST(
        root=data_root,
        split=cfg.split,
        train=True,
        download=cfg.download,
        transform=transform,
    )
    _remap_dataset(train_dataset)
    train_subset, val_subset = _split_train_val(train_dataset, cfg.val_split, cfg.seed)

    test_dataset = datasets.EMNIST(
        root=data_root,
        split=cfg.split,
        train=False,
        download=cfg.download,
        transform=transform,
    )
    _remap_dataset(test_dataset)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }

    return MNISTDataLoaders(
        train=DataLoader(train_subset, shuffle=True, **loader_kwargs),
        val=DataLoader(val_subset, shuffle=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )
