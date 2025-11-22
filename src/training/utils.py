"""Common helpers for training scripts."""
from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def prepare_output_dirs(output_dir: str, log_dir: str, experiment_name: str) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = Path(output_dir) / experiment_name / timestamp
    tb_dir = Path(log_dir) / experiment_name / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return {"ckpt_dir": ckpt_dir, "tb_dir": tb_dir}


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: Dict[str, Any], ckpt_path: Path) -> None:
    torch.save(state, ckpt_path)


def latest_checkpoint(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    checkpoints = sorted(folder.glob("*.pth"))
    return checkpoints[-1] if checkpoints else None
