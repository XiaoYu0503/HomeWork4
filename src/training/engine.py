"""Training loop for MNIST baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm

from .utils import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    model.train()
    metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        metric.update(logits.softmax(dim=-1), labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = metric.compute().item()
    return {"loss": epoch_loss, "acc": epoch_acc}


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            running_loss += loss.item() * labels.size(0)
            metric.update(logits.softmax(dim=-1), labels)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = metric.compute().item()
    return {"loss": epoch_loss, "acc": epoch_acc}


def _build_scheduler(optimizer: Optimizer, scheduler_cfg: Optional[Dict[str, float]], epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if not scheduler_cfg:
        return None
    sched_type = scheduler_cfg.get("type", "").lower()
    if sched_type == "cosine":
        min_lr = scheduler_cfg.get("min_lr", 1e-5)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    return None


def run_training(
    *,
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    num_classes: int,
    ckpt_dir: Path,
    tb_dir: Path,
    scheduler_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    scheduler = _build_scheduler(optimizer, scheduler_cfg, epochs)
    writer = SummaryWriter(log_dir=str(tb_dir))
    best_val_acc = 0.0
    best_ckpt: Optional[Path] = None

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, loaders["train"], optimizer, loss_fn, device, num_classes)
        val_stats = evaluate(model, loaders["val"], loss_fn, device, num_classes)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("Loss/train", train_stats["loss"], epoch)
        writer.add_scalar("Loss/val", val_stats["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_stats["acc"], epoch)
        writer.add_scalar("Accuracy/val", val_stats["acc"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_ckpt = ckpt_dir / f"best-epoch{epoch:02d}.pth"
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                },
                best_ckpt,
            )

    test_stats = evaluate(model, loaders["test"], loss_fn, device, num_classes)
    writer.add_hparams(
        {"epochs": epochs},
        {
            "hparam/best_val_acc": best_val_acc,
            "hparam/test_acc": test_stats["acc"],
        },
    )
    writer.close()

    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_stats["acc"],
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "test_loss": test_stats["loss"],
    }
