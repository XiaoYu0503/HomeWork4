from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
import yaml
from rich.console import Console
from rich.table import Table

from src.data import emnist36, mnist
from src.models.fc import FullyConnectedNet
from src.training.engine import run_training
from src.training.utils import count_trainable_params, prepare_output_dirs, resolve_device, set_seed

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST fully connected baseline")
    parser.add_argument("--config", type=str, default="configs/step1_baseline.yaml", help="Path to YAML config file")
    parser.add_argument("--report", type=str, default="reports/step1_baseline.md", help="Report file to update after training")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    dataset_type = data_cfg.get("type", "mnist").lower()
    common_kwargs = {
        "data_root": data_cfg["data_root"],
        "download": data_cfg.get("download", True),
        "val_split": data_cfg.get("val_split", 0.1),
        "mean": data_cfg["normalization"]["mean"],
        "std": data_cfg["normalization"]["std"],
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg.get("num_workers", 0),
        "seed": cfg.get("seed", 42),
    }

    if dataset_type == "mnist":
        config_obj = mnist.DataConfig(**common_kwargs)
        loaders = mnist.create_dataloaders(config_obj)
    elif dataset_type == "emnist36":
        config_obj = emnist36.EMNISTConfig(split=data_cfg.get("split", "balanced"), **common_kwargs)
        loaders = emnist36.create_dataloaders(config_obj)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return loaders, dataset_type


def update_report(report_path: Path, metrics: Dict[str, Any], title: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- Best validation accuracy: {metrics['best_val_acc'] * 100:.2f}%",
        f"- Test accuracy: {metrics['test_acc'] * 100:.2f}%",
        f"- Test loss: {metrics['test_loss']:.4f}",
        f"- Checkpoint: {metrics.get('best_checkpoint', 'n/a')}",
        "",
        "## Config Snapshot",
        "```json",
        json.dumps(metrics["config_snapshot"], indent=2),
        "```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = resolve_device(cfg.get("device", "auto"))

    dirs = prepare_output_dirs(cfg["output_dir"], cfg["log_dir"], cfg["experiment_name"])

    loaders, dataset_type = build_dataloaders(cfg)

    model_cfg = cfg["model"]
    labels = cfg.get("labels")
    if labels and len(labels) != model_cfg["num_classes"]:
        raise ValueError("Number of labels must match model.num_classes")
    model = FullyConnectedNet(
        input_dim=model_cfg["input_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.0),
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["optimizer"]["lr"],
        weight_decay=cfg["training"]["optimizer"].get("weight_decay", 0.0),
    )
    loss_fn = nn.CrossEntropyLoss()

    table = Table(title="Training Configuration")
    table.add_column("Key", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Device", str(device))
    table.add_row("Parameters", f"{count_trainable_params(model):,}")
    table.add_row("Epochs", str(cfg["training"]["epochs"]))
    table.add_row("Batch Size", str(cfg["training"]["batch_size"]))
    table.add_row("Dataset", dataset_type)
    table.add_row("Checkpoint Dir", str(dirs["ckpt_dir"]))
    table.add_row("TensorBoard Dir", str(dirs["tb_dir"]))
    console.print(table)

    metrics = run_training(
        model=model,
        loaders={"train": loaders.train, "val": loaders.val, "test": loaders.test},
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=cfg["training"]["epochs"],
        num_classes=model_cfg["num_classes"],
        ckpt_dir=dirs["ckpt_dir"],
        tb_dir=dirs["tb_dir"],
        scheduler_cfg=cfg["training"].get("scheduler"),
    )

    metrics["config_snapshot"] = cfg
    report_title = cfg.get("report_title", "Training Report")
    update_report(Path(args.report), metrics, report_title)
    console.print(f"Training complete. Best val acc: {metrics['best_val_acc']:.4f}, test acc: {metrics['test_acc']:.4f}")


if __name__ == "__main__":
    main()