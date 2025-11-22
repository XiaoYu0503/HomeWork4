from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.data import emnist36
from src.inference.predict import load_model

CheckpointSource = Union[str, Path]


def format_label(index: int, labels: Optional[Sequence[str]]) -> str:
    if labels is None or index < 0 or index >= len(labels):
        return str(index)
    return labels[index]


def _to_uint8_image(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return Image.fromarray(array.astype("uint8"))


class DrawingPredictor:
    def __init__(self, checkpoint: CheckpointSource, cfg: Dict[str, Any], device: torch.device, top_k: int = 5) -> None:
        model_cfg = cfg["model"]
        self.device = device
        self.model = load_model(
            checkpoint=Path(checkpoint),
            input_dim=model_cfg["input_dim"],
            hidden_dims=tuple(model_cfg["hidden_dims"]),
            num_classes=model_cfg["num_classes"],
            dropout=model_cfg.get("dropout", 0.0),
            device=device,
        )

        dataset_cfg = cfg["dataset"]
        self.dataset_type = dataset_cfg.get("type", "mnist").lower()
        norm = dataset_cfg.get("normalization", {"mean": 0.1307, "std": 0.3081})
        self.mean = norm["mean"]
        self.std = norm["std"]
        self.image_size = int(dataset_cfg.get("image_size", 28))
        self.labels: Optional[Sequence[str]] = cfg.get("labels")
        num_classes = model_cfg["num_classes"]
        self.top_k = max(1, min(top_k, num_classes))
        self.tensor_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((self.mean,), (self.std,)),
            ]
        )

    def _preprocess_array(self, drawing: np.ndarray) -> torch.Tensor:
        if drawing is None:
            raise ValueError("畫板尚未輸入任何內容")
        image = _to_uint8_image(drawing)
        image = image.convert("L")
        image = ImageOps.invert(image)
        image = image.resize((self.image_size, self.image_size))
        tensor = self.tensor_transform(image)
        if self.dataset_type == "emnist36":
            tensor = emnist36.adjust_tensor_orientation(tensor)
        return tensor.unsqueeze(0)

    def predict_topk(self, tensor: torch.Tensor) -> List[Tuple[str, float]]:
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            top_k = min(self.top_k, probs.shape[0])
            confidences, indices = torch.topk(probs, top_k)
        return [
            (format_label(idx.item(), self.labels), float(conf.item()))
            for idx, conf in zip(indices, confidences)
        ]

    def predict_from_array(self, drawing: np.ndarray) -> List[Tuple[str, float]]:
        tensor = self._preprocess_array(drawing)
        return self.predict_topk(tensor)
