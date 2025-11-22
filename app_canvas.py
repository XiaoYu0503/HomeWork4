from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import gradio as gr
import numpy as np
import torch
import yaml

from src.inference.ui import DrawingPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive canvas for handwritten character recognition")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to a trained checkpoint (.pth)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/step2_emnist_fc.yaml",
        help="Model config describing architecture and dataset metadata",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to display")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a Gradio shareable link (if supported)")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _predict_dict(predictor: DrawingPredictor, drawing: np.ndarray) -> Dict[str, float]:
    try:
        pairs = predictor.predict_from_array(drawing)
    except ValueError:
        return {"請先在畫板輸入內容": 0.0}
    return {label: round(conf, 4) for label, conf in pairs}


def build_interface(predictor: DrawingPredictor) -> gr.Blocks:
    with gr.Blocks(title="Step 3 - Interactive Canvas") as demo:
        gr.Markdown(
            """
            ## 手寫字元即時預測
            - 左側畫板支援滑鼠或手寫筆輸入，按下「預測」後即可看到最高機率的字元。
            - 右側會顯示模型信心分佈，可多次嘗試或清除畫布重畫。
            """
        )
        with gr.Row():
            with gr.Column():
                sketchpad = gr.Sketchpad(label="畫板", brush_radius=12, shape=(280, 280))
                with gr.Row():
                    predict_btn = gr.Button("預測", variant="primary")
                    clear_btn = gr.Button("清除", variant="secondary")
            with gr.Column():
                label = gr.Label(num_top_classes=predictor.top_k, label="模型信心")
                gr.Markdown("請在左側畫板輸入數字或字母，再按下預測。")
        predict_btn.click(fn=lambda img: _predict_dict(predictor, img), inputs=sketchpad, outputs=label)
        clear_btn.click(fn=lambda: (None, {}), inputs=None, outputs=[sketchpad, label])
    return demo


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint_path = Path(args.checkpoint)
    predictor = DrawingPredictor(checkpoint=checkpoint_path, cfg=cfg, device=device, top_k=args.top_k)
    demo = build_interface(predictor)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
