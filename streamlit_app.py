from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

import numpy as np
import streamlit as st
import torch
import yaml
from streamlit_drawable_canvas import st_canvas

from src.inference.ui import DrawingPredictor

DEFAULT_CONFIG = "configs/step2_emnist_fc.yaml"
DEFAULT_CHECKPOINT = "checkpoints/step2_emnist36_fc_best.pth"
CANVAS_SIZE = 280


def load_config(path_str: str) -> Dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource(show_spinner=False)
def load_predictor(checkpoint_path: str, config_path: str, top_k: int) -> DrawingPredictor:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DrawingPredictor(checkpoint=Path(checkpoint_path), cfg=cfg, device=device, top_k=top_k)


def find_existing_checkpoints() -> List[str]:
    candidates: List[str] = []
    for directory in (Path("checkpoints"), Path("artifacts")):
        if directory.exists():
            candidates.extend(str(p) for p in directory.glob("**/*.pth"))
    return sorted(set(candidates))


def list_config_files() -> List[str]:
    config_dir = Path("configs")
    if not config_dir.exists():
        return []
    return sorted(str(p) for p in config_dir.glob("*.yaml"))


def predict_from_canvas(predictor: DrawingPredictor, image_data: np.ndarray) -> List[Tuple[str, float]]:
    return predictor.predict_from_array(image_data)


def main() -> None:
    st.set_page_config(page_title="手寫字元辨識", layout="wide")
    st.title("手寫辨識互動畫板 · Streamlit")
    st.caption("使用已訓練的 MNIST/EMNIST 模型，在瀏覽器中即時預測。")

    checkpoints = find_existing_checkpoints()
    env_default = os.getenv("CHECKPOINT_PATH", "").strip()
    default_ckpt = env_default or (DEFAULT_CHECKPOINT if Path(DEFAULT_CHECKPOINT).exists() else "")
    if not default_ckpt and checkpoints:
        default_ckpt = checkpoints[-1]

    with st.sidebar:
        st.header("設定")
        config_options = sorted(set(list_config_files() + [DEFAULT_CONFIG])) or [DEFAULT_CONFIG]
        config_index = config_options.index(DEFAULT_CONFIG) if DEFAULT_CONFIG in config_options else 0
        config_path = st.selectbox("Config 檔案", options=config_options, index=config_index)
        if checkpoints:
            ckpt_mode = st.radio(
                "Checkpoint 選擇",
                options=("自動偵測", "手動輸入"),
                index=0 if not env_default else 1,
            )
        else:
            ckpt_mode = "手動輸入"
        if ckpt_mode == "自動偵測" and checkpoints:
            default_index = checkpoints.index(default_ckpt) if default_ckpt in checkpoints else len(checkpoints) - 1
            checkpoint_path = st.selectbox("可用權重", options=checkpoints, index=default_index)
        else:
            checkpoint_path = st.text_input(
                "Checkpoint 路徑",
                value=default_ckpt,
                help="請提供 .pth 權重檔相對或絕對路徑，或在部署時透過 CHECKPOINT_PATH 環境變數指定",
            )
        top_k = st.slider("顯示前幾名預測", min_value=1, max_value=10, value=5)
        stroke_width = st.slider("筆刷粗細", min_value=8, max_value=40, value=14)

    if not checkpoint_path:
        st.warning(
            "找不到 checkpoint，請在側邊欄輸入有效路徑，或確認 `checkpoints/` 目錄已隨程式一併部署。"
        )
        return

    try:
        predictor = load_predictor(checkpoint_path, config_path, top_k)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except RuntimeError as exc:
        st.error(f"載入模型失敗：{exc}")
        return

    if "canvas_key" not in st.session_state:
        st.session_state["canvas_key"] = "canvas"

    col_canvas, col_output = st.columns([1, 1])

    with col_canvas:
        st.subheader("繪製區")
        canvas_result = st_canvas(
            fill_color="#00000000",
            stroke_width=stroke_width,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key=st.session_state["canvas_key"],
        )
        cols = st.columns(2)
        predict_clicked = cols[0].button("預測一次", use_container_width=True)
        if cols[1].button("清除畫板", use_container_width=True):
            st.session_state["canvas_key"] = f"canvas-{uuid4()}"
            st.rerun()

    with col_output:
        st.subheader("模型信心")
        has_image = canvas_result.image_data is not None
        should_predict = predict_clicked and has_image

        if should_predict:
            try:
                predictions = predict_from_canvas(predictor, canvas_result.image_data)
            except ValueError:
                st.info("請先在左側畫板上書寫內容。")
                predictions = []
        else:
            predictions = []

        if predictions:
            rows = [{"Label": label, "Confidence": f"{conf * 100:.2f}%"} for label, conf in predictions]
            st.table(rows)
        else:
            st.info("沒有可顯示的預測結果。")

    st.markdown("---")
    st.markdown(
        "**部署提示：** 將 `.pth` 權重檔與 `configs/` 內容一併推送到 Streamlit Cloud，或在啟動前設定 `CHECKPOINT_PATH` 環境變數並填入同名路徑。"
    )


if __name__ == "__main__":
    main()
