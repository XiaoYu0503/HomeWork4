# Handwritten Character Recognition Roadmap

A staged project that starts with a simple MNIST fully connected classifier and progressively adds capabilities until the model can run in a browser or on mobile hardware.

## Implementation Ladder
1. **Baseline MNIST (Step 1)**: Run a fully connected classifier on the standard MNIST dataset, report accuracy, and save weights plus sample predictions.
2. **36-Class Extension (Step 2)**: Replace the final layer to output 36 classes (0-9, A-Z) and train on EMNIST or custom letter data.
3. **Interactive Canvas (Step 3)**: Provide a Gradio or Streamlit front end where users draw digits/letters that are sent to the model for instant predictions.
4. **Data Augmentation (Step 4)**: Add rotation, scaling, Gaussian noise, and morphology transforms to improve robustness.
5. **Confidence Threshold (Step 5)**: Surface the maximum softmax probability and warn users when confidence drops below a configurable cutoff.
6. **Feedback Loop (Step 6)**: Allow users to relabel mistakes, collecting images plus ground truth for future fine-tuning.
7. **Model Comparison (Step 7)**: Implement a simple CNN alongside the FC baseline and compare accuracy, runtime, and robustness.
8. **Deployment (Step 8)**: Export the trained model to ONNX or TensorFlow.js so inference can run directly in the browser or on mobile.

## Step 1 Plan: Fully Connected MNIST Baseline
- **Tech stack**: Python 3.10+, PyTorch, Torchvision, TorchMetrics, TensorBoard, rich CLI logging.
- **Repository layout**:
  - `src/data/`: dataset + transforms utilities.
  - `src/models/`: FC network definitions (`fc.py`, later `cnn.py`).
  - `src/training/`: training loop, evaluation, checkpointing.
  - `src/inference/`: simple CLI or notebook for running saved weights.
  - `configs/`: YAML/JSON hyperparameter files for reproducible runs.
  - `experiments/`: metrics, TensorBoard logs, and sample outputs.
- **Baseline workflow**:
  1. Download MNIST through `torchvision.datasets.MNIST` in `src/data/mnist.py` with standard normalization.
  2. Define a 3-layer fully connected network with dropout and ReLU activations (`src/models/fc.py`).
  3. Implement a training script (`train.py`) that loads config, trains for N epochs, logs accuracy/loss, and saves the best checkpoint under `artifacts/`.
  4. Provide an `infer.py` script that loads the checkpoint and runs predictions on example digits or new images.
  5. Record baseline metrics in `reports/step1_baseline.md` for future comparison.

## Step 1 Status
- Training command: `python train.py --config configs/step1_baseline.yaml`
- Best validation accuracy: **98.24%**, test accuracy: **98.49%**; see `reports/step1_baseline.md` for the full log.
- Best checkpoint stored at `checkpoints/step1_mnist_fc_best.pth` (copied from the latest `artifacts/` run).
- Run sample inference via `python infer.py --checkpoint checkpoints/step1_mnist_fc_best.pth [--image PATH | --index 0]` with `configs/step1_baseline.yaml`.

## Step 2 Status (EMNIST 36-class)
- Training command: `python train.py --config configs/step2_emnist_fc.yaml`
- Uses EMNIST Balanced with automatic label remapping to `0-9` + `A-Z` (36 classes) via `src/data/emnist36.py`.
- Achieved **86.96%** validation accuracy and **86.99%** test accuracy; details in `reports/step2_emnist_fc.md`.
- Checkpoint stored at `checkpoints/step2_emnist36_fc_best.pth`.
- Inference: `python infer.py --checkpoint checkpoints/step2_emnist36_fc_best.pth --config configs/step2_emnist_fc.yaml [--image PATH | --index 0]` (custom images are auto-rotated to EMNIST orientation).

## Step 3 Status (Interactive Canvas)
- Install dependencies (`pip install -r requirements.txt`) which now include Gradio.
  - Streamlit Cloud 目前使用 Python 3.13，因此本專案將 PyTorch 鎖定在 **2.5.1**（搭配 Torchvision **0.20.1**）以確保雲端能取得 CPU 版 wheel；本地端建議同樣採用此版本組合，可在 Python 3.10+ 正常安裝。
- Launch the UI: `python app_canvas.py --checkpoint checkpoints/step2_emnist36_fc_best.pth --config configs/step2_emnist_fc.yaml`.
- Features: sketchpad input, top-k label confidences, orientation fix for EMNIST canvases, CPU/GPU toggle, configurable host/port/share flags.
- Use the Step 1 checkpoint/config if you want a digit-only demo; the interface automatically adapts based on the provided config metadata.
- Streamlit option: `streamlit run streamlit_app.py --server.port 8501 -- --checkpoint checkpoints/step2_emnist36_fc_best.pth --config configs/step2_emnist_fc.yaml` or deploy directly on [streamlit.io](https://streamlit.io/cloud) following the steps below.

### Streamlit Cloud Deployment Checklist
1. Commit/push `streamlit_app.py`, `configs/`, and the curated checkpoint under `checkpoints/` into the repository (or add download logic referencing cloud storage).
2. On streamlit.io, point the app to `streamlit_app.py` and set environment variables if you prefer to load paths dynamically (e.g., `CHECKPOINT_PATH`).
3. Ensure `requirements.txt` includes `streamlit` and `streamlit-drawable-canvas` (already provided) so the cloud build installs the UI dependencies.
4. After deployment, adjust sidebar settings (config path, checkpoint path, brush size, auto-update toggle) directly from the hosted UI.

## Immediate Next Actions
1. Implement configurable data augmentation (rotation/scale/noise/morphology) and ablation logging (Step 4).
2. Surface configurable confidence thresholds plus user messaging (Step 5).
3. Design data-feedback storage (`user_samples/`) and fine-tuning hooks (Step 6).
