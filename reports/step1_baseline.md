# Step 1 Baseline Report

- Best validation accuracy: 98.24%
- Test accuracy: 98.49%
- Test loss: 0.0490
- Checkpoint: artifacts\step1\step1_mnist_fc\20251122-103206\best-epoch08.pth

## Config Snapshot
```json
{
  "experiment_name": "step1_mnist_fc",
  "seed": 1337,
  "device": "auto",
  "output_dir": "artifacts/step1",
  "log_dir": "experiments/step1",
  "training": {
    "batch_size": 128,
    "num_workers": 4,
    "epochs": 10,
    "optimizer": {
      "lr": 0.001,
      "weight_decay": 0.0005
    },
    "scheduler": {
      "type": "cosine",
      "min_lr": 1e-05
    }
  },
  "dataset": {
    "data_root": "data",
    "download": true,
    "val_split": 0.1,
    "normalization": {
      "mean": 0.1307,
      "std": 0.3081
    }
  },
  "model": {
    "input_dim": 784,
    "hidden_dims": [
      512,
      256
    ],
    "dropout": 0.2,
    "num_classes": 10
  }
}
```