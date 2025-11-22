# Step 2 EMNIST-36 Report

- Best validation accuracy: 86.96%
- Test accuracy: 86.99%
- Test loss: 0.3626
- Checkpoint: artifacts\step2\step2_emnist36_fc\20251122-104250\best-epoch12.pth

## Config Snapshot
```json
{
  "experiment_name": "step2_emnist36_fc",
  "seed": 2025,
  "device": "auto",
  "output_dir": "artifacts/step2",
  "log_dir": "experiments/step2",
  "report_title": "Step 2 EMNIST-36 Report",
  "training": {
    "batch_size": 256,
    "num_workers": 4,
    "epochs": 12,
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
    "type": "emnist36",
    "data_root": "data",
    "split": "balanced",
    "download": true,
    "val_split": 0.1,
    "normalization": {
      "mean": 0.1736,
      "std": 0.3317
    }
  },
  "model": {
    "input_dim": 784,
    "hidden_dims": [
      512,
      256
    ],
    "dropout": 0.3,
    "num_classes": 36
  },
  "labels": [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z"
  ]
}
```