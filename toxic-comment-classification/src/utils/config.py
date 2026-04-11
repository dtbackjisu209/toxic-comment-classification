"""Training configuration and shared helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class TrainingConfig:
    model_name: str = "vinai/phobert-base"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    train_val_split: float = 0.2
    seed: int = 42
    num_labels: int = 2
    text_col: str = "text"
    label_col: str = "label"
    train_path: str = "/content/drive/MyDrive/ViCTSD/ViCTSD_train.csv"
    val_path: str = "/content/drive/MyDrive/ViCTSD/ViCTSD_valid.csv"
    test_path: str = "/content/drive/MyDrive/ViCTSD/ViCTSD_test.csv"
    output_dir: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/toxic_outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/toxic_outputs/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("/content/drive/MyDrive/toxic_outputs/logs"))

    @property
    def best_model_dir(self) -> Path:
        return self.checkpoint_dir / "best_model"

    @property
    def metrics_path(self) -> Path:
        return self.log_dir / "metrics.json"


def build_output_paths(config: TrainingConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.best_model_dir.mkdir(parents=True, exist_ok=True)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
