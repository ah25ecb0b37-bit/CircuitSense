"""
CircuitSense — PyTorch Dataset & DataLoader
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from .augmentation import SignalAugmentor

class CircuitWaveformDataset(Dataset):
    def __init__(self, waveforms: np.ndarray, labels: np.ndarray, augment=False, cfg=None):
        self.X = torch.tensor(waveforms, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
        self.Y = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.augmentor = SignalAugmentor(cfg) if augment else None

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment and self.augmentor:
            x = self.augmentor(x)
        return x, self.Y[idx]


def get_dataloaders(data_path: str, cfg: dict):
    df = pd.read_csv(data_path)
    seq_cols = [c for c in df.columns if c.startswith("t")]
    X = df[seq_cols].values.astype(np.float32)
    Y = df["label"].values.astype(np.int64)

    # Normalize per-sample (zero mean, unit variance)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    train_ds = CircuitWaveformDataset(X_train, y_train, augment=cfg.get("augment", True), cfg=cfg)
    val_ds   = CircuitWaveformDataset(X_val,   y_val,   augment=False)
    test_ds  = CircuitWaveformDataset(X_test,  y_test,  augment=False)

    bs = cfg.get("batch_size", 64)
    nw = cfg.get("num_workers", 0)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw)
    test_dl  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw)

    return train_dl, val_dl, test_dl
