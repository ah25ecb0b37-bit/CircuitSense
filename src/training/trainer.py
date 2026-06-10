"""
CircuitSense — Training Loop with W&B logging
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import yaml, time
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score
import numpy as np

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

from ..models import CircuitSenseModel
from ..data import get_dataloaders
from .losses import FocalLoss

console = Console()

def get_device(cfg_device="auto"):
    if cfg_device == "auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)

def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train(train)
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            loss = loss_fn(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / total
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1

def train(config_path="configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device(cfg["inference"]["device"])
    console.print(f"[bold green]CircuitSense Training[/bold green] — device: {device}")

    # Data
    train_dl, val_dl, test_dl = get_dataloaders(
        f"{cfg['data']['raw_dir']}/waveforms.csv", {**cfg["data"], **cfg["training"]}
    )

    # Model
    model = CircuitSenseModel(cfg["model"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Parameters: [cyan]{total_params:,}[/cyan]")

    # Loss & Optimizer
    loss_fn = FocalLoss(gamma=cfg["training"]["focal_gamma"]) if cfg["training"]["loss"] == "focal" else nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"], eta_min=cfg["training"]["min_lr"])

    # W&B
    if WANDB and cfg["wandb"]["entity"]:
        wandb.init(project=cfg["wandb"]["project"], config=cfg)

    Path("checkpoints").mkdir(exist_ok=True)
    best_val_f1, patience_count = 0, 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, optimizer, loss_fn, device, train=True)
        vl_loss, vl_acc, vl_f1 = run_epoch(model, val_dl,   optimizer, loss_fn, device, train=False)
        scheduler.step()

        if WANDB and cfg["wandb"]["entity"]:
            wandb.log({"train/loss": tr_loss, "train/acc": tr_acc, "train/f1": tr_f1,
                       "val/loss": vl_loss, "val/acc": vl_acc, "val/f1": vl_f1, "epoch": epoch})

        console.print(f"Ep {epoch:03d} | tr_loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f} | "
                      f"val_loss={vl_loss:.4f} acc={vl_acc:.3f} f1={vl_f1:.3f} | {time.time()-t0:.1f}s")

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save({"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch}, "checkpoints/best.pt")
            console.print(f"  [green]✓ Best model saved (val_f1={vl_f1:.4f})[/green]")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["training"]["early_stopping_patience"]:
                console.print("[yellow]Early stopping triggered.[/yellow]")
                break

    console.print(f"\n[bold]Training complete. Best val F1: {best_val_f1:.4f}[/bold]")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
