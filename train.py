"""Training pipeline for RMC and baselines.

Trains a model on MNIST, tracks train/val loss + accuracy each epoch,
returns a history dict and saves a checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn


def evaluate(model: nn.Module, loader, device="cpu") -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += crit(logits, y).item()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += y.numel()
    return total_loss / total_n, total_correct / total_n


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 3,
    lr: float = 2e-3,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    device: str = "cpu",
    log_every: int = 100,
    name: str = "model",
    save_to: str | Path | None = None,
) -> dict:
    """Train and return a history dict."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    history = {
        "name": name,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_s": [],
    }

    t_total = time.time()
    print(f"\n[{name}] training {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        t0 = time.time()
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running_loss += loss.item() * y.numel()
            running_correct += (logits.argmax(dim=-1) == y).sum().item()
            running_n += y.numel()

            if log_every and (it + 1) % log_every == 0:
                print(f"  [{name}] epoch {epoch} iter {it+1} "
                      f"loss={running_loss/running_n:.4f} acc={running_correct/running_n:.4f}")

        train_loss = running_loss / running_n
        train_acc = running_correct / running_n
        val_loss, val_acc = evaluate(model, val_loader, device=device)
        dt = time.time() - t0
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_s"].append(dt)
        print(f"[{name}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)")

    history["total_time_s"] = time.time() - t_total

    if save_to is not None:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "history": history}, save_path)
        print(f"[{name}] saved checkpoint -> {save_path}")

    return history
