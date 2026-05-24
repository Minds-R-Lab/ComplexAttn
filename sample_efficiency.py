"""Sample efficiency: RMC-simplified vs MLP-32 at progressively smaller
training set sizes. Tests the core claim of the architecture — that the
structured prior generalizes better from less data.

For each (model, dataset size) pair we train 20 epochs and report the best
validation accuracy across training. Same 5k MNIST val set throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ablations import AblationRMC
from data import get_mnist_loaders
from model import MLPBaseline
from train import train_model

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_one(train_size: int, model_factory, name: str, epochs: int, lr: float,
            seed: int = 42):
    # Batch size: small enough to get >=4 batches per epoch even at n=100
    bs = max(8, min(128, train_size // 4))
    train_loader, val_loader = get_mnist_loaders(
        data_root="./data", batch_size=bs,
        train_subset=train_size, val_subset=5000,
    )
    torch.manual_seed(seed)
    model = model_factory()
    h = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
                    name=name, save_to=None, log_every=0, grad_clip=1.0)
    return max(h["val_acc"]), h


def main():
    torch.set_num_threads(1)
    sizes = [100, 500, 2000, 10000]
    EPOCHS = 12  # converges within 12; 20 wastes compute on overfitting

    results = {
        "sizes": sizes, "epochs": EPOCHS,
        "RMC-simplified": [], "MLP-32": [],
        "RMC_history": [], "MLP_history": [],
    }

    for n in sizes:
        print(f"\n--- training set size: {n} ---")
        best_rmc, h_rmc = run_one(n,
            lambda: AblationRMC(ablation="no_B"),
            f"RMC-{n}", EPOCHS, lr=3e-3)
        best_mlp, h_mlp = run_one(n, MLPBaseline,
            f"MLP-{n}", EPOCHS, lr=2e-3)
        print(f"  RMC best val acc: {best_rmc:.4f}")
        print(f"  MLP best val acc: {best_mlp:.4f}  (gap RMC-MLP: {best_rmc - best_mlp:+.4f})")
        results["RMC-simplified"].append(best_rmc)
        results["MLP-32"].append(best_mlp)
        results["RMC_history"].append([float(v) for v in h_rmc["val_acc"]])
        results["MLP_history"].append([float(v) for v in h_mlp["val_acc"]])

    # ---- plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(sizes, results["RMC-simplified"], "o-", color="C0",
                label="RMC-simplified", linewidth=2, markersize=8)
    ax.semilogx(sizes, results["MLP-32"], "o-", color="C7",
                label="MLP-32", linewidth=2, markersize=8)
    ax.set_xlabel("training set size  (log scale)")
    ax.set_ylabel("best validation accuracy (over 20 epochs)")
    ax.set_title("Sample efficiency: RMC vs MLP on MNIST")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig_path = RESULTS_DIR / "sample_efficiency.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nsaved {fig_path}")

    (RESULTS_DIR / "sample_efficiency.json").write_text(json.dumps(results, indent=2))

    print("\n=== Sample efficiency summary ===")
    print(f"{'size':>8s}  {'RMC':>8s}  {'MLP':>8s}  {'RMC-MLP':>9s}")
    for i, n in enumerate(sizes):
        diff = results['RMC-simplified'][i] - results['MLP-32'][i]
        print(f"{n:>8d}  {results['RMC-simplified'][i]:>8.4f}  {results['MLP-32'][i]:>8.4f}  {diff:>+9.4f}")


if __name__ == "__main__":
    main()
