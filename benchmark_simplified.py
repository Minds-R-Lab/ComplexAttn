"""Benchmark the simplified RMC (no quadratic B potential) head-to-head
against MLP-32 on 5 epochs of the 15k MNIST subset.

The simplified RMC keeps the learnable mass matrix, the MLP potential, and
the resonant Fourier readout — three components instead of four. This was
the surprise winner in ablations.py.
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
from model import MLPBaseline, count_parameters
from train import train_model

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    torch.set_num_threads(1)
    EPOCHS = 5

    print("=== Loading MNIST (15k train / 5k val) ===\n")
    train_loader, val_loader = get_mnist_loaders(
        data_root="./data", batch_size=128,
        train_subset=15000, val_subset=5000,
    )

    torch.manual_seed(0)
    rmc_simple = AblationRMC(ablation="no_B")
    torch.manual_seed(0)
    mlp = MLPBaseline()

    print(f"Simplified RMC trainable params: "
          f"{sum(p.numel() for p in rmc_simple.parameters() if p.requires_grad)}")
    print(f"MLP-32 trainable params: {count_parameters(mlp)}\n")

    h_rmc = train_model(
        rmc_simple, train_loader, val_loader, epochs=EPOCHS, lr=3e-3,
        name="RMC-simplified", save_to=RESULTS_DIR / "rmc_simplified.pt",
        log_every=0,
    )
    h_mlp = train_model(
        mlp, train_loader, val_loader, epochs=EPOCHS, lr=2e-3,
        name="MLP-32", save_to=RESULTS_DIR / "mlp_simplified_bench.pt",
        log_every=0,
    )

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ep = np.arange(1, EPOCHS + 1)
    axes[0].plot(ep, h_rmc["val_loss"], "-o", color="C0",
                 label=f"RMC-simplified (final {h_rmc['val_loss'][-1]:.3f})")
    axes[0].plot(ep, h_mlp["val_loss"], "-o", color="C7",
                 label=f"MLP-32 (final {h_mlp['val_loss'][-1]:.3f})")
    axes[0].plot(ep, h_rmc["train_loss"], "--", color="C0", alpha=0.5)
    axes[0].plot(ep, h_mlp["train_loss"], "--", color="C7", alpha=0.5)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CE loss")
    axes[0].set_title("Simplified RMC vs MLP-32 — loss"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, h_rmc["val_acc"], "-o", color="C0",
                 label=f"RMC-simplified (final {h_rmc['val_acc'][-1]:.4f})")
    axes[1].plot(ep, h_mlp["val_acc"], "-o", color="C7",
                 label=f"MLP-32 (final {h_mlp['val_acc'][-1]:.4f})")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val accuracy")
    axes[1].set_title("Simplified RMC vs MLP-32 — accuracy"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "simplified_vs_mlp.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nsaved {fig_path}")

    summary = {
        "epochs": EPOCHS,
        "RMC-simplified": {
            "history": {k: [float(x) for x in v]
                        for k, v in h_rmc.items() if isinstance(v, list)},
            "final_val_acc": float(h_rmc["val_acc"][-1]),
            "final_val_loss": float(h_rmc["val_loss"][-1]),
            "total_time_s": float(h_rmc["total_time_s"]),
            "trainable_params": sum(p.numel() for p in rmc_simple.parameters() if p.requires_grad),
        },
        "MLP-32": {
            "history": {k: [float(x) for x in v]
                        for k, v in h_mlp.items() if isinstance(v, list)},
            "final_val_acc": float(h_mlp["val_acc"][-1]),
            "final_val_loss": float(h_mlp["val_loss"][-1]),
            "total_time_s": float(h_mlp["total_time_s"]),
            "trainable_params": count_parameters(mlp),
        },
    }
    (RESULTS_DIR / "simplified_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Result ===")
    print(f"  RMC-simplified  val_acc={h_rmc['val_acc'][-1]:.4f}  val_loss={h_rmc['val_loss'][-1]:.4f}")
    print(f"  MLP-32          val_acc={h_mlp['val_acc'][-1]:.4f}  val_loss={h_mlp['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
