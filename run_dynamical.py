"""Train RMC and MLP on the synthetic 4-class trajectory task.

Includes:
- Visualization of a sample trajectory from each class.
- Head-to-head training at n_train=2000, 5000.
- Sample-efficiency sweep at n_train in {100, 300, 1000, 3000}.
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
from dynamical_data import make_dataset, make_loaders
from model import MLPBaseline, count_parameters
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
T_STEPS = 64

NAMES = ["sinusoid", "damped", "two-tone", "chirp"]


def make_rmc(input_dim: int = T_STEPS) -> AblationRMC:
    return AblationRMC(
        ablation="no_B",
        input_dim=input_dim,
        manifold_dim=12,
        num_modes=24,
        num_classes=4,
        num_steps=12,
        dt=0.15,
        potential_hidden=16,
    )


def make_mlp(input_dim: int = T_STEPS) -> MLPBaseline:
    return MLPBaseline(input_dim=input_dim, hidden_dim=32, num_classes=4)


def visualize_classes(path: Path):
    X, y = make_dataset(n_per_class=4, seed=1)
    fig, axes = plt.subplots(4, 4, figsize=(12, 6.5), sharex=True, sharey=True)
    for cls in range(4):
        examples = X[y == cls][:4]
        for j, ex in enumerate(examples):
            axes[cls, j].plot(ex, color=f"C{cls}", linewidth=1.2)
            axes[cls, j].grid(True, alpha=0.3)
            if j == 0:
                axes[cls, j].set_ylabel(NAMES[cls], fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("time step")
    fig.suptitle("Four classes of synthetic trajectories")
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"saved {path}")


def head_to_head(n_train: int, epochs: int = 15) -> dict:
    print(f"\n=== Head-to-head at n_train={n_train} ===")
    tr, va = make_loaders(n_train=n_train, n_val=500, batch_size=min(64, n_train // 4))
    torch.manual_seed(0); rmc = make_rmc()
    torch.manual_seed(0); mlp = make_mlp()
    n_rmc = sum(p.numel() for p in rmc.parameters() if p.requires_grad)
    n_mlp = count_parameters(mlp)
    print(f"  RMC trainable params: {n_rmc}   MLP params: {n_mlp}")

    h_rmc = train_model(rmc, tr, va, epochs=epochs, lr=3e-3,
                        name=f"RMC@{n_train}", save_to=None, log_every=0, grad_clip=1.0)
    h_mlp = train_model(mlp, tr, va, epochs=epochs, lr=2e-3,
                        name=f"MLP@{n_train}", save_to=None, log_every=0, grad_clip=1.0)
    return {
        "n_train": n_train, "epochs": epochs,
        "rmc_best_val_acc": float(max(h_rmc["val_acc"])),
        "mlp_best_val_acc": float(max(h_mlp["val_acc"])),
        "rmc_history": [float(v) for v in h_rmc["val_acc"]],
        "mlp_history": [float(v) for v in h_mlp["val_acc"]],
        "rmc_params": n_rmc, "mlp_params": n_mlp,
    }


def main():
    torch.set_num_threads(1)
    visualize_classes(RESULTS / "dynamical_classes.png")

    sizes = [100, 300, 1000, 3000]
    results = []
    for n in sizes:
        results.append(head_to_head(n, epochs=15))
        last = results[-1]
        print(f"  RMC={last['rmc_best_val_acc']:.4f}  MLP={last['mlp_best_val_acc']:.4f}  "
              f"gap={ (last['rmc_best_val_acc']-last['mlp_best_val_acc'])*100:+.2f}pp")

    # Sample efficiency plot
    rmc_acc = [r["rmc_best_val_acc"] for r in results]
    mlp_acc = [r["mlp_best_val_acc"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(sizes, rmc_acc, "o-", color="C0", label="RMC-simplified", linewidth=2, markersize=9)
    ax.semilogx(sizes, mlp_acc, "o-", color="C7", label="MLP-32",          linewidth=2, markersize=9)
    for n, r, m in zip(sizes, rmc_acc, mlp_acc):
        ax.annotate(f"{(r-m)*100:+.1f}pp", (n, max(r, m) + 0.02),
                    fontsize=9, color="C3", ha="center", weight="bold")
    ax.set_xlabel("training set size (log scale)")
    ax.set_ylabel("best validation accuracy")
    ax.set_title("Dynamical-system classification — sample efficiency\nred = RMC gap vs MLP in pp")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "dynamical_sample_efficiency.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'dynamical_sample_efficiency.png'}")

    (RESULTS / "dynamical_results.json").write_text(json.dumps(results, indent=2))
    print("\n=== Summary ===")
    print(f"{'n_train':>8s}  {'RMC':>8s}  {'MLP':>8s}  {'gap (pp)':>9s}")
    for r in results:
        print(f"{r['n_train']:>8d}  {r['rmc_best_val_acc']:>8.4f}  "
              f"{r['mlp_best_val_acc']:>8.4f}  {(r['rmc_best_val_acc']-r['mlp_best_val_acc'])*100:>+9.2f}")


if __name__ == "__main__":
    main()
