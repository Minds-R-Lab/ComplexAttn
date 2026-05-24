"""Ablations of the Resonant Manifold Cell.

Trains four variants of the RMC at matched parameter count to identify which
architectural components actually contribute:

    full          — baseline RMC with all components active
    no_V_MLP      — anharmonic potential frozen at zero (harmonic dynamics only)
    no_B          — quadratic potential frozen at zero (MLP-only potential)
    no_resonance  — bypass windowed Fourier; use final position x_T projected
                    onto the same ψ directions as features (no time integration
                    in the readout — selectivity comes from x_T alone)

For each variant, every other component is held constant. Frozen parameters
remain in the model (so the parameter count is identical) but receive no
gradient updates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data import get_mnist_loaders
from model import RMCClassifier, MLPBaseline, count_parameters
from train import train_model

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class AblationRMC(RMCClassifier):
    """RMCClassifier with a switch for ablating one component at a time."""

    VALID = {"full", "no_V_MLP", "no_B", "no_resonance"}

    def __init__(self, ablation: str = "full", **kwargs):
        assert ablation in self.VALID, f"unknown ablation: {ablation}"
        super().__init__(**kwargs)
        self.ablation = ablation

        if ablation == "no_V_MLP":
            # Zero and freeze the MLP potential — pure harmonic dynamics.
            with torch.no_grad():
                for p in self.rmc.V_mlp.parameters():
                    p.zero_()
            for p in self.rmc.V_mlp.parameters():
                p.requires_grad = False
            self.rmc.mlp_potential_scale = 0.0  # skip its compute

        elif ablation == "no_B":
            # Zero and freeze the quadratic potential — pure MLP dynamics.
            with torch.no_grad():
                self.rmc.B_raw.zero_()
            self.rmc.B_raw.requires_grad = False

        # no_resonance is handled by overriding forward (no param change).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablation != "no_resonance":
            return super().forward(x)

        # No-resonance variant: integrate as usual, then project the FINAL
        # position x_T onto ψ directions. Same parameters, no time-Fourier.
        x = x.view(x.size(0), -1)
        enc = self.encoder(x)
        x0, p0 = enc.chunk(2, dim=-1)
        xs, _ = self.rmc.integrate_with_trajectory(x0, p0)
        x_T = xs[-1]                          # (B, d)
        features = x_T @ self.rmc.psi.T       # (B, K)
        return self.head(features)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)

    print("=== Loading MNIST (15k train / 5k val) ===\n")
    train_loader, val_loader = get_mnist_loaders(
        data_root="./data",
        batch_size=128,
        train_subset=15000,
        val_subset=5000,
    )

    EPOCHS = 3  # 4 variants x 5 epochs is too slow on 1-thread CPU
    ablations = ["full", "no_V_MLP", "no_B", "no_resonance"]

    histories: list[dict] = []
    for ab in ablations:
        torch.manual_seed(0)  # same init across variants
        model = AblationRMC(ablation=ab)
        n_params = count_parameters(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[{ab}] params total={n_params} trainable={n_trainable}")
        h = train_model(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=3e-3,
            name=f"RMC-{ab}",
            save_to=RESULTS_DIR / f"ablation_{ab}.pt",
            log_every=0,
        )
        h["ablation"] = ab
        h["params_total"] = n_params
        h["params_trainable"] = n_trainable
        histories.append(h)

    # Run an MLP-32 baseline for context.
    torch.manual_seed(0)
    mlp = MLPBaseline()
    h_mlp = train_model(
        mlp, train_loader, val_loader, epochs=EPOCHS, lr=2e-3,
        name="MLP-32 (ref)", save_to=RESULTS_DIR / "ablation_mlp.pt", log_every=0,
    )
    h_mlp["ablation"] = "mlp_ref"
    h_mlp["params_total"] = count_parameters(mlp)
    histories.append(h_mlp)

    # --- comparison plot --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"full": "C0", "no_V_MLP": "C1", "no_B": "C2",
              "no_resonance": "C3", "mlp_ref": "C7"}
    for h in histories:
        epochs = np.arange(1, len(h["val_loss"]) + 1)
        c = colors.get(h["ablation"], "k")
        ls = "--" if h["ablation"] == "mlp_ref" else "-"
        axes[0].plot(epochs, h["val_loss"], "-o", color=c, linestyle=ls, label=h["name"])
        axes[1].plot(epochs, h["val_acc"], "-o", color=c, linestyle=ls, label=h["name"])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val CE loss")
    axes[0].set_title("Ablation: validation loss"); axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val accuracy")
    axes[1].set_title("Ablation: validation accuracy"); axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = RESULTS_DIR / "ablations.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {fig_path}")

    # --- summary table ----------------------------------------------------
    print("\n=== Ablation summary (val accuracy / loss at final epoch) ===")
    rows = []
    for h in histories:
        rows.append({
            "ablation": h["ablation"],
            "name": h["name"],
            "final_val_acc": float(h["val_acc"][-1]),
            "final_val_loss": float(h["val_loss"][-1]),
            "params_trainable": h.get("params_trainable", h.get("params_total")),
            "total_time_s": float(h["total_time_s"]),
        })
        print(f"  {h['name']:20s}  acc={h['val_acc'][-1]:.4f}  "
              f"loss={h['val_loss'][-1]:.4f}  "
              f"trainable={h.get('params_trainable', '-')}  "
              f"time={h['total_time_s']:.1f}s")

    summary = {"epochs": EPOCHS, "results": rows}
    (RESULTS_DIR / "ablations.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {RESULTS_DIR / 'ablations.json'}")


if __name__ == "__main__":
    main()
