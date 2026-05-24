"""Test whether RMC's slower convergence vs MLP closes the gap given more
epochs. At n=3000 only, train both models for 15 epochs across 4 seeds.

Saves incremental JSON so partial progress survives timeouts.
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
from dynamical_data import make_loaders
from model import MLPBaseline
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT_JSON = RESULTS / "multi_seed_long.json"
T_STEPS = 64


def make_rmc():
    return AblationRMC(
        ablation="no_B", input_dim=T_STEPS, manifold_dim=12, num_modes=24,
        num_classes=4, num_steps=12, dt=0.15, potential_hidden=16,
    )


def make_mlp():
    return MLPBaseline(input_dim=T_STEPS, hidden_dim=32, num_classes=4)


def load_state():
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text())
    return {"runs": []}


def save_state(state):
    OUT_JSON.write_text(json.dumps(state, indent=2))


def done(state, seed, model_name):
    return any(r["seed"] == seed and r["model"] == model_name for r in state["runs"])


def run_one(seed, model_name, factory, lr, epochs):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    model = factory()
    h = train_model(model, tr, va, epochs=epochs, lr=lr,
                    name=f"{model_name}-s{seed}", save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), [float(v) for v in h["val_acc"]]


def main():
    torch.set_num_threads(1)
    seeds = [0, 1, 2, 3]
    EPOCHS = 15

    state = load_state()
    print(f"resuming with {len(state['runs'])} runs done")

    for seed in seeds:
        for model_name, factory, lr in [
            ("RMC", make_rmc, 3e-3),
            ("MLP", make_mlp, 2e-3),
        ]:
            if done(state, seed, model_name):
                continue
            best, curve = run_one(seed, model_name, factory, lr, EPOCHS)
            state["runs"].append({
                "seed": seed, "model": model_name,
                "best_val_acc": best, "val_curve": curve,
            })
            save_state(state)
            print(f"  seed={seed} {model_name} best={best:.4f}  "
                  f"last={curve[-1]:.4f}")

    # ---- aggregate ----
    rmc_vals = [r["best_val_acc"] for r in state["runs"] if r["model"] == "RMC"]
    mlp_vals = [r["best_val_acc"] for r in state["runs"] if r["model"] == "MLP"]
    n_done = min(len(rmc_vals), len(mlp_vals))
    rmc_arr = np.array(rmc_vals[:n_done])
    mlp_arr = np.array(mlp_vals[:n_done])

    print(f"\n=== n=3000, 15 epochs, {n_done} seeds ===")
    print(f"RMC: {rmc_arr.mean():.4f} ± {rmc_arr.std(ddof=1):.4f}    "
          f"(per seed: {rmc_arr.round(4)})")
    print(f"MLP: {mlp_arr.mean():.4f} ± {mlp_arr.std(ddof=1):.4f}    "
          f"(per seed: {mlp_arr.round(4)})")
    gap_pp = (rmc_arr.mean() - mlp_arr.mean()) * 100
    paired_diff = rmc_arr - mlp_arr
    print(f"gap (RMC-MLP): {gap_pp:+.2f} pp")
    print(f"paired diffs: {(paired_diff*100).round(2)} pp")
    print(f"RMC wins paired comparisons: {int((paired_diff > 0).sum())}/{n_done}")

    state["aggregated"] = {
        "n_train": 3000, "epochs": EPOCHS, "n_seeds": n_done,
        "rmc_mean": float(rmc_arr.mean()), "rmc_std": float(rmc_arr.std(ddof=1)),
        "mlp_mean": float(mlp_arr.mean()), "mlp_std": float(mlp_arr.std(ddof=1)),
        "gap_pp": float(gap_pp),
        "rmc_per_seed": rmc_arr.tolist(),
        "mlp_per_seed": mlp_arr.tolist(),
        "paired_rmc_wins": int((paired_diff > 0).sum()),
    }
    save_state(state)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: per-seed val-acc curves
    for r in state["runs"]:
        color = "C0" if r["model"] == "RMC" else "C7"
        alpha = 0.5
        axes[0].plot(range(1, len(r["val_curve"]) + 1), r["val_curve"],
                     color=color, alpha=alpha, linewidth=1.2)
    # mean curve
    rmc_curves = np.array([r["val_curve"] for r in state["runs"] if r["model"] == "RMC"])
    mlp_curves = np.array([r["val_curve"] for r in state["runs"] if r["model"] == "MLP"])
    if rmc_curves.size and mlp_curves.size:
        n_done = min(len(rmc_curves), len(mlp_curves))
        ep_axis = np.arange(1, rmc_curves.shape[1] + 1)
        axes[0].plot(ep_axis, rmc_curves[:n_done].mean(axis=0), color="C0",
                     linewidth=2.5, label=f"RMC mean ({n_done} seeds)")
        axes[0].plot(ep_axis, mlp_curves[:n_done].mean(axis=0), color="C7",
                     linewidth=2.5, label=f"MLP mean ({n_done} seeds)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("val accuracy")
    axes[0].set_title(f"n=3000, {EPOCHS} epochs — per-seed and mean")
    axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)

    # Right: scatter of per-seed best for paired comparison
    if rmc_curves.size and mlp_curves.size:
        n_done = min(len(rmc_arr), len(mlp_arr))
        axes[1].scatter(mlp_arr[:n_done], rmc_arr[:n_done], s=80, color="C0", zorder=3)
        lo, hi = min(mlp_arr.min(), rmc_arr.min()) - 0.02, max(mlp_arr.max(), rmc_arr.max()) + 0.02
        axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        for i, (mv, rv) in enumerate(zip(mlp_arr[:n_done], rmc_arr[:n_done])):
            axes[1].annotate(f"s{i}", (mv, rv), textcoords="offset points",
                             xytext=(6, -2), fontsize=9)
        axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
        axes[1].set_xlabel("MLP best val acc")
        axes[1].set_ylabel("RMC best val acc")
        axes[1].set_title("Paired comparison (above line = RMC wins)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "multi_seed_long.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'multi_seed_long.png'}")


if __name__ == "__main__":
    main()
