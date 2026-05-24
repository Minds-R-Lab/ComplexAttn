"""Multi-seed test of CoupledOscillatorRMC vs MLP-L4 (matched params) and
MLP-L1 (best non-RMC baseline on dynamical task). Saves incrementally."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from coupled_rmc import CoupledOscillatorRMC, count_trainable
from dynamical_data import make_loaders
from stacked import DeepMLP
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "coupled.json"


def load_state():
    if OUT.exists(): return json.loads(OUT.read_text())
    return {"runs": []}


def save_state(s): OUT.write_text(json.dumps(s, indent=2))


def done(state, seed, name):
    return any(r["seed"] == seed and r["model"] == name for r in state["runs"])


def run_one(seed, name, factory, lr, epochs=10):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    m = factory()
    h = train_model(m, tr, va, epochs=epochs, lr=lr, name=f"{name}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(m)


def main():
    torch.set_num_threads(1)
    seeds = [0, 1, 2]
    EPOCHS = 10
    configs = [
        ("CO-RMC", lambda: CoupledOscillatorRMC(), 2e-3),
        ("MLP-L1", lambda: DeepMLP(num_layers=1), 2e-3),
        ("MLP-L4", lambda: DeepMLP(num_layers=4), 2e-3),
    ]

    state = load_state()
    print(f"resuming with {len(state['runs'])} runs done")

    for seed in seeds:
        for name, factory, lr in configs:
            if done(state, seed, name): continue
            acc, npar = run_one(seed, name, factory, lr, EPOCHS)
            state["runs"].append({"seed": seed, "model": name, "best_val_acc": acc, "n_params": npar})
            save_state(state)
            print(f"  seed={seed} {name:8s} best={acc:.4f} (params={npar})")

    print(f"\n=== n=3000, {EPOCHS} epochs, seeds {seeds} ===")
    print(f"{'model':10s}  {'mean':>8s}  {'std':>8s}  {'per-seed':>30s}  {'params':>7s}")
    rows = []
    for name, _, _ in configs:
        vals = np.array([r["best_val_acc"] for r in state["runs"] if r["model"] == name])
        params = [r["n_params"] for r in state["runs"] if r["model"] == name]
        if len(vals) == 0: continue
        mu = vals.mean(); sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append({"name": name, "mean": float(mu), "std": float(sd),
                     "vals": vals.tolist(), "params": params[0]})
        print(f"{name:10s}  {mu:>8.4f}  {sd:>8.4f}  {str(vals.round(4).tolist()):>30s}  {params[0]:>7d}")
    state["aggregated"] = rows
    save_state(state)

    # plot — bar chart with error bars
    names = [r["name"] for r in rows]; means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]; params = [r["params"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=8,
                  color=["C0", "C7", "C8"], edgecolor="k")
    for bar, mean, std, np_ in zip(bars, means, stds, params):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.005,
                f"{mean:.3f}±{std:.3f}\n({np_} p)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.5, max(means) + 0.1)
    ax.set_ylabel("best val accuracy (n=3000)")
    ax.set_title("CoupledOscillatorRMC vs MLPs on dynamical task\n"
                 "(3 seeds, 10 epochs; CO-RMC params matched to MLP-L4)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(RESULTS / "coupled.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'coupled.png'}")


if __name__ == "__main__":
    main()
