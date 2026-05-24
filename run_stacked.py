"""Depth × multi-seed experiment.

For each L in {1, 2, 4} and seed in {0, 1, 2}, train StackedRMC and DeepMLP
on the dynamical-system task at n=3000 for 10 epochs. Save best validation
accuracy incrementally so timeouts are recoverable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dynamical_data import make_loaders
from stacked import StackedRMC, DeepMLP, count_trainable
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT_JSON = RESULTS / "stacked.json"


def load_state():
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text())
    return {"runs": []}


def save_state(state):
    OUT_JSON.write_text(json.dumps(state, indent=2))


def done(state, L, seed, model_name):
    return any(r["L"] == L and r["seed"] == seed and r["model"] == model_name
               for r in state["runs"])


def run_one(L, seed, model_name, epochs):
    tr, va = make_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
    torch.manual_seed(seed)
    if model_name == "RMC":
        model = StackedRMC(num_layers=L)
        lr = 3e-3
    else:
        model = DeepMLP(num_layers=L)
        lr = 2e-3
    h = train_model(model, tr, va, epochs=epochs, lr=lr,
                    name=f"{model_name}-L{L}-s{seed}", save_to=None,
                    log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(model)


def main():
    torch.set_num_threads(1)
    Ls = [1, 2, 4]
    seeds = [0, 1, 2]
    EPOCHS = 10

    state = load_state()
    print(f"resuming with {len(state['runs'])} runs done")

    for L in Ls:
        for seed in seeds:
            for model_name in ["RMC", "MLP"]:
                if done(state, L, seed, model_name):
                    continue
                best, n_params = run_one(L, seed, model_name, EPOCHS)
                state["runs"].append({
                    "L": L, "seed": seed, "model": model_name,
                    "best_val_acc": best, "n_params": n_params,
                })
                save_state(state)
                print(f"  L={L} seed={seed} {model_name} best={best:.4f} (params={n_params})")

    # ---- aggregate ----
    summary = {"L": Ls,
               "RMC_mean": [], "RMC_std": [],
               "MLP_mean": [], "MLP_std": [],
               "RMC_params": [], "MLP_params": [],
               "RMC_wins": [], "n_seeds": []}

    print(f"\n=== Depth × seed summary, n=3000, {EPOCHS} epochs ===")
    print(f"{'L':>3s}  {'RMC mean':>10s}  {'RMC std':>9s}  "
          f"{'MLP mean':>10s}  {'MLP std':>9s}  {'gap pp':>8s}  {'RMC wins':>10s}")
    for L in Ls:
        rmc = np.array([r["best_val_acc"] for r in state["runs"]
                        if r["L"] == L and r["model"] == "RMC"])
        mlp = np.array([r["best_val_acc"] for r in state["runs"]
                        if r["L"] == L and r["model"] == "MLP"])
        ns = min(len(rmc), len(mlp))
        rmc, mlp = rmc[:ns], mlp[:ns]
        if ns == 0:
            continue
        rmu = rmc.mean(); rsd = rmc.std(ddof=1) if ns > 1 else 0.0
        mmu = mlp.mean(); msd = mlp.std(ddof=1) if ns > 1 else 0.0
        gap = (rmu - mmu) * 100
        wins = int((rmc > mlp).sum())
        rp = [r["n_params"] for r in state["runs"] if r["L"] == L and r["model"] == "RMC"][0]
        mp = [r["n_params"] for r in state["runs"] if r["L"] == L and r["model"] == "MLP"][0]
        summary["RMC_mean"].append(float(rmu)); summary["RMC_std"].append(float(rsd))
        summary["MLP_mean"].append(float(mmu)); summary["MLP_std"].append(float(msd))
        summary["RMC_params"].append(rp);       summary["MLP_params"].append(mp)
        summary["RMC_wins"].append(wins);       summary["n_seeds"].append(ns)
        print(f"{L:>3d}  {rmu:>10.4f}  {rsd:>9.4f}  "
              f"{mmu:>10.4f}  {msd:>9.4f}  {gap:>+8.2f}  {wins:>3d}/{ns}")

    state["aggregated"] = summary
    save_state(state)

    # ---- plot ----
    Ls_done = summary["L"][:len(summary["RMC_mean"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ls_done, summary["RMC_mean"], yerr=summary["RMC_std"],
                fmt="o-", color="C0", label="StackedRMC", linewidth=2,
                markersize=10, capsize=6)
    ax.errorbar(Ls_done, summary["MLP_mean"], yerr=summary["MLP_std"],
                fmt="o-", color="C7", label="DeepMLP", linewidth=2,
                markersize=10, capsize=6)
    for i, L in enumerate(Ls_done):
        gap = (summary["RMC_mean"][i] - summary["MLP_mean"][i]) * 100
        peak = max(summary["RMC_mean"][i] + summary["RMC_std"][i],
                   summary["MLP_mean"][i] + summary["MLP_std"][i])
        ax.annotate(f"{gap:+.1f}pp", (L, peak + 0.015),
                    fontsize=9, color="C3", ha="center", weight="bold")
    ax.set_xlabel("number of layers L")
    ax.set_ylabel("best val accuracy (n=3000)")
    ax.set_xticks(Ls_done)
    ax.set_title("Depth × multi-seed: StackedRMC vs DeepMLP\n"
                 "(dynamical-system task, 10 epochs, 3 seeds)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "stacked_depth.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'stacked_depth.png'}")


if __name__ == "__main__":
    main()
