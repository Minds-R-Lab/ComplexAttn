"""Multi-seed validation of the dynamical-system result.

For each n in [100, 300, 1000, 3000], for each seed in {0, 1, 2}, train both
the RMC-simplified and MLP-32 from a fresh init on a freshly-sampled dataset
(also seeded). Save best validation accuracy. Report mean +/- std.

Each (n, seed) run regenerates the dataset with its own seed so we vary both
weight initialization AND data sampling — that's what an honest error bar on
generalization should reflect.

Results are saved after each (n, seed, model) so a timeout doesn't lose
partial progress.
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
OUT_JSON = RESULTS / "multi_seed.json"
T_STEPS = 64


def make_rmc():
    return AblationRMC(
        ablation="no_B", input_dim=T_STEPS, manifold_dim=12, num_modes=24,
        num_classes=4, num_steps=12, dt=0.15, potential_hidden=16,
    )


def make_mlp():
    return MLPBaseline(input_dim=T_STEPS, hidden_dim=32, num_classes=4)


def load_or_init():
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text())
    return {"runs": []}


def already_done(state, n, seed, model_name):
    for r in state["runs"]:
        if r["n"] == n and r["seed"] == seed and r["model"] == model_name:
            return True
    return False


def save(state):
    OUT_JSON.write_text(json.dumps(state, indent=2))


def run_one(n, seed, model_name, factory, lr, epochs=10):
    bs = max(8, min(64, n // 4))
    tr, va = make_loaders(n_train=n, n_val=500, batch_size=bs, seed=seed)
    torch.manual_seed(seed)
    model = factory()
    h = train_model(model, tr, va, epochs=epochs, lr=lr,
                    name=f"{model_name}-n{n}-s{seed}",
                    save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"]))


def main():
    torch.set_num_threads(1)
    sizes = [100, 300, 1000, 3000]
    seeds = [0, 1, 2]
    EPOCHS = 10

    state = load_or_init()
    print(f"resuming with {len(state['runs'])} previously completed runs")

    for n in sizes:
        for seed in seeds:
            for model_name, factory, lr in [
                ("RMC", make_rmc, 3e-3),
                ("MLP", make_mlp, 2e-3),
            ]:
                if already_done(state, n, seed, model_name):
                    continue
                best = run_one(n, seed, model_name, factory, lr, epochs=EPOCHS)
                state["runs"].append({
                    "n": n, "seed": seed, "model": model_name, "best_val_acc": best,
                })
                save(state)
                print(f"  n={n:>5d}  seed={seed}  {model_name}  best_val_acc={best:.4f}")

    # ---- aggregate ----
    summary = {}
    for n in sizes:
        for model in ["RMC", "MLP"]:
            vals = [r["best_val_acc"] for r in state["runs"]
                    if r["n"] == n and r["model"] == model]
            summary[(model, n)] = vals

    print("\n=== Multi-seed summary (mean ± std over seeds) ===")
    print(f"{'n':>6s}  {'RMC mean':>10s}  {'RMC std':>9s}  "
          f"{'MLP mean':>10s}  {'MLP std':>9s}  {'gap (pp)':>10s}  {'p(RMC>MLP)':>11s}")
    aggregated = {"sizes": sizes, "RMC_mean": [], "RMC_std": [],
                  "MLP_mean": [], "MLP_std": [], "gap_pp": [], "n_seeds_completed": []}
    for n in sizes:
        rm = np.array(summary[("RMC", n)])
        mm = np.array(summary[("MLP", n)])
        nseeds = min(len(rm), len(mm))
        rm, mm = rm[:nseeds], mm[:nseeds]
        rmu, rsd = rm.mean(), rm.std(ddof=1) if nseeds > 1 else 0.0
        mmu, msd = mm.mean(), mm.std(ddof=1) if nseeds > 1 else 0.0
        gap_pp = (rmu - mmu) * 100
        # Probability RMC > MLP using paired wins
        paired_wins = float(np.mean(rm > mm)) if nseeds else 0.0
        aggregated["RMC_mean"].append(float(rmu))
        aggregated["RMC_std"].append(float(rsd))
        aggregated["MLP_mean"].append(float(mmu))
        aggregated["MLP_std"].append(float(msd))
        aggregated["gap_pp"].append(float(gap_pp))
        aggregated["n_seeds_completed"].append(nseeds)
        print(f"{n:>6d}  {rmu:>10.4f}  {rsd:>9.4f}  "
              f"{mmu:>10.4f}  {msd:>9.4f}  {gap_pp:>+10.2f}  {paired_wins:>11.2f}")

    state["aggregated"] = aggregated
    save(state)

    # ---- plot with error bars ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sizes, aggregated["RMC_mean"], yerr=aggregated["RMC_std"],
                fmt="o-", color="C0", label="RMC-simplified",
                linewidth=2, markersize=9, capsize=5)
    ax.errorbar(sizes, aggregated["MLP_mean"], yerr=aggregated["MLP_std"],
                fmt="o-", color="C7", label="MLP-32",
                linewidth=2, markersize=9, capsize=5)
    ax.set_xscale("log")
    for n, gap in zip(sizes, aggregated["gap_pp"]):
        peak = max(aggregated["RMC_mean"][sizes.index(n)] + aggregated["RMC_std"][sizes.index(n)],
                   aggregated["MLP_mean"][sizes.index(n)] + aggregated["MLP_std"][sizes.index(n)])
        ax.annotate(f"{gap:+.1f}pp", (n, peak + 0.02),
                    fontsize=9, color="C3", ha="center", weight="bold")
    ax.set_xlabel("training set size (log scale)")
    ax.set_ylabel("best validation accuracy")
    seeds_str = aggregated["n_seeds_completed"]
    title = "Dynamical-system task — mean ± std over seeds"
    title += f" (seeds completed: {seeds_str})"
    ax.set_title(title)
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "multi_seed_curves.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'multi_seed_curves.png'}")


if __name__ == "__main__":
    main()
