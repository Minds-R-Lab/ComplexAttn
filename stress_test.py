"""Stress test for MRBNet vs DeepMLP.

Runs three experiments designed to confirm or refute the multi-seed positive
result from run_mrb.py. Use a machine faster than a single CPU thread — this
will take ~10-30 minutes depending on your hardware.

Experiments:
  1. Dynamical task, 8 seeds × 20 epochs at L=1, 2, 4. Tightens the error bars
     and removes the "10 epochs may not be converged" caveat.
  2. Dynamical task sample-efficiency at n_train ∈ {300, 1000, 3000, 10000},
     5 seeds × 15 epochs, L=2 (the best depth for MRB).
  3. MNIST head-to-head at matched params, 3 seeds × 8 epochs on a 15k subset.
     This is the *out-of-domain* test — MRB was designed/tuned on dynamical
     data, so MNIST is a fair generalization check.

Saves incrementally to results/stress_test.json so timeouts are recoverable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import get_mnist_loaders
from dynamical_data import make_loaders as dyn_loaders
from mrb_net import MRBNet, count_trainable
from model import MLPBaseline
from stacked import DeepMLP
from train import train_model

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "stress_test.json"


def load_state():
    return json.loads(OUT.read_text()) if OUT.exists() else {"runs": []}

def save_state(s):
    OUT.write_text(json.dumps(s, indent=2))

def done(state, experiment, **keys):
    for r in state["runs"]:
        if r["experiment"] != experiment: continue
        if all(r.get(k) == v for k, v in keys.items()): return True
    return False


def train_with_seed(make_model, loaders, seed, lr, epochs, name):
    tr, va = loaders
    torch.manual_seed(seed)
    m = make_model()
    h = train_model(m, tr, va, epochs=epochs, lr=lr,
                    name=name, save_to=None, log_every=0, grad_clip=1.0)
    return float(max(h["val_acc"])), count_trainable(m)


# ---------------------------------------------------------------------------
# Experiment 1: dynamical task, more seeds and more epochs
# ---------------------------------------------------------------------------
def experiment_1_dynamical_robustness(state):
    print("\n=== Experiment 1: dynamical task, 8 seeds × 20 epochs ===")
    Ls = [1, 2, 4]; seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    EPOCHS = 20
    for L in Ls:
        for seed in seeds:
            for model_name, factory, lr in [
                ("MRB", lambda L=L: MRBNet(num_layers=L), 2e-3),
                ("MLP", lambda L=L: DeepMLP(num_layers=L), 2e-3),
            ]:
                if done(state, "exp1", L=L, seed=seed, model=model_name): continue
                loaders = dyn_loaders(n_train=3000, n_val=500, batch_size=64, seed=seed)
                acc, npar = train_with_seed(factory, loaders, seed, lr, EPOCHS,
                                            f"E1-{model_name}-L{L}-s{seed}")
                state["runs"].append({"experiment": "exp1", "L": L, "seed": seed,
                                       "model": model_name, "best_val_acc": acc,
                                       "n_params": npar})
                save_state(state)
                print(f"  L={L} seed={seed} {model_name} acc={acc:.4f}")


# ---------------------------------------------------------------------------
# Experiment 2: dynamical task sample efficiency (L=2)
# ---------------------------------------------------------------------------
def experiment_2_sample_efficiency(state):
    print("\n=== Experiment 2: sample efficiency at L=2, 5 seeds × 15 epochs ===")
    sizes = [300, 1000, 3000, 10000]
    seeds = [0, 1, 2, 3, 4]
    EPOCHS = 15
    L = 2
    for n in sizes:
        for seed in seeds:
            for model_name, factory, lr in [
                ("MRB", lambda: MRBNet(num_layers=L), 2e-3),
                ("MLP", lambda: DeepMLP(num_layers=L), 2e-3),
            ]:
                if done(state, "exp2", n=n, seed=seed, model=model_name): continue
                bs = max(16, min(64, n // 8))
                loaders = dyn_loaders(n_train=n, n_val=500, batch_size=bs, seed=seed)
                acc, npar = train_with_seed(factory, loaders, seed, lr, EPOCHS,
                                            f"E2-{model_name}-n{n}-s{seed}")
                state["runs"].append({"experiment": "exp2", "n": n, "seed": seed,
                                       "model": model_name, "best_val_acc": acc,
                                       "n_params": npar})
                save_state(state)
                print(f"  n={n} seed={seed} {model_name} acc={acc:.4f}")


# ---------------------------------------------------------------------------
# Experiment 3: MNIST sanity check (out-of-domain for MRB)
# ---------------------------------------------------------------------------
def experiment_3_mnist(state):
    print("\n=== Experiment 3: MNIST 15k subset, 3 seeds × 8 epochs ===")
    seeds = [0, 1, 2]; EPOCHS = 8
    for seed in seeds:
        for model_name, factory, lr in [
            ("MRB", lambda: MRBNet(input_dim=784, num_layers=2, dim=32, hidden_dim=24, num_classes=10), 2e-3),
            ("MLP", lambda: MLPBaseline(input_dim=784, hidden_dim=32, num_classes=10), 2e-3),
        ]:
            if done(state, "exp3", seed=seed, model=model_name): continue
            loaders = get_mnist_loaders(data_root="./data", batch_size=128,
                                         train_subset=15000, val_subset=5000, seed=seed)
            acc, npar = train_with_seed(factory, loaders, seed, lr, EPOCHS,
                                        f"E3-{model_name}-s{seed}")
            state["runs"].append({"experiment": "exp3", "seed": seed,
                                   "model": model_name, "best_val_acc": acc,
                                   "n_params": npar})
            save_state(state)
            print(f"  seed={seed} {model_name} acc={acc:.4f}")


# ---------------------------------------------------------------------------
# Summary + plots
# ---------------------------------------------------------------------------
def summarize_and_plot(state):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # -- Experiment 1 --
    print("\nExperiment 1 — dynamical task, multi-seed × 20 epochs:")
    print(f"{'L':>3s}  {'MRB mean±std':>16s}  {'MLP mean±std':>16s}  {'gap':>8s}  {'wins':>6s}")
    exp1_summary = []
    for L in [1, 2, 4]:
        mrb = np.array([r["best_val_acc"] for r in state["runs"]
                        if r["experiment"]=="exp1" and r["L"]==L and r["model"]=="MRB"])
        mlp = np.array([r["best_val_acc"] for r in state["runs"]
                        if r["experiment"]=="exp1" and r["L"]==L and r["model"]=="MLP"])
        n = min(len(mrb), len(mlp))
        if n == 0: continue
        mrb, mlp = mrb[:n], mlp[:n]
        rmu, rsd = mrb.mean(), mrb.std(ddof=1) if n>1 else 0.0
        mmu, msd = mlp.mean(), mlp.std(ddof=1) if n>1 else 0.0
        gap = (rmu-mmu)*100; wins = int((mrb>mlp).sum())
        print(f"{L:>3d}  {rmu:.4f}±{rsd:.4f}  {mmu:.4f}±{msd:.4f}  {gap:>+7.2f}pp  {wins}/{n}")
        exp1_summary.append({"L": L, "mrb_mean": float(rmu), "mrb_std": float(rsd),
                             "mlp_mean": float(mmu), "mlp_std": float(msd),
                             "wins": wins, "n_seeds": n})

    # -- Experiment 2 --
    print("\nExperiment 2 — sample efficiency at L=2:")
    print(f"{'n':>6s}  {'MRB mean±std':>16s}  {'MLP mean±std':>16s}  {'gap':>8s}  {'wins':>6s}")
    exp2_summary = []
    for n_train in [300, 1000, 3000, 10000]:
        mrb = np.array([r["best_val_acc"] for r in state["runs"]
                        if r["experiment"]=="exp2" and r["n"]==n_train and r["model"]=="MRB"])
        mlp = np.array([r["best_val_acc"] for r in state["runs"]
                        if r["experiment"]=="exp2" and r["n"]==n_train and r["model"]=="MLP"])
        n_s = min(len(mrb), len(mlp))
        if n_s == 0: continue
        mrb, mlp = mrb[:n_s], mlp[:n_s]
        rmu, rsd = mrb.mean(), mrb.std(ddof=1) if n_s>1 else 0.0
        mmu, msd = mlp.mean(), mlp.std(ddof=1) if n_s>1 else 0.0
        gap = (rmu-mmu)*100; wins = int((mrb>mlp).sum())
        print(f"{n_train:>6d}  {rmu:.4f}±{rsd:.4f}  {mmu:.4f}±{msd:.4f}  {gap:>+7.2f}pp  {wins}/{n_s}")
        exp2_summary.append({"n": n_train, "mrb_mean": float(rmu), "mrb_std": float(rsd),
                             "mlp_mean": float(mmu), "mlp_std": float(msd),
                             "wins": wins, "n_seeds": n_s})

    # -- Experiment 3 --
    print("\nExperiment 3 — MNIST sanity check:")
    mrb = np.array([r["best_val_acc"] for r in state["runs"]
                    if r["experiment"]=="exp3" and r["model"]=="MRB"])
    mlp = np.array([r["best_val_acc"] for r in state["runs"]
                    if r["experiment"]=="exp3" and r["model"]=="MLP"])
    if len(mrb) > 0 and len(mlp) > 0:
        n_s = min(len(mrb), len(mlp))
        mrb, mlp = mrb[:n_s], mlp[:n_s]
        rmu, rsd = mrb.mean(), mrb.std(ddof=1) if n_s>1 else 0.0
        mmu, msd = mlp.mean(), mlp.std(ddof=1) if n_s>1 else 0.0
        gap = (rmu-mmu)*100; wins = int((mrb>mlp).sum())
        print(f"  MRB={rmu:.4f}±{rsd:.4f}  MLP={mmu:.4f}±{msd:.4f}  gap={gap:+.2f}pp  wins={wins}/{n_s}")

    # -- Plot all results --
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if exp1_summary:
        Ls = [s["L"] for s in exp1_summary]
        rmu = [s["mrb_mean"] for s in exp1_summary]; rsd = [s["mrb_std"] for s in exp1_summary]
        mmu = [s["mlp_mean"] for s in exp1_summary]; msd = [s["mlp_std"] for s in exp1_summary]
        axes[0].errorbar(Ls, rmu, yerr=rsd, fmt="o-", color="C0", label="MRBNet",
                         linewidth=2, markersize=10, capsize=6)
        axes[0].errorbar(Ls, mmu, yerr=msd, fmt="o-", color="C7", label="DeepMLP",
                         linewidth=2, markersize=10, capsize=6)
        axes[0].set_xticks(Ls); axes[0].set_xlabel("L"); axes[0].set_ylabel("val acc")
        axes[0].set_title("E1: dynamical task, 8 seeds × 20 epochs")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if exp2_summary:
        ns = [s["n"] for s in exp2_summary]
        rmu = [s["mrb_mean"] for s in exp2_summary]; rsd = [s["mrb_std"] for s in exp2_summary]
        mmu = [s["mlp_mean"] for s in exp2_summary]; msd = [s["mlp_std"] for s in exp2_summary]
        axes[1].errorbar(ns, rmu, yerr=rsd, fmt="o-", color="C0", label="MRBNet",
                         linewidth=2, markersize=10, capsize=6)
        axes[1].errorbar(ns, mmu, yerr=msd, fmt="o-", color="C7", label="DeepMLP",
                         linewidth=2, markersize=10, capsize=6)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("n_train (log)"); axes[1].set_ylabel("val acc")
        axes[1].set_title("E2: sample efficiency at L=2")
        axes[1].legend(); axes[1].grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "stress_test.png", dpi=130); plt.close(fig)
    print(f"\nsaved {RESULTS / 'stress_test.png'}")

    state["summary"] = {"exp1": exp1_summary, "exp2": exp2_summary}
    save_state(state)


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    state = load_state()
    print(f"resuming with {len(state['runs'])} runs done")

    experiment_1_dynamical_robustness(state)
    experiment_2_sample_efficiency(state)
    experiment_3_mnist(state)
    summarize_and_plot(state)


if __name__ == "__main__":
    main()
